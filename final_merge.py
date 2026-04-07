import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1. 경로 및 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)

TARGET_ASSETS = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "042700": "한미반도체",
    "058470": "리노공업",
    "240810": "원익IPS",
    "039030": "이오테크닉스",
    "000990": "DB하이텍",
    "403870": "HPSP",
    "357780": "솔브레인",
    "005290": "동진쎄미켐",
}

# =========================
# 2. 파일 찾기
# =========================
def find_file(stock_code, suffix):
    candidates = list(DATA_DIR.glob(f"{stock_code}*_{suffix}.csv"))
    return candidates[0] if candidates else None


# =========================
# 3. 기술적 지표 계산 (Leakage 방지)
# =========================
def add_technical_indicators(df):
    """Train/Test 구분 없이 호출되지만, 호출 전에 temporal split이 되어 있어야 함"""
    df = df.copy()
    df = df.sort_values("dt").reset_index(drop=True)
    
    # 이동평균 수익률
    df["ma5_return"] = df["log_return"].rolling(5).mean()
    df["ma20_return"] = df["log_return"].rolling(20).mean()
    
    # 변동성
    df["volatility_5"] = df["log_return"].rolling(5).std()
    df["volatility_20"] = df["log_return"].rolling(20).std()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    
    # 모멘텀
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["roc_10"] = df["close"].pct_change(10) * 100
    
    # 거래량 지표
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    
    return df


# =========================
# 4. 개별 종목 처리 (핵심 수정 부분)
# =========================
def process_single_stock(stock_code, stock_name, train_ratio=0.8):
    p = find_file(stock_code, "price")
    e = find_file(stock_code, "energy")
    f = find_file(stock_code, "fundamental")
    
    if not all([p, e, f]):
        print(f"⚠️ {stock_code} {stock_name} 파일 부족")
        return None
    
    # 데이터 로드
    df_p = pd.read_csv(p)
    df_e = pd.read_csv(e)
    df_f = pd.read_csv(f)
    
    # 날짜 및 코드 정리
    df_p["dt"] = pd.to_datetime(df_p["dt"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
    df_e["dt"] = pd.to_datetime(df_e["dt"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
    
    for d in [df_p, df_e, df_f]:
        d["stk_cd"] = d["stk_cd"].astype(str).str.zfill(6)
    
    df_p = df_p.sort_values("dt").reset_index(drop=True)
    
    # ==================== Temporal Split (Leakage 방지) ====================
    split_idx = int(len(df_p) * train_ratio)
    train_p = df_p.iloc[:split_idx].copy()
    test_p = df_p.iloc[split_idx:].copy()
    
    # Train 데이터로 기술적 지표 계산
    train_p = add_technical_indicators(train_p)
    
    # Test 데이터 기술적 지표 계산 (Train 통계량으로 보정)
    test_p = add_technical_indicators(test_p)
    
    # Test 구간의 초기 NaN을 Train 마지막 값으로 forward fill
    for col in ['ma5_return', 'ma20_return', 'volatility_5', 'volatility_20',
                'rsi_14', 'bb_width', 'volume_ma20', 'volume_ratio']:
        if col in train_p.columns:
            test_p[col] = test_p[col].fillna(train_p[col].iloc[-20:].mean())  # 최근 20일 평균으로 보정
    
    # Train + Test 다시 합치기
    df_p = pd.concat([train_p, test_p], ignore_index=True).sort_values("dt").reset_index(drop=True)
    
    # ==================== Energy Merge ====================
    final_df = pd.merge(
        df_p,
        df_e[["dt", "stk_cd", "frgn_net_qty", "orgn_net_qty"]],
        on=["dt", "stk_cd"],
        how="left"
    )
    
    final_df[["frgn_net_qty", "orgn_net_qty"]] = final_df[["frgn_net_qty", "orgn_net_qty"]].ffill().fillna(0)
    
    # ==================== Fundamental Merge ====================
    fund_cols = ["stk_cd", "per", "pbr", "roe", "eps", "bps", "cap",
                 "eps_log", "bps_log", "cap_log", "value_score"]
    fund_data = df_f[[c for c in fund_cols if c in df_f.columns]].drop_duplicates(subset=["stk_cd"])
    final_df = pd.merge(final_df, fund_data, on="stk_cd", how="left")
    
    # ==================== 마무리 ====================
    final_df["stk_name"] = stock_name
    final_df = final_df.fillna(0)
    
    print(f"✅ {stock_code} {stock_name} 처리 완료 | 행수: {len(final_df)}")
    return final_df


# =========================
# 5. 전체 병합
# =========================
def merge_all_stocks():
    all_dfs = []
    
    for code, name in TARGET_ASSETS.items():
        result = process_single_stock(code, name, train_ratio=0.8)
        if result is not None:
            all_dfs.append(result)
    
    if not all_dfs:
        print("❌ 병합할 데이터가 없습니다.")
        return None
    
    total_df = pd.concat(all_dfs, ignore_index=True)
    total_df = total_df.sort_values(["stk_cd", "dt"]).reset_index(drop=True)
    
    output_path = DATA_DIR / "capstone_semiconductor_train_data_fixed.csv"
    total_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n🎉 최종 병합 완료!")
    print(f"   파일 저장: {output_path}")
    print(f"   총 행수: {len(total_df):,} 행")
    print(f"   종목 수: {total_df['stk_cd'].nunique()}개")
    
    return output_path


if __name__ == "__main__":
    merge_all_stocks()