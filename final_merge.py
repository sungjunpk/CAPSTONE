import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# ============================== [0] 경로 설정 (상대 경로 방식) ==============================
# 현재 파일의 위치를 기준으로 프로젝트 루트와 csv 폴더를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"

# 데이터 저장을 위한 폴더 확인
if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

TOP_10_STOCKS = ["005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "105560", "035420"]

def process_single_stock(stock_code):
    # 파일 경로 설정 (Path 객체 활용)
    p = DATA_DIR / f"{stock_code}_price.csv"
    e = DATA_DIR / f"{stock_code}_energy.csv"
    f = DATA_DIR / f"{stock_code}_fundamental.csv"
    
    # 파일 존재 여부 확인
    if not all(x.exists() for x in [p, e, f]):
        print(f"⚠️ {stock_code}: 데이터 파일이 부족하여 건너뜜 (Price: {p.exists()}, Energy: {e.exists()}, Fund: {f.exists()})")
        return None
    
    # 데이터 로드
    df_p = pd.read_csv(p)
    df_e = pd.read_csv(e)
    df_f = pd.read_csv(f)
    
    # 날짜 형식 통일 (병합을 위해 문자열 포맷 일치)
    df_p['dt'] = pd.to_datetime(df_p['dt']).dt.strftime('%Y%m%d')
    df_e['dt'] = df_e['dt'].astype(str)
    
    # 종목 코드 포맷 통일 (000660 형태)
    df_p['stk_cd'] = df_p['stk_cd'].astype(str).str.zfill(6)
    df_e['stk_cd'] = df_e['stk_cd'].astype(str).str.zfill(6)
    
    # 이동평균 피처 생성 (기술적 지표 추가)
    df_p = df_p.sort_values('dt').reset_index(drop=True)
    df_p['ma5_return'] = df_p['log_return'].rolling(5).mean()
    df_p['ma20_return'] = df_p['log_return'].rolling(20).mean()
    
    # [Price + Energy] 내부 병합 (날짜와 종목코드가 모두 있는 데이터만 남김)
    final_df = pd.merge(df_p, df_e, on=['dt', 'stk_cd'], how='inner')
    
    # [Fundamental] 재무 데이터 병합 (모든 행에 해당 종목의 재무 수치 할당)
    fund_cols = ['per', 'pbr', 'roe', 'eps_log', 'bps_log', 'cap_log']
    for col in fund_cols:
        if col in df_f.columns:
            # 첫 번째 행의 값을 모든 행에 복사 (재무 데이터는 시계열이 아니므로)
            final_df[col] = df_f[col].iloc[0]
            
    # 정규화 대상 피처 (입력값의 범위를 0~1 사이로 맞춤)
    scaling_cols = [
        'log_return', 'volume_log', 'ma5_return', 'ma20_return', 
        'frgn_net_qty', 'orgn_net_qty'
    ]
    
    # 결측치(이동평균 계산 시 발생하는 NaN 등) 제거
    final_df = final_df.dropna(subset=scaling_cols).reset_index(drop=True)
    
    if len(final_df) > 0:
        scaler = MinMaxScaler()
        # 데이터가 0인 경우를 대비해 replace(inf, 0) 처리 추가 가능
        final_df[scaling_cols] = scaler.fit_transform(final_df[scaling_cols])
        return final_df
    return None

def merge_all_stocks():
    all_dfs = []
    print(f"🔄 {DATA_DIR} 폴더 내 데이터 병합 시작...")
    
    for code in TOP_10_STOCKS:
        res = process_single_stock(code)
        if res is not None:
            all_dfs.append(res)
            print(f"✅ {code} 병합 성공 ({len(res)}행)")
    
    if all_dfs:
        total_df = pd.concat(all_dfs, ignore_index=True)
        
        # 수치형 데이터 float32 변환 (메모리 절약 및 MPS 가속 최적화)
        num_cols = total_df.columns.difference(['dt', 'stk_cd'])
        total_df[num_cols] = total_df[num_cols].astype('float32')
        
        # 최종 결과 저장
        output_path = DATA_DIR / "total_top10_train_data.csv"
        total_df.to_csv(output_path, index=False)
        
        print("-" * 40)
        print(f"🎉 통합 완료! 파일 저장 경로: {output_path}")
        print(f"📊 총 데이터 수: {len(total_df)}행")
    else:
        print("🚨 병합할 수 있는 데이터가 없습니다. price.py, energy.py 등을 먼저 실행하세요.")

if __name__ == "__main__":
    merge_all_stocks()