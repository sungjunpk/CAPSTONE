import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def create_final_train_data(stock_code):
    print(f"🔄 {stock_code} 3대 요소 최종 병합 및 정규화 시작...")
    
    # 1. 파일 로드
    price_file = f"{stock_code}_price.csv"
    energy_file = f"{stock_code}_energy.csv"
    fund_file = f"{stock_code}_fundamental.csv"
    
    if not all(os.path.exists(f) for f in [price_file, energy_file, fund_file]):
        print("🚨 필수 CSV 파일이 누락되었습니다. 수집 상태를 확인하세요.")
        return

    df_price = pd.read_csv(price_file)
    df_energy = pd.read_csv(energy_file)
    df_fund = pd.read_csv(fund_file)
    
    # 2. 날짜 형식 통일 (YYYYMMDD 문자열로 통일)
    df_price['dt'] = pd.to_datetime(df_price['dt']).dt.strftime('%Y%m%d')
    df_energy['dt'] = df_energy['dt'].astype(str)
    
    # 3. [보강] 시계열 지표 재계산 (과거->미래 정렬 필수)
    df_price = df_price.sort_values('dt').reset_index(drop=True)
    if 'log_return' not in df_price.columns:
        df_price['log_return'] = np.log(df_price['close'] / df_price['close'].shift(1))
    
    df_price['ma5_return'] = df_price['log_return'].rolling(window=5).mean()
    df_price['ma20_return'] = df_price['log_return'].rolling(window=20).mean()
    
    # 4. 시계열 데이터 병합 (Price + Energy)
    final_df = pd.merge(df_price, df_energy, on='dt', how='inner')
    
    # 5. 가치(Fundamental) 데이터 결합
    # 시계열이 아닌 정적 피처는 병합 후 모든 행에 동일하게 복사됩니다.
    fund_cols = ['per', 'pbr', 'roe', 'eps_log', 'bps_log', 'cap_log']
    for col in fund_cols:
        if col in df_fund.columns:
            final_df[col] = df_fund[col].iloc[0]
    
    # 6. 피처 그룹 분리 (중요!)
    # 매일 값이 변하여 0~1 스케일링이 필요한 항목
    scaling_features = [
        'log_return', 'volume_log', 'ma5_return', 'ma20_return',
        'frgn_net_qty', 'frgn_net_diff', 'frgn_ma5', 'frgn_ma20',
        'orgn_net_qty', 'orgn_net_diff', 'orgn_ma5', 'orgn_ma20'
    ]
    
    # 단일 종목에서 값이 변하지 않아 스케일링하면 0이 되어버리는 항목 (그대로 유지)
    static_features = [col for col in fund_cols if col in final_df.columns]
    
    # 결측치 제거 (이동평균 등으로 발생한 NaN 행 삭제)
    final_df = final_df.dropna(subset=scaling_features).reset_index(drop=True)
    
    # 7. 시계열 데이터만 정규화 (MinMax Scaling)
    scaler = MinMaxScaler()
    final_df[scaling_features] = scaler.fit_transform(final_df[scaling_features])
    
    # 8. M1 맥북 GPU(MPS) 최적화: 모든 수치를 float32로 변환
    all_numeric_cols = scaling_features + static_features
    final_df[all_numeric_cols] = final_df[all_numeric_cols].astype('float32')
    
    # 최종 결과물 저장
    output_file = f"{stock_code}_final_train.csv"
    final_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"✅ 최종 학습 데이터 생성 완료: {output_file}")
    print(f"📊 데이터 형상: {final_df.shape}")
    print(f"📌 재무 데이터 보존 확인 (PER): {final_df['per'].iloc[0]}")
    print(f"💡 모델 학습 준비 완료 (총 {len(all_numeric_cols)}개 피처)")

if __name__ == "__main__":
    create_final_train_data("000660")