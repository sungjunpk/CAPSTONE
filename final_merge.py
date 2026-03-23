import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def create_final_train_data(stock_code):
    print(f"🔄 {stock_code} 3대 요소 최종 병합 시작...")
    
    # 1. 파일 로드
    df_price = pd.read_csv(f"{stock_code}_price.csv")
    df_energy = pd.read_csv(f"{stock_code}_energy.csv")
    df_fund = pd.read_csv(f"{stock_code}_fundamental.csv")
    
    # 2. 날짜 형식 통일
    df_price['dt'] = pd.to_datetime(df_price['dt']).dt.strftime('%Y%m%d')
    df_energy['dt'] = df_energy['dt'].astype(str)
    
    # 3. [에러 방지] 필수 기술적 지표 재계산 (없을 경우를 대비)
    df_price = df_price.sort_values('dt').reset_index(drop=True)
    if 'log_return' not in df_price.columns:
        df_price['log_return'] = np.log(df_price['close'] / df_price['close'].shift(1))
    
    # KeyError가 났던 지표들을 여기서 직접 생성합니다.
    df_price['ma5_return'] = df_price['log_return'].rolling(window=5).mean()
    df_price['ma20_return'] = df_price['log_return'].rolling(window=20).mean()
    
    # 4. 시계열 데이터 병합 (Price + Energy)
    final_df = pd.merge(df_price, df_energy, on='dt', how='inner')
    
    # 5. 가치(Fundamental) 데이터 결합
    # 기존 log 컬럼이 없을 경우를 대비해 기본 컬럼 사용
    fund_cols = ['per', 'pbr', 'roe']
    for col in fund_cols:
        final_df[col] = df_fund[col].iloc[0]
    
    # 6. 학습용 피처 리스트 확정 (실제 존재하는 컬럼들로만 구성)
    features = [
        'log_return', 'volume_log', 'ma5_return', 'ma20_return',           # 현상
        'frgn_net_qty', 'frgn_net_diff', 'frgn_ma5', 'frgn_ma20',         # 에너지(외인)
        'orgn_net_qty', 'orgn_net_diff', 'orgn_ma5', 'orgn_ma20',         # 에너지(기관)
        'per', 'pbr'                                                      # 가치
    ]
    
    # 결측치 제거 (이동평균으로 인해 발생한 초기 행들 제거)
    final_df = final_df.dropna(subset=features).reset_index(drop=True)
    
    # 7. 전체 데이터 정규화 (Scaling)
    scaler = MinMaxScaler()
    final_df[features] = scaler.fit_transform(final_df[features])
    
    # 8. M1 최적화: float32 변환 및 저장
    final_df[features] = final_df[features].astype('float32')
    
    output_file = f"{stock_code}_final_train.csv"
    final_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"✅ 최종 학습 데이터 생성 완료: {output_file}")
    print(f"📊 최종 데이터 형상: {final_df.shape} (행, 열)")
    print(f"💡 모델 주입 준비 완료: {len(features)}개 피처")

if __name__ == "__main__":
    create_final_train_data("000660")