import pandas as pd
import numpy as np
import os

def merge_stock_data(stock_code):
    print(f"🔄 {stock_code} 데이터 통합 시작...")
    
    # 1. 파일 로드
    price_file = f"{stock_code}_price.csv"
    energy_file = f"{stock_code}_energy.csv"
    fund_file = f"{stock_code}_fundamental.csv"
    
    if not all(os.path.exists(f) for f in [price_file, energy_file, fund_file]):
        print("🚨 필수 CSV 파일 중 일부가 없습니다. 수집 상태를 확인하세요.")
        return

    df_price = pd.read_csv(price_file)
    df_energy = pd.read_csv(energy_file)
    df_fund = pd.read_csv(fund_file)

    # 2. 시계열 데이터 병합 (Price + Energy)
    # 날짜(dt)를 기준으로 교집합(inner) 병합
    final_df = pd.merge(df_price, df_energy, on='dt', how='inner')
    
    # 3. 정적 데이터 결합 (Fundamental)
    # 재무 지표는 현재 시점의 값을 모든 날짜에 동일하게 부여 (Static Feature)
    for col in ['per', 'pbr', 'roe', 'eps', 'bps', 'cap']:
        final_df[col] = df_fund[col].iloc[0]

    # 4. M1 최적화: 데이터 타입 변환 및 정렬
    # dt를 제외한 모든 컬럼을 float32로 변환 (MPS 가속 시 연산 속도 향상)
    final_df['dt'] = pd.to_datetime(final_df['dt'].astype(str))
    final_df = final_df.sort_values(by='dt').reset_index(drop=True)
    
    numeric_cols = final_df.columns.difference(['dt'])
    final_df[numeric_cols] = final_df[numeric_cols].astype('float32')

    # 5. 최종 파일 저장
    output_name = f"{stock_code}_final_train.csv"
    final_df.to_csv(output_name, index=False)
    
    print("-" * 30)
    print(f"✅ 통합 완료: {output_name}")
    print(f"📊 최종 피처 수: {len(final_df.columns)}개")
    print(f"📅 데이터 기간: {final_df['dt'].min()} ~ {final_df['dt'].max()}")
    print(final_df.tail(3))

if __name__ == "__main__":
    merge_stock_data("000660")