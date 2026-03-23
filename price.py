import requests
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    res = requests.post(auth_url, json={
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    })
    return res.json().get("token")

def preprocess_price_data(df):
    """성준님이 요청한 3대 전처리 로직 적용"""
    print("🛠 Price 데이터 전처리 시작...")
    
    # [1] 날짜 정렬 및 형식 변환 (과거 -> 미래 오름차순)
    df['dt'] = pd.to_datetime(df['dt'].astype(str))
    df = df.sort_values(by='dt').reset_index(drop=True)
    
    # [2] 수익률 변환 (Log Returns)
    # 단순 가격은 추세(Trend)가 있어 모델이 패턴을 학습하기 어렵습니다.
    # log(현재가/이전날종가)를 통해 변동폭 중심으로 정규화합니다.
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # [3] 거래량 로그 스케일링 (Log Scaling)
    # 거래량은 이상치가 많으므로 분포를 압축합니다. (+1은 거래량 0인 경우 방지)
    df['volume_log'] = np.log1p(df['volume'])
    
    # 결측치 처리 (shift 연산으로 발생한 첫 번째 행의 NaN 제거)
    df = df.dropna().reset_index(drop=True)
    
    return df

def collect_price(stock_code, base_dt):
    token = get_access_token()
    url = f"{BASE_URL}/api/dostk/chart"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10081"
    }
    body = {
        "stk_cd": stock_code,
        "base_dt": base_dt,
        "upd_stkpc_tp": "1"
    }

    res = requests.post(url, json=body, headers=headers)
    raw_list = res.json().get("stk_dt_pole_chart_qry", [])
    
    if raw_list:
        df = pd.DataFrame(raw_list)
        # 기본 컬럼 추출 및 숫자 변환
        df = df[['dt', 'open_pric', 'high_pric', 'low_pric', 'cur_prc', 'trde_qty']]
        df.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(str).str.replace('+', '', regex=False).str.replace('-', '', regex=False).astype(float)
        
        # ❗ 성준님이 요청한 전처리 로직 호출
        df = preprocess_price_data(df)
        
        # 결과 저장 (M1 최적화를 위해 float32 지정)
        numeric_cols = df.columns.difference(['dt'])
        df[numeric_cols] = df[numeric_cols].astype('float32')
        
        df.to_csv(f"{stock_code}_price.csv", index=False)
        print(f"✅ 전처리 완료 후 {len(df)}건 저장됨")
        print(df[['dt', 'close', 'log_return', 'volume_log']].tail())
    else:
        print("⚠️ 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    collect_price("000660", "20260323")