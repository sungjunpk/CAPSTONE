import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# ============================== [0] 경로 설정 (상대 경로 방식) ==============================
# 현재 파일(price.py)이 있는 폴더를 기준으로 프로젝트 루트를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"

# csv 폴더가 존재하지 않으면 자동으로 생성
if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

load_dotenv()
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

TOP_10_STOCKS = ["005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "105560", "035420"]

def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    res = requests.post(auth_url, json={
        "grant_type": "client_credentials", 
        "appkey": APP_KEY, 
        "secretkey": APP_SECRET
    })
    return res.json().get("token")

def preprocess_price_data(df, stock_code):
    # 날짜 형식 변환 (dt를 문자열에서 datetime 객체로)
    df['dt'] = pd.to_datetime(df['dt'].astype(str))
    df = df.sort_values(by='dt').reset_index(drop=True)
    
    # 로그 수익률 및 거래량 로그 변환 (수치 안정성 확보)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volume_log'] = np.log1p(df['volume'])
    df['stk_cd'] = stock_code
    
    return df.dropna().reset_index(drop=True)

def collect_price(stock_code, base_dt, token):
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
        df = pd.DataFrame(raw_list)[['dt', 'open_pric', 'high_pric', 'low_pric', 'cur_prc', 'trde_qty']]
        df.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']
        
        # 기호(+, -) 제거 및 숫자형 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(str).str.replace('+', '', regex=False).str.replace('-', '', regex=False).astype(float)
            
        df = preprocess_price_data(df, stock_code)
        
        # 개별 종목 파일 저장 (상대 경로 활용)
        save_path = DATA_DIR / f"{stock_code}_price.csv"
        df.to_csv(save_path, index=False)
        return df
    return None

if __name__ == "__main__":
    token = get_access_token()
    if not token:
        print("🚨 Access Token을 받지 못했습니다. .env 설정을 확인하세요.")
    else:
        all_data = []
        for code in TOP_10_STOCKS:
            # 2026-03-24 기준 데이터 수집
            df = collect_price(code, "20260324", token)
            if df is not None:
                all_data.append(df)
                print(f"✅ {code} Price 수집 완료")
            time.sleep(0.5)
            
        if all_data:
            total_path = DATA_DIR / "total_top10_price.csv"
            pd.concat(all_data).to_csv(total_path, index=False)
            print(f"\n🚀 전체 Price 데이터 통합 완료: {total_path}")