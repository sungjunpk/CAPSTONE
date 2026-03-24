import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# 1. 경로 설정 (상대 경로 방식)
# 현재 파일(fundamental.py)의 위치를 기준으로 프로젝트 루트를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"

# csv 폴더가 없으면 자동으로 생성
if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

load_dotenv()
# .env 파일도 BASE_DIR에 있다고 가정하고 로드할 수 있습니다.
# load_dotenv(BASE_DIR / ".env")

APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

TOP_10_STOCKS = ["005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "105560", "035420"]

def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    # 실제 키움 API 명세에 따라 요청 필드명(appkey, secretkey 등)을 확인하세요.
    res = requests.post(auth_url, json={
        "grant_type": "client_credentials", 
        "appkey": APP_KEY, 
        "secretkey": APP_SECRET
    })
    return res.json().get("token")

def collect_fundamental(stock_code, token):
    url = f"{BASE_URL}/api/dostk/stkinfo"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {token}", 
        "api-id": "ka10001"
    }
    
    try:
        res = requests.post(url, json={"stk_cd": stock_code}, headers=headers)
        data = res.json()
        
        fund_data = {
            "stk_cd": stock_code, 
            "per": data.get("per"), 
            "pbr": data.get("pbr"), 
            "roe": data.get("roe"), 
            "eps": data.get("eps"), 
            "bps": data.get("bps"), 
            "cap": data.get("cap")
        }
        
        df = pd.DataFrame([fund_data])
        
        # 숫자 전처리 및 로그 변환
        for col in ['eps', 'bps', 'cap']:
            val = str(df[col].iloc[0]).replace(',', '')
            numeric_val = pd.to_numeric(val, errors='coerce') or 0
            df[f'{col}_log'] = np.log1p(numeric_val)
            
        # 파일 저장 (상대 경로 활용)
        save_path = DATA_DIR / f"{stock_code}_fundamental.csv"
        df.to_csv(save_path, index=False)
        return df
    except Exception as e:
        print(f"❌ {stock_code} 수집 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    token = get_access_token()
    if not token:
        print("🚨 Access Token을 받지 못했습니다. .env 파일과 API 설정을 확인하세요.")
    else:
        all_fund = []
        for code in TOP_10_STOCKS:
            df = collect_fundamental(code, token)
            if df is not None:
                all_fund.append(df)
            time.sleep(0.3)
            
        if all_fund:
            total_save_path = DATA_DIR / "total_top10_fundamental.csv"
            pd.concat(all_fund).to_csv(total_save_path, index=False)
            print(f"✅ Fundamental 수집 완료: {total_save_path}")