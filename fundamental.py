import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    res = requests.post(auth_url, json={"grant_type": "client_credentials", "appkey": APP_KEY, "secretkey": APP_SECRET})
    return res.json().get("token")

def preprocess_fundamental(df):
    print("🛠 Fundamental 데이터 최적화 중...")
    cols = ['per', 'pbr', 'roe', 'eps', 'bps', 'cap']
    for col in cols:
        val = str(df[col].iloc[0]).replace(',', '').replace('+', '').replace('-', '').strip()
        df[col] = float(val) if val not in ["None", ""] else 0.0
    
    # 스케일 압축 (큰 단위 지표 대상 로그 변환)
    for col in ['eps', 'bps', 'cap']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    return df.astype({c: 'float32' for c in df.columns if c != 'stk_cd'})

def collect_fundamental(stock_code):
    token = get_access_token()
    headers = {"Content-Type": "application/json", "authorization": f"Bearer {token}", "api-id": "ka10001"}
    body = {"stk_cd": stock_code}

    res = requests.post(f"{BASE_URL}/api/dostk/stkinfo", json=body, headers=headers)
    data = res.json()
    
    fund_data = { "stk_cd": stock_code, "per": data.get("per"), "pbr": data.get("pbr"), 
                 "roe": data.get("roe"), "eps": data.get("eps"), "bps": data.get("bps"), "cap": data.get("cap") }
    
    df = preprocess_fundamental(pd.DataFrame([fund_data]))
    df.to_csv(f"{stock_code}_fundamental.csv", index=False)
    print(f"✅ Fundamental 저장 완료")

if __name__ == "__main__":
    collect_fundamental("000660")