import requests
import pandas as pd
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
        # 필요한 컬럼만 추출 및 이름 변경
        df = df[['dt', 'open_pric', 'high_pric', 'low_pric', 'cur_prc', 'trde_qty']]
        df.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']
        
        # 부호 제거 및 숫자 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)
        
        df.to_csv(f"{stock_code}_price.csv", index=False)
        print(f"✅ Price 수집 완료: {len(df)}건 저장됨")
    else:
        print("⚠️ Price 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    # 실행 시 종목코드와 기준일자 입력
    collect_price("000660", "20260323")