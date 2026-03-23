import requests
import pandas as pd
import json
import os # 시스템 환경변수 접근용
from dotenv import load_dotenv # .env 파일 로드용

# [1] .env 파일 로드
load_dotenv()

# [2] 환경변수에서 키값 읽기
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

def get_daily_data(stock_code, base_dt):
    # 키가 제대로 로드되었는지 확인 (디버깅용)
    if not APP_KEY or not APP_SECRET:
        print("🚨 .env 파일에서 키를 찾을 수 없습니다!")
        return

    # 1. 토큰 발급 (au10001)
    auth_url = f"{BASE_URL}/oauth2/token"
    auth_body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }
    
    # ... (이하 기존 코드와 동일) ...
    auth_res = requests.post(auth_url, json=auth_body)
    token = auth_res.json().get("token")
    
    if not token:
        print(f"🚨 인증 실패: {auth_res.json()}")
        return

    # 2. 일봉 차트 조회 (ka10081)
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

    try:
        res = requests.post(url, json=body, headers=headers)
        data = res.json()
        raw_list = data.get("stk_dt_pole_chart_qry", [])
        
        if raw_list:
            df = pd.DataFrame(raw_list)
            for col in ['cur_prc', 'trde_qty', 'open_pric', 'high_pric', 'low_pric']:
                df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)
            
            print(f"✅ {stock_code} 데이터 수집 완료 ({len(df)}건)")
            df.to_csv(f"{stock_code}_train.csv", index=False)
        else:
            print("⚠️ 데이터를 찾을 수 없습니다.")
    except Exception as e:
        print(f"🚨 에러: {e}")

if __name__ == "__main__":
    get_daily_data("000660", "20260323")