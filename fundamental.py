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
    """au10001 명세서 기준 토큰 발급"""
    auth_url = f"{BASE_URL}/oauth2/token"
    # 명세서상 Body 파라미터 확인: grant_type, appkey, secretkey
    payload = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }
    headers = {"Content-Type": "application/json;charset=UTF-8"}
    
    try:
        res = requests.post(auth_url, json=payload, headers=headers)
        auth_data = res.json()
        
        # ❗ 명세서상 응답 키는 'token'입니다.
        token = auth_data.get("token")
        if not token:
            print(f"🚨 토큰 발급 실패: {auth_data}")
            return None
        return token
    except Exception as e:
        print(f"🚨 인증 요청 중 에러: {e}")
        return None

def collect_fundamental(stock_code):
    token = get_access_token()
    if not token:
        return

    # ka10001: 주식기본정보요청
    url = f"{BASE_URL}/api/dostk/stkinfo"
    
    # ❗ 명세서 [Header] 부분: "Bearer " 접두어와 공백 필수 반영
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}", 
        "api-id": "ka10001"
    }
    
    # ❗ 명세서 [Body] 부분: stk_cd 필드명 확인
    body = {"stk_cd": stock_code}

    try:
        res = requests.post(url, json=body, headers=headers)
        data = res.json()
        
        # 인증 에러(8005) 발생 시 상세 리포트
        if data.get("return_code") != 0:
            print(f"❌ API 호출 실패: {data.get('return_msg')}")
            return

        # 데이터 매핑 (명세서 필드명 기준)
        fund_data = {
            "stk_cd": stock_code,
            "per": data.get("per"),
            "pbr": data.get("pbr"),
            "roe": data.get("roe"),
            "eps": data.get("eps"),
            "bps": data.get("bps"),
            "cap": data.get("cap") # 시가총액(억)
        }
        
        df = pd.DataFrame([fund_data])
        
        # 숫자 전처리: 부호(+,-), 콤마(,), 공백 제거 후 float 변환
        for col in ['per', 'pbr', 'roe', 'eps', 'bps', 'cap']:
            val = str(df[col].iloc[0]).strip()
            if val == "None" or val == "":
                df[col] = 0.0
            else:
                clean_val = val.replace(',', '').replace('+', '').replace('-', '')
                df[col] = float(clean_val) if clean_val else 0.0
        
        df.to_csv(f"{stock_code}_fundamental.csv", index=False)
        print(f"✅ Fundamental 수집 성공: {stock_code}_fundamental.csv")
        print(df)

    except Exception as e:
        print(f"🚨 수집 중 에러: {e}")

if __name__ == "__main__":
    collect_fundamental("000660")