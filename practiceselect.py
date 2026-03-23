import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv

# [1] .env 파일 로드
load_dotenv()

# [2] 환경변수에서 키값 읽기
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

def get_access_token():
    """접근토큰 발급 (au10001)"""
    url = f"{BASE_URL}/oauth2/token"
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }
    res = requests.post(url, json=body)
    return res.json().get("token")

def get_daily_data(stock_code, base_dt):
    if not APP_KEY or not APP_SECRET:
        print("🚨 .env 파일에서 키를 찾을 수 없습니다!")
        return

    token = get_access_token()
    if not token:
        print("🚨 토큰 발급 실패")
        return

    # [STEP 1] 주식기본정보요청 (ka10001) - 재무 지표 수집
    info_url = f"{BASE_URL}/api/dostk/stkinfo"
    info_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10001"
    }
    info_body = {"stk_cd": stock_code}
    
    fundamental_data = {}
    try:
        info_res = requests.post(info_url, json=info_body, headers=info_headers)
        info_json = info_res.json()
        
        # 문서 명세에 따른 필드 추출
        fundamental_data = {
            "per": info_json.get("per"),
            "pbr": info_json.get("pbr"),
            "roe": info_json.get("roe"),
            "eps": info_json.get("eps"),
            "bps": info_json.get("bps"),
            "cap": info_json.get("cap") # 시가총액
        }
        # 부호 제거 및 숫자 변환
        for key, value in fundamental_data.items():
            if value:
                fundamental_data[key] = float(str(value).replace('+', '').replace('-', '').replace(',', ''))
            else:
                fundamental_data[key] = 0.0
    except Exception as e:
        print(f"⚠️ 재무 정보 수집 중 에러: {e}")

    # [STEP 2] 일봉 차트 조회 (ka10081) - 가격 데이터 수집
    chart_url = f"{BASE_URL}/api/dostk/chart"
    chart_headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10081"
    }
    chart_body = {
        "stk_cd": stock_code,
        "base_dt": base_dt,
        "upd_stkpc_tp": "1"
    }

    try:
        res = requests.post(chart_url, json=chart_body, headers=chart_headers)
        data = res.json()
        raw_list = data.get("stk_dt_pole_chart_qry", [])
        
        if raw_list:
            df = pd.DataFrame(raw_list)
            # 기존 가격 데이터 전처리
            for col in ['cur_prc', 'trde_qty', 'open_pric', 'high_pric', 'low_pric']:
                df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)
            
            # [STEP 3] 재무 지표 컬럼 추가
            for key, value in fundamental_data.items():
                df[key] = value
            
            print(f"✅ {stock_code} 데이터 및 재무 지표 수집 완료 ({len(df)}건)")
            # 컬럼 순서 정리 (날짜, 가격정보..., 재무정보...)
            cols = ['dt', 'open_pric', 'high_pric', 'low_pric', 'cur_prc', 'trde_qty', 'per', 'pbr', 'roe', 'eps', 'bps', 'cap']
            df = df[cols]
            
            df.to_csv(f"{stock_code}_train.csv", index=False)
            print(f"💾 {stock_code}_train.csv 저장 완료")
        else:
            print("⚠️ 차트 데이터를 찾을 수 없습니다.")
    except Exception as e:
        print(f"🚨 에러: {e}")

if __name__ == "__main__":
    # SK하이닉스(000660) 데이터 수집
    get_daily_data("000660", "20260323")