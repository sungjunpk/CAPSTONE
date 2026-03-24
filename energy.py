import requests
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
from pathlib import Path

# 1. 경로 설정 (상대 경로 방식)
# 현재 파일(energy.py)의 위치를 기준으로 프로젝트 루트를 잡습니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"

# csv 폴더가 없으면 자동으로 생성
if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

load_dotenv()
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

MAX_DAYS = 600
TOP_10_STOCKS = ["005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "105560", "035420"]

def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    res = requests.post(auth_url, json={
        "grant_type": "client_credentials", 
        "appkey": APP_KEY, 
        "secretkey": APP_SECRET
    })
    return res.json().get("token")

def collect_energy(stock_code, dt, token):
    url = f"{BASE_URL}/api/dostk/chart"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {token}", 
        "api-id": "ka10060"
    }
    body = {
        "dt": dt, 
        "stk_cd": stock_code, 
        "amt_qty_tp": "2", 
        "trde_tp": "0", 
        "unit_tp": "1"
    }
    
    all_data, seen_dates, cont_yn, next_key = [], set(), "N", ""
    
    for _ in range(30):
        headers["cont-yn"], headers["next-key"] = cont_yn, next_key
        res = requests.post(url, json=body, headers=headers)
        
        # API 응답 결과 확인
        res_json = res.json()
        chunk = res_json.get("stk_invsr_orgn_chart", [])
        
        if not chunk:
            break
            
        for row in chunk:
            if row['dt'] not in seen_dates:
                seen_dates.add(row['dt'])
                all_data.append(row)
        
        # 반복 종료 조건 (데이터 수 충족 혹은 다음 데이터 없음)
        if len(seen_dates) >= MAX_DAYS or res.headers.get("cont-yn") == "N":
            break
            
        cont_yn, next_key = res.headers.get("cont-yn"), res.headers.get("next-key")
        time.sleep(0.2)
    
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data)[["dt", "frgnr_invsr", "orgn"]]
    df.columns = ["dt", "frgn_net_qty", "orgn_net_qty"]
    
    for col in ["frgn_net_qty", "orgn_net_qty"]:
        # 콤마 제거 및 숫자 변환
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype('float32')
    
    df['stk_cd'] = stock_code
    
    # 개별 종목 파일 저장 (상대 경로 사용)
    df.to_csv(DATA_DIR / f"{stock_code}_energy.csv", index=False)
    return df

if __name__ == "__main__":
    token = get_access_token()
    if not token:
        print("🚨 Access Token을 받지 못했습니다. .env 설정을 확인하세요.")
    else:
        all_collected = []
        # 오늘 날짜를 기준으로 수집하려면 dt 값을 유동적으로 조절하세요.
        # 예: dt = time.strftime("%Y%m%d")
        for code in TOP_10_STOCKS:
            df = collect_energy(code, "20260324", token)
            if df is not None:
                all_collected.append(df)
                print(f"✅ {code} Energy 데이터 수집 완료")
            time.sleep(0.5)
            
        if all_collected:
            total_path = DATA_DIR / "total_top10_energy.csv"
            pd.concat(all_collected).to_csv(total_path, index=False)
            print(f"\n🚀 전체 Energy 데이터 통합 완료: {total_path}")