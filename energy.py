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
    try:
        res = requests.post(auth_url, json={
            "grant_type": "client_credentials",
            "appkey": APP_KEY,
            "secretkey": APP_SECRET
        })
        token = res.json().get("token")
        if token:
            print("✅ 토큰 발급 성공")
            return token
        else:
            print(f"❌ 토큰 발급 실패: {res.json()}")
            return None
    except Exception as e:
        print(f"🚨 인증 오류: {e}")
        return None

def collect_energy(stock_code):
    print(f"🚀 {stock_code} 수급 데이터 수집 시작...")
    token = get_access_token()
    if not token: return

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}"
    }
    body = {"stk_cd": stock_code}

    # 1. 외국인 수급 (ka10008)
    print("📡 외국인 데이터(ka10008) 요청 중...")
    headers["api-id"] = "ka10008"
    res_f = requests.post(f"{BASE_URL}/api/dostk/frgnistt", json=body, headers=headers)
    data_f = res_f.json().get("stk_frgnr", [])
    
    df_f = pd.DataFrame(data_f)
    if not df_f.empty:
        df_f = df_f[['dt', 'chg_qty']]
        df_f.columns = ['dt', 'frgn_net_qty']
        print(f"✅ 외국인 데이터 {len(df_f)}건 수신")
    else:
        print("⚠️ 외국인 데이터가 비어있습니다.")
        # 에러 방지를 위해 컬럼 구조 유지
        df_f = pd.DataFrame(columns=['dt', 'frgn_net_qty'])

    # 2. 기관 수급 (ka10009)
    print("📡 기관 데이터(ka10009) 요청 중...")
    headers["api-id"] = "ka10009"
    res_o = requests.post(f"{BASE_URL}/api/dostk/frgnistt", json=body, headers=headers)
    
    # ka10009 응답에서 데이터 리스트 탐색 (키가 다를 수 있음)
    res_o_json = res_o.json()
    data_o = res_o_json.get("stk_frgn_trde_trend", res_o_json.get("stk_frgnr", []))
    
    df_o = pd.DataFrame(data_o)
    if not df_o.empty and 'orgn_netprps_qty' in df_o.columns:
        df_o = df_o[['dt', 'orgn_netprps_qty']]
        df_o.columns = ['dt', 'orgn_net_qty']
        print(f"✅ 기관 데이터 {len(df_o)}건 수신")
    else:
        print("⚠️ 기관 데이터가 비어있거나 컬럼이 없습니다.")
        # ❗ 중요: 병합 에러 방지를 위해 'dt' 컬럼이 포함된 빈 DF 생성
        df_o = pd.DataFrame(columns=['dt', 'orgn_net_qty'])

    # 3. 데이터 병합
    if df_f.empty and df_o.empty:
        print("🚨 수집된 데이터가 하나도 없어 파일을 만들지 않습니다.")
        return

    print("📊 데이터 병합 및 전처리 중...")
    # 'dt' 컬럼이 양쪽에 존재하므로 이제 에러가 나지 않습니다.
    df_energy = pd.merge(df_f, df_o, on='dt', how='outer').fillna('0')

    # 숫자 전처리
    for col in ['frgn_net_qty', 'orgn_net_qty']:
        if col in df_energy.columns:
            df_energy[col] = df_energy[col].astype(str).str.replace('+', '', regex=False) \
                                           .str.replace('-', '', regex=False) \
                                           .str.replace(',', '', regex=False) \
                                           .str.strip()
            df_energy[col] = pd.to_numeric(df_energy[col], errors='coerce').fillna(0).astype('float32')

    # 4. 저장
    file_name = f"{stock_code}_energy.csv"
    df_energy.sort_values(by='dt', ascending=False).to_csv(file_name, index=False)
    print(f"💾 최종 저장 완료: {os.getcwd()}/{file_name}")

if __name__ == "__main__":
    collect_energy("000660")