import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# ============================== [0] 경로 ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()

APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

# ============================== [1] 종목 ==============================
TARGET_ASSETS = {
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "042700": "한미반도체",
    "058470": "리노공업",
    "240810": "원익IPS",
    "039030": "이오테크닉스",
    "000990": "DB하이텍",
    "403870": "HPSP",
    "357780": "솔브레인",
    "005290": "동진쎄미켐",
}

# ============================== [2] 토큰 ==============================
def get_access_token():
    url = f"{BASE_URL}/oauth2/token"

    payload = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()

        token = data.get("token")
        if not token:
            print("🚨 토큰 없음:", data)
            return None

        return token

    except Exception as e:
        print("🚨 토큰 발급 실패:", e)
        return None


# ============================== [3] 숫자 정리 ==============================
def clean_numeric(value):
    if value is None:
        return 0.0
    return pd.to_numeric(
        str(value).replace(",", ""),
        errors="coerce"
    ) or 0.0


# ============================== [4] 수집 ==============================
def collect_fundamental(stock_code, stock_name, token):
    url = f"{BASE_URL}/api/dostk/stkinfo"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10001"
    }

    try:
        res = requests.post(url, json={"stk_cd": stock_code}, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()

        # API 에러 체크
        if data.get("return_code") not in (None, 0):
            print(f"🚨 {stock_code} API 오류")
            print(data)
            return None

        fund_data = {
            "stk_cd": stock_code,
            "stk_name": stock_name,
            "per": clean_numeric(data.get("per")),
            "pbr": clean_numeric(data.get("pbr")),
            "roe": clean_numeric(data.get("roe")),
            "eps": clean_numeric(data.get("eps")),
            "bps": clean_numeric(data.get("bps")),
            "cap": clean_numeric(data.get("cap")),
        }

        df = pd.DataFrame([fund_data])

        # ======================
        # 로그 변환 (ML용 핵심)
        # ======================
        for col in ["eps", "bps", "cap"]:
            df[f"{col}_log"] = np.log1p(df[col])

        # 추가 파생 피처
        df["value_score"] = df["roe"] / (df["pbr"] + 1e-6)  # ROE 대비 PBR

        # 파일 저장
        safe_name = stock_name.replace("/", "_").replace(" ", "_")
        save_path = DATA_DIR / f"{stock_code}_{safe_name}_fundamental.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

        return df

    except Exception as e:
        print(f"❌ {stock_code} ({stock_name}) 오류:", e)
        return None


# ============================== [5] 실행 ==============================
if __name__ == "__main__":
    if not APP_KEY or not APP_SECRET:
        print("🚨 .env 설정 필요")
        raise SystemExit

    token = get_access_token()
    if not token:
        print("🚨 토큰 발급 실패")
        raise SystemExit

    all_fund = []

    for code, name in TARGET_ASSETS.items():
        df = collect_fundamental(code, name, token)

        if df is not None:
            all_fund.append(df)
            print(f"✅ {code} {name} 수집 완료")
        else:
            print(f"❌ {code} {name} 실패")

        time.sleep(0.3)

    if all_fund:
        total_df = pd.concat(all_fund, ignore_index=True)
        total_path = DATA_DIR / "capstone_semiconductor_fundamental.csv"
        total_df.to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"\n🚀 전체 Fundamental 완료: {total_path}")