import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()

APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

MAX_ROWS = 2000

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

DEFAULT_BASE_DT = datetime.today().strftime("%Y%m%d")


def get_access_token():
    auth_url = f"{BASE_URL}/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }

    try:
        res = requests.post(auth_url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data.get("token")
    except requests.exceptions.RequestException as e:
        print("🚨 토큰 발급 실패:", e)
        return None


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace("+", "", regex=False)
              .str.replace("-", "", regex=False),
        errors="coerce"
    ).fillna(0)


def preprocess_price_data(df, stock_code, stock_name):
    df["dt"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_log"] = np.log1p(df["volume"])

    df["stk_cd"] = stock_code
    df["stk_name"] = stock_name

    return df.dropna().reset_index(drop=True)


def collect_price(stock_code, stock_name, base_dt, token):
    url = f"{BASE_URL}/api/dostk/chart"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10081"
    }

    body = {
        "stk_cd": stock_code,
        "base_dt": base_dt,
        "upd_stkpc_tp": "1",
        "tic_scope": "1"
    }

    all_rows = []
    seen_dates = set()
    cont_yn = "N"
    next_key = ""

    for _ in range(30):
        headers["cont-yn"] = cont_yn
        headers["next-key"] = next_key

        try:
            res = requests.post(url, json=body, headers=headers, timeout=15)
            res.raise_for_status()
            res_json = res.json()
        except requests.exceptions.RequestException as e:
            print(f"🚨 {stock_code} ({stock_name}) 요청 실패:", e)
            return None

        if res_json.get("return_code") not in (None, 0):
            print(f"🚨 {stock_code} ({stock_name}) API 오류")
            print("return_code:", res_json.get("return_code"))
            print("return_msg :", res_json.get("return_msg"))
            print("response:", res_json)
            return None

        raw_list = (
            res_json.get("stk_dt_pole_chart_qry")
            or res_json.get("stk_day_pole_chart_qry")
            or res_json.get("stk_chart_qry")
            or []
        )

        if not raw_list:
            break

        for row in raw_list:
            row_dt = row.get("dt")
            if row_dt and row_dt not in seen_dates:
                seen_dates.add(row_dt)
                all_rows.append(row)

        res_cont_yn = res.headers.get("cont-yn", "N")
        res_next_key = res.headers.get("next-key", "")

        if len(seen_dates) >= MAX_ROWS or res_cont_yn == "N":
            break

        cont_yn = res_cont_yn
        next_key = res_next_key
        time.sleep(0.2)

    if not all_rows:
        print(f"⚠️ {stock_code} ({stock_name}) 가격 데이터가 없습니다.")
        return None

    df = pd.DataFrame(all_rows).copy()

    required_cols = ["dt", "open_pric", "high_pric", "low_pric", "cur_prc", "trde_qty"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"🚨 {stock_code} ({stock_name}) 응답 컬럼 누락:", missing_cols)
        print("응답 컬럼:", list(df.columns))
        return None

    df = df[required_cols]
    df.columns = ["dt", "open", "high", "low", "close", "volume"]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = clean_numeric(df[col]).astype("float32")

    df = preprocess_price_data(df, stock_code, stock_name)

    safe_name = stock_name.replace("/", "_").replace(" ", "_")
    save_path = DATA_DIR / f"{stock_code}_{safe_name}_price.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    if not APP_KEY or not APP_SECRET:
        print("🚨 .env에 KIWOOM_APP_KEY, KIWOOM_APP_SECRET 설정이 필요합니다.")
        raise SystemExit

    token = get_access_token()
    if not token:
        print("🚨 Access Token을 받지 못했습니다.")
        raise SystemExit

    all_data = []

    for code, name in TARGET_ASSETS.items():
        df = collect_price(code, name, DEFAULT_BASE_DT, token)

        if df is not None:
            all_data.append(df)
            print(f"✅ {code} {name} Price 수집 완료 ({len(df)}건)")
        else:
            print(f"❌ {code} {name} Price 수집 실패")

        time.sleep(0.5)

    if all_data:
        total_df = pd.concat(all_data, ignore_index=True)
        total_path = DATA_DIR / "capstone_semiconductor_price.csv"
        total_df.to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"\n🚀 전체 Price 데이터 통합 완료: {total_path}")
    else:
        print("⚠️ 수집된 가격 데이터가 없습니다.")