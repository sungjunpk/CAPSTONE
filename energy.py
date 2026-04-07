import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()

APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

MAX_DAYS = 2000

# =========================
# 2. 수집 대상 종목/ETF
# =========================
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

# 기준일: 오늘 날짜 자동 사용
DEFAULT_DT = datetime.today().strftime("%Y%m%d")


# =========================
# 3. 공통 함수
# =========================
def get_access_token():
    """키움 REST API 접근 토큰 발급"""
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

        token = data.get("token")
        if not token:
            print("🚨 토큰 응답은 왔는데 token 값이 없습니다.")
            print("응답:", data)
            return None

        return token

    except requests.exceptions.RequestException as e:
        print("🚨 토큰 발급 실패:", e)
        return None


def clean_numeric(series):
    """숫자 문자열 정리"""
    return (
        pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        .fillna(0)
        .astype("float32")
    )


def collect_energy(asset_code, asset_name, dt, token):
    """
    종목별투자자기관별차트요청(ka10060) 데이터 수집
    - 외국인 순매수 수량
    - 기관 순매수 수량
    """
    url = f"{BASE_URL}/api/dostk/chart"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10060"
    }

    body = {
        "dt": dt,             # 기준일
        "stk_cd": asset_code, # 종목코드/ETF코드
        "amt_qty_tp": "2",    # 1: 금액, 2: 수량
        "trde_tp": "0",       # 0: 순매수, 1: 매수, 2: 매도
        "unit_tp": "1"        # 1: 단주, 1000: 천주
    }

    all_data = []
    seen_dates = set()
    cont_yn = "N"
    next_key = ""

    for _ in range(30):  # 연속조회 최대 30번 제한
        headers["cont-yn"] = cont_yn
        headers["next-key"] = next_key

        try:
            res = requests.post(url, json=body, headers=headers, timeout=15)
            res.raise_for_status()
            res_json = res.json()
        except requests.exceptions.RequestException as e:
            print(f"🚨 {asset_code} ({asset_name}) 요청 실패:", e)
            return None

        # API 에러 메시지 출력용
        if res_json.get("return_code") not in (None, 0):
            print(f"🚨 {asset_code} ({asset_name}) API 오류")
            print("return_code:", res_json.get("return_code"))
            print("return_msg :", res_json.get("return_msg"))
            return None

        chunk = res_json.get("stk_invsr_orgn_chart", [])
        if not chunk:
            break

        for row in chunk:
            row_dt = row.get("dt")
            if row_dt and row_dt not in seen_dates:
                seen_dates.add(row_dt)
                all_data.append(row)

        # 종료 조건
        res_cont_yn = res.headers.get("cont-yn", "N")
        res_next_key = res.headers.get("next-key", "")

        if len(seen_dates) >= MAX_DAYS or res_cont_yn == "N":
            break

        cont_yn = res_cont_yn
        next_key = res_next_key
        time.sleep(0.2)

    if not all_data:
        print(f"⚠️ {asset_code} ({asset_name}) 데이터가 없습니다.")
        return None

    # 필요한 컬럼만 추출
    df = pd.DataFrame(all_data)[["dt", "frgnr_invsr", "orgn"]].copy()
    df.columns = ["dt", "frgn_net_qty", "orgn_net_qty"]

    df["frgn_net_qty"] = clean_numeric(df["frgn_net_qty"])
    df["orgn_net_qty"] = clean_numeric(df["orgn_net_qty"])

    df["stk_cd"] = asset_code
    df["stk_name"] = asset_name

    # 날짜 기준 정렬
    df = df.sort_values("dt").reset_index(drop=True)

    # 개별 파일 저장
    safe_name = asset_name.replace("/", "_").replace(" ", "_")
    file_path = DATA_DIR / f"{asset_code}_{safe_name}_energy.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    return df


# =========================
# 4. 실행부
# =========================
if __name__ == "__main__":
    if not APP_KEY or not APP_SECRET:
        print("🚨 .env에 KIWOOM_APP_KEY, KIWOOM_APP_SECRET 설정이 필요합니다.")
        raise SystemExit

    token = get_access_token()
    if not token:
        print("🚨 Access Token을 받지 못했습니다. .env 설정 또는 API 사용신청 상태를 확인하세요.")
        raise SystemExit

    all_collected = []

    for code, name in TARGET_ASSETS.items():
        df = collect_energy(code, name, DEFAULT_DT, token)

        if df is not None:
            all_collected.append(df)
            print(f"✅ {code} {name} Energy 데이터 수집 완료 ({len(df)}건)")
        else:
            print(f"❌ {code} {name} 수집 실패")

        time.sleep(0.5)

    if all_collected:
        total_df = pd.concat(all_collected, ignore_index=True)
        total_path = DATA_DIR / "capstone_semiconductor_energy.csv"
        total_df.to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"\n🚀 전체 Energy 데이터 통합 완료: {total_path}")
    else:
        print("⚠️ 수집된 데이터가 없습니다.")