import requests
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

# [1] 환경 설정 및 API 키 로드
load_dotenv()
APP_KEY = os.getenv("KIWOOM_APP_KEY")
APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
BASE_URL = "https://api.kiwoom.com"

def get_access_token():
    """OAuth2 인증을 통해 접근 토큰 발급"""
    auth_url = f"{BASE_URL}/oauth2/token"
    try:
        res = requests.post(
            auth_url,
            json={
                "grant_type": "client_credentials",
                "appkey": APP_KEY,
                "secretkey": APP_SECRET
            },
            headers={"Content-Type": "application/json;charset=UTF-8"}
        )
        data = res.json()
        token = data.get("token")
        if not token:
            print(f"❌ 토큰 발급 실패: {data}")
            return None
        print("✅ 토큰 발급 성공")
        return token
    except Exception as e:
        print(f"🚨 인증 오류: {e}")
        return None

def to_number(series):
    """문자열 데이터를 정제하여 float32 숫자로 변환"""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
        .replace("", "0")
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float32")
    )

def collect_energy(stock_code, dt):
    """
    수급 데이터 수집 및 고급 전처리:
    1. 외국인/기관 순매수 수량 수집 (ka10060)
    2. 수급 가속도(Net Diff) 추가
    3. 추세 피처(MA5, MA20) 생성
    """
    print(f"🚀 {stock_code} 수급 데이터 수집 및 전처리 시작...")

    token = get_access_token()
    if not token:
        return

    # ka10060: 투자자별기관적차트요청 (외인/기관 동시 수집 가능)
    url = f"{BASE_URL}/api/dostk/chart"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka10060"
    }

    body = {
        "dt": dt,              # 기준일자 (예: 20260323)
        "stk_cd": stock_code,  # 종목코드
        "amt_qty_tp": "2",     # 1: 금액, 2: 수량
        "trde_tp": "0",        # 0: 순매수, 1: 매수, 2: 매도
        "unit_tp": "1"         # 1: 단주
    }

    try:
        res = requests.post(url, json=body, headers=headers)
        data = res.json()

        if data.get("return_code") != 0:
            print(f"❌ API 호출 실패: {data.get('return_msg')}")
            return

        raw_list = data.get("stk_invsr_orgn_chart", [])
        if not raw_list:
            print("⚠️ 수급 차트 데이터가 비어 있습니다.")
            return

        df = pd.DataFrame(raw_list)

        # 1. 필요한 컬럼 추출 및 이름 변경
        needed = ["dt", "frgnr_invsr", "orgn"]
        df = df[needed].copy()
        df.columns = ["dt", "frgn_net_qty", "orgn_net_qty"]

        # 2. 날짜 정렬 (과거 -> 미래 오름차순)
        df["dt"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

        # 3. 숫자 타입 변환
        df["frgn_net_qty"] = to_number(df["frgn_net_qty"])
        df["orgn_net_qty"] = to_number(df["orgn_net_qty"])

        # 4. [고급 전처리] 수급 가속도 (Relative Intensity) 추가
        # 전일 대비 수급의 변화량을 측정하여 매집 강도를 파악합니다.
        df["frgn_net_diff"] = df["frgn_net_qty"].diff().fillna(0).astype("float32")
        df["orgn_net_diff"] = df["orgn_net_qty"].diff().fillna(0).astype("float32")

        # 5. 추세 피처 (이동평균) 추가
        df["frgn_ma5"] = df["frgn_net_qty"].rolling(5, min_periods=1).mean().astype("float32")
        df["frgn_ma20"] = df["frgn_net_qty"].rolling(20, min_periods=1).mean().astype("float32")
        df["orgn_ma5"] = df["orgn_net_qty"].rolling(5, min_periods=1).mean().astype("float32")
        df["orgn_ma20"] = df["orgn_net_qty"].rolling(20, min_periods=1).mean().astype("float32")

        # 6. 저장용 날짜 포맷 복구
        df["dt"] = df["dt"].dt.strftime("%Y%m%d")

        # 7. 데이터 무결성 체크
        if (df["orgn_net_qty"] == 0).all():
            print("⚠️ 경고: 기관 순매수 데이터가 모두 0입니다. (시장 상황 또는 API 필드 확인 필요)")

        # 8. 최종 결과 저장
        file_name = f"{stock_code}_energy.csv"
        df.to_csv(file_name, index=False, encoding="utf-8-sig")

        print("-" * 30)
        print(f"✅ 수급 전처리 완료 및 저장: {file_name}")
        print(f"📊 최종 데이터 행 수: {len(df)}행")
        print(df[["dt", "frgn_net_qty", "frgn_net_diff", "orgn_net_qty"]].tail(3))

    except Exception as e:
        print(f"🚨 수집/전처리 중 예외 발생: {e}")

if __name__ == "__main__":
    collect_energy("000660", "20260323")