import requests
import json

# 1. 인증 정보
APP_KEY = "Dfqx2NY8o0CVnBVuO6FQ927JlmXRZJLvuvYd8fE3WCU"
APP_SECRET = "tVcKZo1wjMIStEAfkdTiutPLFWLuP0BvMJDmaReuIFg"

# 2. 토큰 발급 요청
url = "https://api.kiwoom.com/oauth2/token"
headers = {"Content-Type": "application/json;charset=UTF-8"}
body = {
    "grant_type": "client_credentials",
    "appkey": APP_KEY,
    "secretkey": APP_SECRET  # 스크린샷에 secretkey라고 되어있으므로 맞춰줍니다.
}

print("--- [1] 토큰 발급 시도 ---")
response = requests.post(url, json=body, headers=headers)
data = response.json()

# 명세서에 따라 'token' 키를 사용합니다.
access_token = data.get("token")

if access_token:
    print(f"✅ 토큰 발급 성공!")
    print(f"발급된 토큰: {access_token[:30]}...")
    print(f"만료 일시: {data.get('expires_dt')}")
    
    # 3. 이제 이 토큰으로 삼성전자 현재가를 조회해봅시다.
    # (조회 API 주소는 보통 /v1/quotes/stocks/005930 형태입니다.)
    print("\n--- [2] 다음 단계 진행 가능 ---")
    print("이제 이 토큰을 Authorization 헤더에 담아 데이터를 요청하면 됩니다.")
else:
    print("❌ 토큰 발급 실패. 응답을 확인하세요:")
    print(json.dumps(data, indent=4, ensure_ascii=False))