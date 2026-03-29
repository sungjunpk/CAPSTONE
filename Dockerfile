# 1. 아키텍처 호환성이 높은 베이스 이미지 사용
FROM python:3.10-slim-bullseye

# 2. 컨테이너 내부 작업 폴더
WORKDIR /app

# 3. 라이브러리 설치 (캐시 무시 및 상세 로그 출력 설정)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. 실행
CMD ["python", "transformer.py"]