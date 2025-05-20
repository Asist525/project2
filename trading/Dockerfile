FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (PostgreSQL, 컴파일 도구 등)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 프로젝트 전체 복사 (web + trading)
COPY . .

# static 파일 수집
RUN export $(cat .env.prod | xargs) && python manage.py collectstatic --noinput

# 실행 기본은 Django 웹서버
CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]
