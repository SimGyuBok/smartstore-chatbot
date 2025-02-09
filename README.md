# 네이버 스마트스토어 FAQ 챗봇

## 프로젝트 소개

이 프로젝트는 네이버 스마트스토어에 대한 자주 묻는 질문(FAQ)에 AI 기반으로 응답하는 챗봇 시스템입니다.

## 주요 기능

- AI 기반 FAQ 검색 및 응답
- 벡터 데이터베이스를 활용한 의미론적 질문 매칭
- 실시간 스트리밍 응답
- 대화 맥락 유지

## 기술 스택

- Backend: FastAPI
- AI: OpenAI GPT
- Vector DB: ChromaDB
- Embedding: OpenAI Embedding API

## 설치 및 실행

1. 가상환경 생성/활성화

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. 환경변수 설정
   .env 파일 생성, OPENAI_API_KEY 설정
   OPENAI_API_KEY=your_openai_api_key

4. 데이터베이스 초기화

```bash
python init_db.py
```

5. 서버 실행

```bash
uvicorn app.main:app --reload
```

## 사용 방법

1. 챗봇 페이지 접속
   http://localhost:8000/static/index.html

2. 질문 입력 후 전송
3. 응답 확인
