import re
from typing import List, Dict, Any
from datetime import datetime

def clean_text(text: str) -> str:
    """
    텍스트 전처리를 위한 유틸리티 함수
    - 불필요한 공백 제거
    - 특수문자 처리
    - 개행문자 정리
    """
    # 여러 줄 공백을 하나의 개행문자로 변경
    text = re.sub(r'\n\s*\n', '\n', text)
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 텍스트 앞뒤 공백 제거
    text = text.strip()
    return text

def validate_session_id(session_id: str) -> bool:
    """
    세션 ID의 유효성을 검사하는 함수
    """
    # 세션 ID는 영문자, 숫자, 하이픈으로만 구성되어야 함
    pattern = re.compile(r'^[a-zA-Z0-9-]+$')
    return bool(pattern.match(session_id))

def format_chat_history(history: List[Dict[str, Any]]) -> str:
    """
    채팅 이력을 문자열로 포맷팅하는 함수
    """
    formatted = []
    for msg in history:
        role = "사용자" if msg["role"] == "user" else "챗봇"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def log_chat(session_id: str, message: str, role: str):
    """
    채팅 로그를 기록하는 함수
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Session: {session_id} | {role}: {message}\n"
    
    with open("chat_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

def truncate_chat_history(history: List[Dict[str, Any]], max_length: int = 5) -> List[Dict[str, Any]]:
    """
    채팅 이력을 최대 길이로 제한하는 함수
    """
    if len(history) > max_length:
        return history[-max_length:]
    return history

def extract_key_points(text: str) -> List[str]:
    """
    텍스트에서 주요 포인트를 추출하는 함수
    """
    # 구분자로 분리하여 주요 포인트 추출
    points = []
    
    # 번호로 시작하는 문장 찾기
    numbered = re.findall(r'\d+\.\s+[^.!?]+[.!?]', text)
    if numbered:
        points.extend(numbered)
    
    # 줄바꿈으로 구분된 문장 찾기
    if not points:
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        points.extend(sentences)
    
    # 문장부호로 구분된 문장 찾기
    if not points:
        sentences = re.split(r'[.!?]+', text)
        points.extend([s.strip() for s in sentences if s.strip()])
    
    return points

def is_valid_question(text: str) -> bool:
    """
    유효한 질문인지 확인하는 함수
    """
    # 최소 길이 체크
    if len(text.strip()) < 2:
        return False
        
    # 질문형 문장인지 체크
    question_patterns = [
        r'\?$',  # 물음표로 끝나는 경우
        r'(무엇|어떻게|언제|어디서|누구|왜|어떤|몇|얼마|습니까|있나요|없나요|인가요|가요|나요)[\?]?$'
    ]
    
    return any(re.search(pattern, text) for pattern in question_patterns)

def sanitize_input(text: str) -> str:
    """
    사용자 입력을 안전하게 처리하는 함수
    """
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 이모지 제거
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 연속된 특수문자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()