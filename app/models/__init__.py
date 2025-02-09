# models/__init__.py
from .chat import ChatMessage, ChatResponse

# 다른 파일에서:
from app.models import ChatMessage  # 이렇게 하면 순환 임포트 위험이 줄어듦