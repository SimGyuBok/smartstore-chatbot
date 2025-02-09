# app/services/__init__.py
from .chat import chat_stream, ChatRequest, ChatResponse
from .embedding import get_embedding

# 서비스 초기화 코드
default_client = None
def initialize_service():
    global default_client
    default_client = ...