# app/database/__init__.py
from .vector_store import VectorStore

# 이렇게 하면 다른 파일에서:
from app.database import VectorStore  # 가능
# 대신:
from app.database.vector_store import VectorStore  # 긴 임포트 필요없음