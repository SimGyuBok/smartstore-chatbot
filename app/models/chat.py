from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatHistory(BaseModel):
    messages: List[ChatMessage]
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default-session")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatResponse(BaseModel):
    answer: str
    follow_up_questions: List[str] = []
    sources: List[str] = []
    confidence_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)