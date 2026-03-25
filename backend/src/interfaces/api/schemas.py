"""API 请求/响应 Pydantic 模型。"""
from typing import Any, Dict, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    model_type: str = "ollama"


class ChatResponse(BaseModel):
    success: bool
    type: str = "chat"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    session_id: str
    timestamp: str


class ResearchRequest(BaseModel):
    topic: str
    model_type: str = "ollama"


class ResearchResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None
