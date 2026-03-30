"""API 请求/响应 Pydantic 模型。"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None


class ChatResponseData(BaseModel):
    runtime: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    success: bool
    type: str = "chat"
    message: Optional[str] = None
    data: Optional[ChatResponseData | Dict[str, Any]] = None
    session_id: str
    timestamp: str
