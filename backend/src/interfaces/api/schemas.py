"""API 请求/响应 Pydantic 模型。"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(default="default", min_length=1, max_length=128)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    workflow_id: Optional[str] = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")
