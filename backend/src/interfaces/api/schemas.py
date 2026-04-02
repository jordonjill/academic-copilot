"""API 请求/响应 Pydantic 模型。"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32000)
    user_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_@.-]+$")
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    workflow_id: Optional[str] = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")

    model_config = ConfigDict(extra="forbid")

    @field_validator("user_id")
    @classmethod
    def _validate_user_id_path_component(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("user_id cannot be '.' or '..'")
        return value


class ChatResponseData(BaseModel):
    runtime: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class ChatResponse(BaseModel):
    success: bool
    type: str = "chat"
    message: Optional[str] = None
    data: Optional[ChatResponseData | Dict[str, Any]] = None
    session_id: str
    timestamp: str

    model_config = ConfigDict(extra="forbid")
