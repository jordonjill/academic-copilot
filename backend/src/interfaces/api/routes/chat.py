"""POST /chat — 无状态多轮对话（Supervisor 路由）。"""
from __future__ import annotations
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.schemas import ChatRequest, ChatResponse
from src.interfaces.api.service import create_copilot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str = Depends(verify_access_key),
) -> ChatResponse:
    """
    Supervisor 统一调度端点。

    主 Supervisor 可直接回答、调用子 Agent，或启动 Workflow。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        copilot = create_copilot()
        result = await copilot.chat_async(
            user_message=request.message,
            user_id=request.user_id,
            session_id=session_id,
            workflow_id=request.workflow_id,
            recursion_limit=25,
        )
    except Exception as e:
        logger.error(f"[chat] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    rtype = result.get("type", "chat")
    message = result.get("message") or (result.get("proposal") or {}).get("title", "")
    data = {k: v for k, v in result.items() if k not in ("success", "type", "message")} or None

    return ChatResponse(
        success=result.get("success", False),
        type=rtype,
        message=message,
        data=data,
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
    )
