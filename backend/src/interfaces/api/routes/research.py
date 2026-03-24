"""
WebSocket 流式研究端点（向后兼容）。

  POST /research/start          — 创建会话，返回 session_id
  GET  /research/status/{id}    — 查询会话状态
  WS   /ws/{id}?access_key=...  — 流式推送研究进度
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from src.interfaces.api.deps import ACCESS_KEY, verify_access_key
from src.interfaces.api.schemas import ResearchRequest, ResearchResponse
from src.interfaces.api.session import session_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["research"])


@router.post("/research/start", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    _: str = Depends(verify_access_key),
) -> ResearchResponse:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Research topic cannot be empty.")
    if request.model_type not in ("ollama", "gemini", "openai"):
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {request.model_type}")

    session_id = session_manager.create(request.topic, request.model_type)
    return ResearchResponse(
        success=True,
        message="Session created. Connect via WebSocket /ws/{session_id} to start.",
        session_id=session_id,
    )


@router.get("/research/status/{session_id}")
async def research_status(
    session_id: str,
    _: str = Depends(verify_access_key),
) -> Dict:
    meta = session_manager.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found.")
    agent = session_manager.get_agent(session_id)
    return {
        "session_id": session_id,
        **meta,
        "current_state": agent.get_current_state() if agent else None,
    }


@router.websocket("/ws/{session_id}")
async def websocket_research(websocket: WebSocket, session_id: str) -> None:
    # WebSocket 认证（query param）
    if websocket.query_params.get("access_key") != ACCESS_KEY:
        await websocket.close(code=1008, reason="Invalid access key")
        return

    await websocket.accept()

    meta = session_manager.get_meta(session_id)
    if not meta:
        await websocket.send_json({"type": "error", "message": "Session not found."})
        await websocket.close()
        return

    agent = session_manager.get_agent(session_id)
    if not agent:
        await websocket.send_json({"type": "error", "message": "Agent not initialized."})
        await websocket.close()
        return

    await websocket.send_json({
        "type": "connection",
        "message": f"Connected — session {session_id}",
        "topic": meta["topic"],
        "timestamp": datetime.now().isoformat(),
    })

    async def ws_send(payload: Dict) -> None:
        await websocket.send_json(payload)
        await asyncio.sleep(0.05)

    session_manager.set_status(session_id, "running")
    try:
        await agent.run_research_async(
            initial_topic=meta["topic"],
            websocket_send=ws_send,
            recursion_limit=15,
        )
        session_manager.set_status(session_id, "completed")
    except Exception as e:
        logger.error(f"[ws] research error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
        })
        session_manager.set_status(session_id, "failed")
    finally:
        # 保持连接直到客户端断开或发送 disconnect
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                elif msg.get("type") == "disconnect":
                    break
        except (WebSocketDisconnect, Exception):
            pass
        session_manager.remove(session_id)
