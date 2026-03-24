"""
WebSocket 流式研究会话管理器。

职责：创建/获取/销毁 ResearchAgentApp 实例及对应会话元数据。
仅供 /research/* 和 /ws/* 路由使用；新 /chat 端点无状态，不经过此处。
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import HTTPException, WebSocket

from src.interfaces.api.service import ResearchAgentApp, create_research_agent


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict] = {}
        self._agents: Dict[str, ResearchAgentApp] = {}

    # ── 创建 ──────────────────────────────────────────────────────────────────
    def create(self, topic: str, model_type: str = "ollama") -> str:
        session_id = str(uuid.uuid4())
        try:
            agent = create_research_agent(model_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to init agent: {e}")
        self._agents[session_id] = agent
        self._sessions[session_id] = {
            "topic": topic,
            "model_type": model_type,
            "status": "created",
            "created_at": datetime.now().isoformat(),
        }
        return session_id

    # ── 查询 ──────────────────────────────────────────────────────────────────
    def get_meta(self, session_id: str) -> Optional[Dict]:
        return self._sessions.get(session_id)

    def get_agent(self, session_id: str) -> Optional[ResearchAgentApp]:
        return self._agents.get(session_id)

    def all_sessions(self) -> Dict[str, Dict]:
        return {
            sid: {k: v for k, v in meta.items() if k != "websocket"}
            for sid, meta in self._sessions.items()
        }

    # ── 状态更新 ──────────────────────────────────────────────────────────────
    def set_status(self, session_id: str, status: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["status"] = status

    # ── 销毁 ──────────────────────────────────────────────────────────────────
    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._agents.pop(session_id, None)


# 全局单例（由各路由模块共享）
session_manager = SessionManager()
