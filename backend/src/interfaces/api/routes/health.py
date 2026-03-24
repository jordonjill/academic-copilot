"""系统状态端点。"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.session import session_manager
from src.interfaces.api.service import create_research_agent

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health")
async def health(
    model_type: str = "ollama",
    _: str = Depends(verify_access_key),
) -> Dict:
    if model_type not in ("ollama", "gemini", "openai"):
        return {"healthy": False, "status": f"Unsupported model_type: {model_type}"}
    try:
        result = create_research_agent(model_type).health_check()
        return {
            "healthy": result["status"] == "healthy",
            "status": result["message"],
            "model_type": model_type,
            "active_sessions": len(session_manager.all_sessions()),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"[health] {e}")
        return {
            "healthy": False,
            "status": str(e),
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/sessions")
async def list_sessions(_: str = Depends(verify_access_key)) -> Dict:
    return session_manager.all_sessions()
