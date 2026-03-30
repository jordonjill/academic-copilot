"""系统状态端点。"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.service import create_copilot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health")
async def health(
    _: str = Depends(verify_access_key),
) -> Dict:
    try:
        result = create_copilot().health_check()
        return {
            "healthy": result["status"] == "healthy",
            "status": result["message"],
            "probe": result.get("probe"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"[health] {e}")
        return {
            "healthy": False,
            "status": str(e),
            "timestamp": datetime.now().isoformat(),
        }
