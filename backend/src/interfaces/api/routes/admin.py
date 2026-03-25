"""POST /admin/reload — runtime config reload endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.service import reload_runtime_config

router = APIRouter(tags=["admin"])


@router.post("/admin/reload")
async def reload_runtime_config_route(
    _: str = Depends(verify_access_key),
) -> dict:
    report = reload_runtime_config()
    return {
        "success": True,
        "message": "Runtime configuration reloaded.",
        "data": report,
    }
