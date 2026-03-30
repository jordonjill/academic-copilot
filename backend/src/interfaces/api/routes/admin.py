"""Admin reload endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.service import reload_all_config, reload_runtime_config, reload_tools_config

router = APIRouter(tags=["admin"])


@router.post("/admin/reload")
async def reload_all_config_route(
    _: str = Depends(verify_access_key),
) -> dict:
    report = await reload_all_config()
    runtime_failed = report.get("runtime", {}).get("failed", [])
    ok = len(runtime_failed) == 0
    return {
        "success": ok,
        "message": "Runtime and tools configuration reloaded." if ok else "Reload completed with validation issues.",
        "data": report,
    }


@router.post("/admin/reload-runtime")
async def reload_runtime_only_route(
    _: str = Depends(verify_access_key),
) -> dict:
    report = reload_runtime_config()
    ok = len(report.get("failed", [])) == 0
    return {
        "success": ok,
        "message": "Runtime configuration reloaded." if ok else "Runtime reload completed with validation issues.",
        "data": report,
    }


@router.post("/admin/reload-tools")
async def reload_tools_only_route(
    _: str = Depends(verify_access_key),
) -> dict:
    report = await reload_tools_config()
    return {
        "success": True,
        "message": "Tools configuration reloaded.",
        "data": report,
    }
