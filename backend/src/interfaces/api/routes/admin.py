"""Admin reload endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from src.interfaces.api.deps import verify_admin_access_key
from src.interfaces.api.service import reload_all_config, reload_runtime_config, reload_tools_config

router = APIRouter(tags=["admin"])
logger = logging.getLogger(__name__)


@router.post("/admin/reload")
async def reload_all_config_route(
    request: Request,
    _: str = Depends(verify_admin_access_key),
) -> dict:
    logger.info("admin.reload.start client=%s path=%s", request.client.host if request.client else "-", request.url.path)
    report = await reload_all_config()
    runtime_failed = report.get("runtime", {}).get("failed", [])
    tools_failed = report.get("tools", {}).get("failed", [])
    ok = len(runtime_failed) == 0 and len(tools_failed) == 0
    logger.info(
        "admin.reload.complete client=%s runtime_failed=%d tools_failed=%d",
        request.client.host if request.client else "-",
        len(runtime_failed),
        len(tools_failed),
    )
    return {
        "success": ok,
        "message": "Runtime and tools configuration reloaded." if ok else "Reload completed with validation issues.",
        "data": report,
    }


@router.post("/admin/reload-runtime")
async def reload_runtime_only_route(
    request: Request,
    _: str = Depends(verify_admin_access_key),
) -> dict:
    logger.info("admin.reload_runtime.start client=%s", request.client.host if request.client else "-")
    report = reload_runtime_config()
    ok = len(report.get("failed", [])) == 0
    logger.info(
        "admin.reload_runtime.complete client=%s failed=%d",
        request.client.host if request.client else "-",
        len(report.get("failed", [])),
    )
    return {
        "success": ok,
        "message": "Runtime configuration reloaded." if ok else "Runtime reload completed with validation issues.",
        "data": report,
    }


@router.post("/admin/reload-tools")
async def reload_tools_only_route(
    request: Request,
    _: str = Depends(verify_admin_access_key),
) -> dict:
    logger.info("admin.reload_tools.start client=%s", request.client.host if request.client else "-")
    report = await reload_tools_config()
    failed = report.get("failed", [])
    ok = len(failed) == 0
    logger.info(
        "admin.reload_tools.complete client=%s failed=%d",
        request.client.host if request.client else "-",
        len(failed),
    )
    return {
        "success": ok,
        "message": "Tools configuration reloaded." if ok else "Tools reload completed with validation issues.",
        "data": report,
    }
