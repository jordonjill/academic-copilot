"""Session management endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.service import create_copilot

router = APIRouter(tags=["sessions"])

_SESSION_ID_PATTERN = r"^[A-Za-z0-9_.-]+$"


@router.delete("/sessions/{session_id}")
async def delete_session_route(
    session_id: Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=_SESSION_ID_PATTERN),
    ],
    _: str = Depends(verify_access_key),
) -> dict:
    deleted = create_copilot().delete_session(session_id)
    removed = deleted.get("sessions", 0) > 0
    return {
        "success": True,
        "message": "Session deleted." if removed else "Session not found.",
        "data": {
            "session_id": session_id,
            "deleted": deleted,
        },
    }
