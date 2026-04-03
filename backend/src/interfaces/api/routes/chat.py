"""POST /chat — 无状态多轮对话（Supervisor 路由）。"""
from __future__ import annotations
import asyncio
import contextlib
import json
import uuid
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.interfaces.api.deps import verify_access_key
from src.interfaces.api.rate_limit import enforce_chat_rate_limit
from src.interfaces.api.schemas import ChatRequest, ChatResponse
from src.interfaces.api.service import create_copilot, get_config_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
_SSE_HEARTBEAT_SECONDS = 15.0


def _validate_chat_request(request: ChatRequest) -> None:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if request.workflow_id:
        registry = get_config_registry()
        if registry.workflows and request.workflow_id not in registry.workflows:
            raise HTTPException(status_code=400, detail=f"Unknown workflow_id: {request.workflow_id}")


def _coerce_chat_response_payload(
    result: dict[str, Any],
    session_id: str,
) -> ChatResponse:
    rtype = result.get("type", "chat")
    message = result.get("message") or (result.get("proposal") or {}).get("title", "")

    data = result.get("data")
    if data is None:
        extras = {k: v for k, v in result.items() if k not in ("success", "type", "message")}
        if "data" in extras and len(extras) == 1:
            data = extras["data"]
        elif extras:
            data = extras
        else:
            data = None

    if (not message) and isinstance(data, dict):
        for key in ("final_text", "summary", "title"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                message = value
                break

    return ChatResponse(
        success=result.get("success", False),
        type=rtype,
        message=message,
        data=data,
        session_id=session_id,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _encode_sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _build_error_event(status_code: int, message: str, session_id: str) -> dict[str, Any]:
    return {
        "type": "error",
        "status_code": status_code,
        "message": message,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def _run_chat_turn(
    request: ChatRequest,
    session_id: str,
    emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Execute one chat turn through streaming-capable path.

    Returns:
      - final result dict
      - whether a completion event was already emitted
    """
    copilot = create_copilot()
    completion_result: dict[str, Any] | None = None
    completion_emitted = False

    async def _dispatch(payload: dict[str, Any]) -> None:
        nonlocal completion_result, completion_emitted
        if str(payload.get("type", "")) == "completion":
            maybe_result = payload.get("final_result")
            if isinstance(maybe_result, dict):
                completion_result = maybe_result
                completion_emitted = True
        if emit is not None:
            await emit(payload)

    result = await copilot.chat_async(
        user_message=request.message,
        user_id=request.user_id,
        session_id=session_id,
        workflow_id=request.workflow_id,
        websocket_send=_dispatch,
    )
    return completion_result or result, completion_emitted


@router.post("/chat", response_model=ChatResponse)
async def chat(
    http_request: Request,
    request: ChatRequest,
    __: None = Depends(enforce_chat_rate_limit),
    _: str = Depends(verify_access_key),
) -> ChatResponse:
    """
    Supervisor 统一调度端点。

    主 Supervisor 可直接回答、调用子 Agent，或启动 Workflow。
    """
    _validate_chat_request(request)
    del http_request

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result, _ = await _run_chat_turn(request=request, session_id=session_id)
    except (asyncio.TimeoutError, TimeoutError) as e:
        logger.warning("[chat] timeout: %s", e)
        raise HTTPException(status_code=504, detail=str(e))
    except ValueError as e:
        logger.warning("[chat] bad request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[chat] error")
        raise HTTPException(status_code=500, detail=str(e))

    return _coerce_chat_response_payload(result, session_id=session_id)


@router.post("/chat/stream")
async def chat_stream(
    http_request: Request,
    request: ChatRequest,
    __: None = Depends(enforce_chat_rate_limit),
    _: str = Depends(verify_access_key),
) -> StreamingResponse:
    _validate_chat_request(request)

    session_id = request.session_id or str(uuid.uuid4())
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _send(payload: dict[str, Any]) -> None:
        await queue.put(payload)

    async def _run_chat() -> None:
        try:
            result, completion_emitted = await _run_chat_turn(
                request=request,
                session_id=session_id,
                emit=_send,
            )
            if not completion_emitted:
                await _send(
                    {
                        "type": "completion",
                        "final_result": result,
                        "session_id": session_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
        except (asyncio.TimeoutError, TimeoutError) as e:
            logger.warning("[chat/stream] timeout: %s", e)
            await _send(_build_error_event(504, str(e), session_id))
        except ValueError as e:
            logger.warning("[chat/stream] bad request: %s", e)
            await _send(_build_error_event(400, str(e), session_id))
        except Exception as e:
            logger.exception("[chat/stream] error")
            await _send(_build_error_event(500, str(e), session_id))
        finally:
            await queue.put(None)

    async def _event_generator() -> AsyncGenerator[str, None]:
        worker = asyncio.create_task(_run_chat())
        try:
            yield _encode_sse(
                "connected",
                {
                    "type": "connected",
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            while True:
                if await http_request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if payload is None:
                    break
                event_type = str(payload.get("type", "message"))
                yield _encode_sse(event_type, payload)
        finally:
            if not worker.done():
                worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
