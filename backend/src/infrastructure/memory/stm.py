"""
STM（短期记忆）压缩管道。

算法：
  1. 优先使用 tiktoken 估算当前 messages token 数（不可用时回退）
  2. 若 > STM_TOKEN_THRESHOLD：
       a. filter_backbone() — 保留 Human/AI text，丢弃 ToolMessage
       b. LLM 摘要旧消息（目标 STM_SUMMARY_TARGET_TOKENS）
       c. 保留最近上下文（优先按 STM_RECENT_TARGET_TOKENS，失败回退 STM_KEEP_RECENT）
       d. 构造 [SystemMessage(compressed_summary), ...recent_n]
  3. 将主干消息持久化到 SQLite
  4. 返回更新后的 state 字段
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import json
import logging
import threading
from datetime import UTC, datetime
import sqlite3
from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)

from src.infrastructure.config.config import (
    MEMORY_PIPELINE_ENABLED,
    STM_KEEP_RECENT,
    STM_RECENT_TARGET_TOKENS,
    STM_SUMMARY_TARGET_TOKENS,
    STM_TOKEN_THRESHOLD,
)
from src.infrastructure.config.prompt import STM_COMPRESSION_PROMPT
from src.infrastructure.memory.sqlite_store import SQLiteStore

COMPRESSION_SUMMARY_VERSION = "stm-v1"
logger = logging.getLogger(__name__)
_LTM_TASKS: set[asyncio.Future[Any] | concurrent.futures.Future[Any]] = set()
_LTM_TASKS_LOCK = threading.Lock()


def _track_ltm_task(task: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
    with _LTM_TASKS_LOCK:
        _LTM_TASKS.add(task)

    def _on_done(done: asyncio.Future[Any] | concurrent.futures.Future[Any]) -> None:
        with _LTM_TASKS_LOCK:
            _LTM_TASKS.discard(done)
        try:
            done.result()
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("[STM] LTM background task failed: %s", exc)

    task.add_done_callback(_on_done)


async def drain_ltm_tasks(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Wait briefly for pending LTM tasks during shutdown.

    Returns a small report for logging/observability.
    """
    timeout_seconds = max(0.1, float(timeout_seconds))
    with _LTM_TASKS_LOCK:
        pending = [task for task in list(_LTM_TASKS) if not task.done()]
    report: dict[str, Any] = {
        "initial_pending": len(pending),
        "completed": 0,
        "cancelled": 0,
        "remaining_pending": 0,
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
    }
    if not pending:
        return report

    loop = asyncio.get_running_loop()
    pending_awaitables: list[asyncio.Future[Any]] = []
    for pending_item in pending:
        if pending_item.done():
            continue
        if isinstance(pending_item, asyncio.Future):
            pending_awaitables.append(pending_item)
        else:
            pending_awaitables.append(asyncio.wrap_future(pending_item, loop=loop))
    if not pending_awaitables:
        return report

    try:
        await asyncio.wait_for(
            asyncio.gather(*pending_awaitables, return_exceptions=True),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        report["timed_out"] = True
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending_awaitables, return_exceptions=True)

    report["completed"] = sum(1 for task in pending if task.done() and not task.cancelled())
    report["cancelled"] = sum(1 for task in pending if task.cancelled())
    report["remaining_pending"] = sum(1 for task in pending if not task.done())
    return report


def _try_get_token_encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_text_tokens(text: str, encoder: Any | None) -> int:
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _estimate_tokens(messages: List[BaseMessage], *, encoder: Any | None) -> int:
    total = 0
    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        total += _estimate_text_tokens(content, encoder) + 4
    return max(1, total)


def _select_recent_messages_by_token_budget(
    messages: List[BaseMessage],
    token_budget: int,
    *,
    encoder: Any,
) -> List[BaseMessage]:
    if token_budget <= 0:
        return []
    selected_reversed: List[BaseMessage] = []
    used = 0
    for message in reversed(messages):
        content = message.content if isinstance(message.content, str) else str(message.content)
        message_tokens = _estimate_text_tokens(content, encoder) + 4
        if selected_reversed and (used + message_tokens) > token_budget:
            break
        selected_reversed.append(message)
        used += message_tokens
    selected_reversed.reverse()
    return selected_reversed


def _trim_text_to_token_budget(text: str, token_budget: int, *, encoder: Any | None) -> str:
    cleaned = text.strip()
    if token_budget <= 0 or not cleaned:
        return cleaned
    if encoder is None:
        approx_chars = max(32, token_budget * 4)
        return cleaned[:approx_chars]
    try:
        tokens = encoder.encode(cleaned)
        if len(tokens) <= token_budget:
            return cleaned
        return encoder.decode(tokens[:token_budget]).strip()
    except Exception:
        approx_chars = max(32, token_budget * 4)
        return cleaned[:approx_chars]


def _filter_backbone(messages: List[BaseMessage]) -> List[BaseMessage]:
    """提取对话主干：保留 HumanMessage 和 AIMessage（纯文本），丢弃 ToolMessage。"""
    backbone: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage):
            if isinstance(m.content, list):
                text_parts = [
                    part.get("text", "")
                    for part in m.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text = " ".join(text_parts).strip()
                if text:
                    backbone.append(AIMessage(content=text))
            elif isinstance(m.content, str) and m.content.strip():
                backbone.append(m)
        else:
            backbone.append(m)
    return backbone


def _persist_backbone(
    store: SQLiteStore,
    session_id: str,
    backbone: List[BaseMessage],
    *,
    conn: sqlite3.Connection | None = None,
) -> None:
    """将对话主干写入 SQLite messages 表。"""
    rows: List[tuple[str, str, bool, int]] = []
    for m in backbone:
        role = "human" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        rows.append((role, content, True, len(content) // 4))
    if rows:
        store.save_messages(session_id, rows, conn=conn)


def _serialize_messages(messages: List[BaseMessage]) -> str:
    serialized = [message_to_dict(m) for m in messages]
    return json.dumps(serialized, ensure_ascii=False)


def _normalize_summary_text(summary_response: object) -> str:
    content = getattr(summary_response, "content", summary_response)
    if isinstance(content, str):
        return content
    try:
        if isinstance(content, BaseMessage):
            return json.dumps(message_to_dict(content), ensure_ascii=False)
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def stm_compression_node(
    state: dict[str, Any],
    llm: BaseLanguageModel,
    *,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> Dict[str, Any]:
    """STM 压缩节点：每轮 chat turn 结束后调用。"""
    if not MEMORY_PIPELINE_ENABLED:
        return {"stm_token_count": 0, "stm_compressed": False, "memory_pipeline_skipped": True}

    messages: List[BaseMessage] = list(state.get("messages", []))
    session_id = state.get("session_id", "unknown")
    user_id = state.get("user_id", "default")
    token_encoder = _try_get_token_encoder()

    token_count = _estimate_tokens(messages, encoder=token_encoder)

    store = SQLiteStore()
    final_messages = messages
    final_token_count = token_count
    stm_compressed = False

    with store.transaction() as conn:
        store.upsert_session(session_id, user_id, state.get("topic") or "", conn=conn)

        backbone = _filter_backbone(messages)
        _persist_backbone(store, session_id, backbone, conn=conn)

        raw_rows: List[tuple[str, str, int]] = []
        for m in messages:
            if not isinstance(m, (HumanMessage, AIMessage)):
                continue
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            role = "human" if isinstance(m, HumanMessage) else "assistant"
            raw_rows.append((role, content, _estimate_tokens([m], encoder=token_encoder)))
        store.save_raw_messages(session_id, raw_rows, conn=conn)

        threshold_exceeded = token_count > STM_TOKEN_THRESHOLD
        if threshold_exceeded:
            recent_messages: List[BaseMessage]
            token_budget_usable = token_encoder is not None
            if token_budget_usable:
                recent_budget = min(
                    max(0, STM_RECENT_TARGET_TOKENS),
                    max(0, STM_TOKEN_THRESHOLD - STM_SUMMARY_TARGET_TOKENS),
                )
                if recent_budget > 0:
                    recent_messages = _select_recent_messages_by_token_budget(
                        messages,
                        recent_budget,
                        encoder=token_encoder,
                    )
                else:
                    keep_recent = STM_KEEP_RECENT if STM_KEEP_RECENT > 0 else len(messages)
                    keep_recent = min(len(messages), keep_recent)
                    recent_messages = messages[-keep_recent:] if keep_recent > 0 else []
            else:
                keep_recent = STM_KEEP_RECENT if STM_KEEP_RECENT > 0 else len(messages)
                keep_recent = min(len(messages), keep_recent)
                recent_messages = messages[-keep_recent:] if keep_recent > 0 else []

            recent_start = max(0, len(messages) - len(recent_messages))
            old_segment = messages[:recent_start]
            old_messages = _filter_backbone(old_segment)

            if old_messages:
                conversation_text = "\n".join(
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in old_messages
                )
                compression_chain = STM_COMPRESSION_PROMPT | llm
                summary_response = compression_chain.invoke({"conversation_to_compress": conversation_text})
                summary_text = _normalize_summary_text(summary_response)
                summary_text = _trim_text_to_token_budget(
                    summary_text,
                    STM_SUMMARY_TARGET_TOKENS,
                    encoder=token_encoder,
                )

                header = f"[Compressed Context — {datetime.now(UTC).strftime('%Y-%m-%d')}]"
                compressed_messages = [
                    SystemMessage(content=f"{header}\n{summary_text}")
                ] + recent_messages
                final_messages = compressed_messages
                final_token_count = _estimate_tokens(final_messages, encoder=token_encoder)
                stm_compressed = True

                store.save_compression_event(
                    session_id=session_id,
                    trigger_reason="stm_token_threshold",
                    pre_tokens=token_count,
                    post_tokens=final_token_count,
                    summary_text=summary_text,
                    summary_version=COMPRESSION_SUMMARY_VERSION,
                    conn=conn,
                )

                try:
                    from src.infrastructure.memory.ltm import extract_and_update_ltm

                    target_loop = event_loop
                    if target_loop is None:
                        # Fallback keeps direct/test-only callers working even without
                        # service layer passing an event loop explicitly.
                        try:
                            target_loop = asyncio.get_running_loop()
                        except RuntimeError:
                            target_loop = None
                    if target_loop is None:
                        raise RuntimeError("No target event loop for LTM scheduling")
                    task = asyncio.run_coroutine_threadsafe(
                        extract_and_update_ltm(
                            user_id=user_id,
                            session_id=session_id,
                            backbone=backbone,
                            llm=llm,
                        ),
                        target_loop,
                    )
                    _track_ltm_task(task)
                except RuntimeError:
                    logger.debug("[STM] No running event loop, skip LTM async extraction")
                except ModuleNotFoundError as exc:
                    logger.warning("[STM] LTM module unavailable: %s", exc)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.exception("[STM] LTM async task scheduling failed: %s", exc)
            else:
                store.save_compression_event(
                    session_id=session_id,
                    trigger_reason="stm_token_threshold",
                    pre_tokens=token_count,
                    post_tokens=token_count,
                    summary_text="no-op compression",
                    summary_version=COMPRESSION_SUMMARY_VERSION,
                    conn=conn,
                )

        store.save_working_context_snapshot(
            session_id,
            _serialize_messages(final_messages),
            final_token_count,
            stm_compressed,
            conn=conn,
        )

    if not stm_compressed:
        return {
            "stm_token_count": final_token_count,
            "stm_compressed": False,
        }

    return {
        "messages": final_messages,
        "stm_token_count": final_token_count,
        "stm_compressed": True,
        "ltm_extraction_done": False,
    }
