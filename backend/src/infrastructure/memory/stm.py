"""
STM（短期记忆）压缩管道。

算法：
  1. tiktoken 估算当前 messages token 数
  2. 若 > STM_TOKEN_THRESHOLD：
       a. filter_backbone() — 保留 Human/AI text，丢弃 ToolMessage
       b. LLM 摘要旧消息
       c. 保留最近 STM_KEEP_RECENT 条原文
       d. 构造 [SystemMessage(compressed_summary), ...recent_n]
  3. 将主干消息持久化到 SQLite
  4. 返回更新后的 state 字段
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime
from typing import Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)

from src.domain.state import GlobalState
from src.infrastructure.config.config import STM_KEEP_RECENT, STM_TOKEN_THRESHOLD
from src.infrastructure.config.prompt import STM_COMPRESSION_PROMPT
from src.infrastructure.memory.sqlite_store import SQLiteStore

COMPRESSION_SUMMARY_VERSION = "stm-v1"


def _estimate_tokens(messages: List[BaseMessage]) -> int:
    """用 tiktoken 估算 token 数（cl100k_base），回退到字符 / 4。"""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += len(enc.encode(content))
        return total
    except Exception:
        return sum(len(str(m.content)) // 4 for m in messages)


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


def _persist_backbone(store: SQLiteStore, session_id: str, backbone: List[BaseMessage]) -> None:
    """将对话主干写入 SQLite messages 表。"""
    rows: List[tuple[str, str, bool, int]] = []
    for m in backbone:
        role = "human" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        rows.append((role, content, True, len(content) // 4))
    if rows:
        store.save_messages(session_id, rows)


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


def stm_compression_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """STM 压缩节点：每轮工作流结束后调用。"""
    messages: List[BaseMessage] = list(state.get("messages", []))
    session_id = state.get("session_id", "unknown")
    user_id = state.get("user_id", "default")

    token_count = _estimate_tokens(messages)

    store = SQLiteStore()
    store.upsert_session(session_id, user_id, state.get("initial_topic") or "")

    backbone = _filter_backbone(messages)
    _persist_backbone(store, session_id, backbone)

    raw_rows: List[tuple[str, str, int]] = []
    for m in messages:
        if not isinstance(m, (HumanMessage, AIMessage)):
            continue
        content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
        role = "human" if isinstance(m, HumanMessage) else "assistant"
        raw_rows.append((role, content, _estimate_tokens([m])))
    store.save_raw_messages(session_id, raw_rows)

    final_messages = messages
    final_token_count = token_count
    stm_compressed = False

    threshold_exceeded = token_count > STM_TOKEN_THRESHOLD
    if threshold_exceeded:
        keep_recent = STM_KEEP_RECENT if STM_KEEP_RECENT > 0 else len(messages)
        keep_recent = min(len(messages), keep_recent)
        keep_backbone_recent = min(len(backbone), keep_recent)
        old_messages = (
            backbone[: len(backbone) - keep_backbone_recent]
            if keep_backbone_recent < len(backbone)
            else []
        )
        recent_messages = messages[-keep_recent:] if keep_recent > 0 else []

        if old_messages:
            conversation_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in old_messages
            )
            compression_chain = STM_COMPRESSION_PROMPT | llm
            summary_response = compression_chain.invoke({"conversation_to_compress": conversation_text})
            summary_text = _normalize_summary_text(summary_response)

            header = f"[Compressed Context — {datetime.utcnow().strftime('%Y-%m-%d')}]"
            compressed_messages = [
                SystemMessage(content=f"{header}\n{summary_text}")
            ] + recent_messages
            final_messages = compressed_messages
            final_token_count = _estimate_tokens(final_messages)
            stm_compressed = True

            store.save_compression_event(
                session_id=session_id,
                trigger_reason="stm_token_threshold",
                pre_tokens=token_count,
                post_tokens=final_token_count,
                summary_text=summary_text,
                summary_version=COMPRESSION_SUMMARY_VERSION,
            )

            try:
                from src.infrastructure.memory.ltm import extract_and_update_ltm

                loop = asyncio.get_running_loop()
                asyncio.ensure_future(
                    extract_and_update_ltm(
                        user_id=user_id,
                        session_id=session_id,
                        backbone=backbone,
                        llm=llm,
                    )
                )
            except (RuntimeError, ModuleNotFoundError):
                pass
            except Exception as exc:  # pragma: no cover - best effort
                print(f"[STM] LTM async task scheduling failed: {exc}")
        else:
            store.save_compression_event(
                session_id=session_id,
                trigger_reason="stm_token_threshold",
                pre_tokens=token_count,
                post_tokens=token_count,
                summary_text="no-op compression",
                summary_version=COMPRESSION_SUMMARY_VERSION,
            )

    store.save_working_context_snapshot(
        session_id,
        _serialize_messages(final_messages),
        final_token_count,
        stm_compressed,
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
