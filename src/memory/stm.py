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
from datetime import datetime
from typing import Dict, List
from functools import partial

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)

from src.state import GlobalState
from src.config.config import STM_TOKEN_THRESHOLD, STM_KEEP_RECENT
from src.config.prompt import STM_COMPRESSION_PROMPT
from src.memory.sqlite_store import SQLiteStore


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
    backbone = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage):
            # 去除 tool_use 片段，只保 text
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
    rows = []
    for m in backbone:
        role = "human" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        rows.append((role, content, True, len(content) // 4))
    if rows:
        store.save_messages(session_id, rows)


def stm_compression_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """
    STM 压缩节点：每轮工作流结束后调用。
    - token 未超阈值：仅持久化，不修改 messages
    - token 超阈值：压缩旧消息 + 持久化主干
    """
    messages: List[BaseMessage] = list(state.get("messages", []))
    session_id = state.get("session_id", "unknown")
    user_id = state.get("user_id", "default")

    token_count = _estimate_tokens(messages)

    store = SQLiteStore()
    store.upsert_session(session_id, user_id, state.get("initial_topic") or "")

    backbone = _filter_backbone(messages)
    _persist_backbone(store, session_id, backbone)

    if token_count <= STM_TOKEN_THRESHOLD:
        return {
            "stm_token_count": token_count,
            "stm_compressed": False,
        }

    # --- 需要压缩 ---
    old_messages = backbone[:-STM_KEEP_RECENT]
    recent_messages = messages[-STM_KEEP_RECENT:]

    if not old_messages:
        return {
            "stm_token_count": token_count,
            "stm_compressed": False,
        }

    # LLM 摘要旧消息
    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    )
    compression_chain = STM_COMPRESSION_PROMPT | llm
    summary_response = compression_chain.invoke({"conversation_to_compress": conversation_text})
    summary_text = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

    compressed_messages = [
        SystemMessage(content=f"[Compressed Context — {datetime.utcnow().strftime('%Y-%m-%d')}]\n{summary_text}")
    ] + recent_messages

    new_token_count = _estimate_tokens(compressed_messages)

    # 触发 LTM 异步提取（不阻塞主流程）
    try:
        from src.memory.ltm import extract_and_update_ltm
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                extract_and_update_ltm(user_id=user_id, session_id=session_id, backbone=backbone, llm=llm)
            )
    except Exception as e:
        print(f"[STM] LTM async task scheduling failed: {e}")

    return {
        "messages": compressed_messages,
        "stm_token_count": new_token_count,
        "stm_compressed": True,
        "ltm_extraction_done": False,
    }
