from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, message_to_dict
from langchain_core.messages.utils import messages_from_dict

from src.infrastructure.config.config import MEMORY_PIPELINE_ENABLED
from src.infrastructure.memory.sqlite_store import SQLiteStore
from src.infrastructure.memory.stm import stm_compression_node

logger = logging.getLogger(__name__)
_COMPRESSED_HEADER = "[Compressed Context"


class MemoryAdapter:
    """Bridge runtime state <-> SQLite-backed STM/LTM pipeline."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store or SQLiteStore()

    def load_context(self, session_id: str) -> tuple[list[BaseMessage], str]:
        if not MEMORY_PIPELINE_ENABLED:
            return [], ""

        row = self.store.get_latest_working_context_snapshot(session_id)
        if row is None:
            return [], ""

        raw = row["serialized_messages"]
        try:
            payload = json.loads(raw)
            if not isinstance(payload, list):
                return [], ""
            messages = list(messages_from_dict(payload))
        except Exception:
            logger.exception("Failed to parse working_context snapshot for session %s", session_id)
            return [], ""

        summary = self.extract_memory_summary(messages)
        return messages, summary

    def persist_turn(
        self,
        state: dict[str, Any],
        llm: BaseLanguageModel,
        *,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> dict[str, Any]:
        if not MEMORY_PIPELINE_ENABLED:
            return {"memory_pipeline_skipped": True}

        input_state = state.get("input", {})
        context_state = state.get("context", {})
        artifacts = state.get("artifacts", {})

        messages = list(context_state.get("messages", []))
        session_id = str(input_state.get("session_id") or "unknown")
        user_id = str(input_state.get("user_id") or "default")
        topic = artifacts.get("topic")
        if not isinstance(topic, str):
            topic = str(input_state.get("user_text") or "")

        try:
            stm_state = {
                "session_id": session_id,
                "user_id": user_id,
                "topic": topic,
                "messages": messages,
            }
            if event_loop is None:
                result = stm_compression_node(stm_state, llm)
            else:
                result = stm_compression_node(stm_state, llm, event_loop=event_loop)
        except Exception as exc:
            logger.exception("STM pipeline failed, fallback to raw snapshot for session %s", session_id)
            self._persist_uncompressed_snapshot(
                session_id=session_id,
                user_id=user_id,
                topic=topic,
                messages=messages,
            )
            if not context_state.get("memory_summary"):
                context_state["memory_summary"] = self.extract_memory_summary(messages)
            return {
                "stm_compressed": False,
                "memory_pipeline_degraded": True,
                "error": str(exc),
            }

        if result.get("stm_compressed") and isinstance(result.get("messages"), list):
            compressed_messages = result["messages"]
            if compressed_messages:
                context_state["messages"] = compressed_messages
                context_state["memory_summary"] = self.extract_memory_summary(compressed_messages)
            else:
                logger.warning(
                    "STM compression returned empty messages for session %s; keep original context messages",
                    session_id,
                )
        elif not context_state.get("memory_summary"):
            context_state["memory_summary"] = self.extract_memory_summary(messages)

        return result

    def _persist_uncompressed_snapshot(
        self,
        *,
        session_id: str,
        user_id: str,
        topic: str,
        messages: list[BaseMessage],
    ) -> None:
        backbone_rows: list[tuple[str, str, bool, int]] = []
        raw_rows: list[tuple[str, str, int]] = []
        for message in messages:
            if not isinstance(message, (HumanMessage, AIMessage)):
                continue
            role = "human" if isinstance(message, HumanMessage) else "assistant"
            content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
            token_estimate = max(1, len(content) // 4)
            backbone_rows.append((role, content, True, token_estimate))
            raw_rows.append((role, content, token_estimate))

        serialized = json.dumps([message_to_dict(m) for m in messages], ensure_ascii=False)
        token_count = sum(max(1, len(str(getattr(m, "content", ""))) // 4) for m in messages)
        with self.store.transaction() as conn:
            self.store.upsert_session(session_id, user_id, topic, conn=conn)
            if backbone_rows:
                self.store.save_messages(session_id, backbone_rows, conn=conn)
            if raw_rows:
                self.store.save_raw_messages(session_id, raw_rows, conn=conn)
            self.store.save_working_context_snapshot(
                session_id=session_id,
                serialized_messages=serialized,
                token_count=token_count,
                is_compressed=False,
                conn=conn,
            )

    @staticmethod
    def extract_memory_summary(messages: list[BaseMessage]) -> str:
        if not messages:
            return ""
        first = messages[0]
        if not isinstance(first, SystemMessage):
            return ""
        content = first.content if isinstance(first.content, str) else str(first.content)
        if not content.startswith(_COMPRESSED_HEADER):
            return ""
        _, _, rest = content.partition("\n")
        return rest.strip()
