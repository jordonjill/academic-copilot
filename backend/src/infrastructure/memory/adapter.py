from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage
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

        result = stm_compression_node(
            {
                "session_id": session_id,
                "user_id": user_id,
                "initial_topic": topic,
                "messages": messages,
            },
            llm,
        )

        if result.get("stm_compressed") and isinstance(result.get("messages"), list):
            context_state["messages"] = result["messages"]
            context_state["memory_summary"] = self.extract_memory_summary(result["messages"])
        elif not context_state.get("memory_summary"):
            context_state["memory_summary"] = self.extract_memory_summary(messages)

        return result

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
