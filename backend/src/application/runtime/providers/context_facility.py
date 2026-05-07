from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from langchain_core.messages import BaseMessage

from src.application.runtime.utils.env_utils import read_env_int


_ContextScope = Literal["default", "supervisor", "react"]


@dataclass(frozen=True)
class ContextPolicy:
    default_messages_window: int
    supervisor_messages_window: int
    trace_recent_window: int
    trace_max_items: int
    text_preview_chars: int
    trace_output_preview_chars: int
    trace_reason_chars: int
    trace_instruction_chars: int
    subagent_messages_token_cap: int = 0
    supervisor_messages_token_cap: int = 0
    react_messages_token_cap: int = 0

    @classmethod
    def from_env(cls) -> "ContextPolicy":
        default_messages_window = max(1, read_env_int("CONTEXT_MESSAGES_WINDOW_DEFAULT", 16))
        supervisor_messages_window = max(1, read_env_int("CONTEXT_MESSAGES_WINDOW_SUPERVISOR", 24))
        subagent_messages_token_cap = max(0, read_env_int("SUBAGENT_MESSAGES_TOKEN_CAP", 64000))
        supervisor_messages_token_cap = max(0, read_env_int("SUPERVISOR_MESSAGES_TOKEN_CAP", 96000))
        react_messages_token_cap = max(0, read_env_int("REACT_MESSAGES_TOKEN_CAP", subagent_messages_token_cap))
        trace_recent_window = max(1, read_env_int("CONTEXT_TRACE_WINDOW", 8))
        trace_max_items = max(trace_recent_window, read_env_int("CONTEXT_TRACE_MAX_ITEMS", 64))
        text_preview_chars = max(32, read_env_int("CONTEXT_TEXT_PREVIEW_CHARS", 180))
        trace_output_preview_chars = max(32, read_env_int("CONTEXT_TRACE_OUTPUT_PREVIEW_CHARS", 240))
        trace_reason_chars = max(16, read_env_int("CONTEXT_TRACE_REASON_CHARS", 220))
        trace_instruction_chars = max(16, read_env_int("CONTEXT_TRACE_INSTRUCTION_CHARS", 220))
        return cls(
            default_messages_window=default_messages_window,
            supervisor_messages_window=supervisor_messages_window,
            subagent_messages_token_cap=subagent_messages_token_cap,
            supervisor_messages_token_cap=supervisor_messages_token_cap,
            react_messages_token_cap=react_messages_token_cap,
            trace_recent_window=trace_recent_window,
            trace_max_items=trace_max_items,
            text_preview_chars=text_preview_chars,
            trace_output_preview_chars=trace_output_preview_chars,
            trace_reason_chars=trace_reason_chars,
            trace_instruction_chars=trace_instruction_chars,
        )


class ContextFacility:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy.from_env()
        self._token_encoder = _try_get_token_encoder()

    @classmethod
    def from_env(cls) -> "ContextFacility":
        return cls(policy=ContextPolicy.from_env())

    def messages_to_text(
        self,
        messages: list[BaseMessage],
        *,
        scope: _ContextScope = "default",
    ) -> str:
        selected_messages = self.select_recent_messages(messages, scope=scope)
        lines: list[str] = []
        for message in selected_messages:
            role = message.__class__.__name__.replace("Message", "").lower() or "message"
            content = _coerce_message_content(message)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def select_recent_messages(
        self,
        messages: list[BaseMessage],
        *,
        scope: _ContextScope = "default",
    ) -> list[BaseMessage]:
        window = self.policy.default_messages_window
        token_cap = self.policy.subagent_messages_token_cap
        if scope == "supervisor":
            window = self.policy.supervisor_messages_window
            token_cap = self.policy.supervisor_messages_token_cap
        elif scope == "react":
            token_cap = self.policy.react_messages_token_cap
        token_selected = self._select_recent_messages_by_token_cap(messages, token_cap)
        if token_selected is not None:
            return token_selected
        return messages[-window:]

    def _select_recent_messages_by_token_cap(
        self,
        messages: list[BaseMessage],
        token_cap: int,
    ) -> list[BaseMessage] | None:
        if token_cap <= 0:
            return None
        encoder = self._token_encoder
        if encoder is None:
            return None

        selected_reversed: list[BaseMessage] = []
        used_tokens = 0
        for message in reversed(messages):
            content = _coerce_message_content(message)
            message_tokens = _estimate_text_tokens(content, encoder) + 4
            if selected_reversed and (used_tokens + message_tokens) > token_cap:
                break
            selected_reversed.append(message)
            used_tokens += message_tokens

        if not selected_reversed:
            return []
        selected_reversed.reverse()
        return selected_reversed

    def compact_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        excluded_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(artifacts, dict):
            return {}
        excluded = set(excluded_keys or [])
        keys = sorted([key for key in artifacts.keys() if isinstance(key, str) and key not in excluded])
        compact: dict[str, Any] = {"keys": keys}

        artifact_summary: dict[str, Any] = {}
        for key in keys:
            artifact_summary[key] = self.summarize_artifact_value(artifacts.get(key))
        compact["artifact_summary"] = artifact_summary

        return compact

    def summarize_artifact_value(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "bool", "value": value}
        if isinstance(value, (int, float)):
            return {"type": "number", "value": value}
        if isinstance(value, str):
            return {
                "type": "string",
                "chars": len(value),
                "preview": value.strip()[: self.policy.text_preview_chars],
            }
        if isinstance(value, dict):
            keys = sorted([key for key in value.keys() if isinstance(key, str)])
            return {
                "type": "dict",
                "size": len(value),
                "keys": keys[:16],
            }
        if isinstance(value, list):
            sample_types: list[str] = []
            for item in value[:3]:
                sample_types.append(type(item).__name__)
            return {
                "type": "list",
                "count": len(value),
                "sample_types": sample_types,
            }
        return {"type": type(value).__name__}

    def append_trace(
        self,
        artifacts: dict[str, Any],
        *,
        entry: dict[str, Any],
        trace_key: str = "execution_trace",
    ) -> None:
        if not isinstance(artifacts, dict):
            return
        trace = artifacts.get(trace_key)
        if not isinstance(trace, list):
            trace = []
            artifacts[trace_key] = trace
        trace.append(entry)
        if len(trace) > self.policy.trace_max_items:
            del trace[: len(trace) - self.policy.trace_max_items]

    def recent_trace(
        self,
        artifacts: dict[str, Any],
        *,
        trace_key: str = "execution_trace",
    ) -> list[dict[str, Any]]:
        if not isinstance(artifacts, dict):
            return []
        trace = artifacts.get(trace_key)
        if not isinstance(trace, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in trace[-self.policy.trace_recent_window :]:
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows


def _coerce_message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _try_get_token_encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_text_tokens(text: str, encoder: Any) -> int:
    try:
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)
