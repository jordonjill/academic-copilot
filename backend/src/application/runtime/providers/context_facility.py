from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from langchain_core.messages import BaseMessage

from src.application.runtime.utils.env_utils import read_env_int


_ContextScope = Literal["default", "supervisor"]


@dataclass(frozen=True)
class ContextPolicy:
    default_messages_window: int
    supervisor_messages_window: int
    trace_recent_window: int
    trace_max_items: int
    shared_summary_items: int
    text_preview_chars: int
    trace_output_preview_chars: int
    trace_reason_chars: int
    trace_instruction_chars: int

    @classmethod
    def from_env(cls) -> "ContextPolicy":
        default_messages_window = max(1, read_env_int("CONTEXT_MESSAGES_WINDOW_DEFAULT", 12))
        supervisor_messages_window = max(1, read_env_int("CONTEXT_MESSAGES_WINDOW_SUPERVISOR", 20))
        trace_recent_window = max(1, read_env_int("CONTEXT_TRACE_WINDOW", 8))
        trace_max_items = max(trace_recent_window, read_env_int("CONTEXT_TRACE_MAX_ITEMS", 64))
        shared_summary_items = max(1, read_env_int("CONTEXT_SHARED_SUMMARY_ITEMS", 6))
        text_preview_chars = max(32, read_env_int("CONTEXT_TEXT_PREVIEW_CHARS", 180))
        trace_output_preview_chars = max(32, read_env_int("CONTEXT_TRACE_OUTPUT_PREVIEW_CHARS", 240))
        trace_reason_chars = max(16, read_env_int("CONTEXT_TRACE_REASON_CHARS", 220))
        trace_instruction_chars = max(16, read_env_int("CONTEXT_TRACE_INSTRUCTION_CHARS", 220))
        return cls(
            default_messages_window=default_messages_window,
            supervisor_messages_window=supervisor_messages_window,
            trace_recent_window=trace_recent_window,
            trace_max_items=trace_max_items,
            shared_summary_items=shared_summary_items,
            text_preview_chars=text_preview_chars,
            trace_output_preview_chars=trace_output_preview_chars,
            trace_reason_chars=trace_reason_chars,
            trace_instruction_chars=trace_instruction_chars,
        )


class ContextFacility:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy.from_env()

    @classmethod
    def from_env(cls) -> "ContextFacility":
        return cls(policy=ContextPolicy.from_env())

    def messages_to_text(
        self,
        messages: list[BaseMessage],
        *,
        scope: _ContextScope = "default",
    ) -> str:
        window = self.policy.default_messages_window
        if scope == "supervisor":
            window = self.policy.supervisor_messages_window
        lines: list[str] = []
        for message in messages[-window:]:
            role = message.__class__.__name__.replace("Message", "").lower() or "message"
            content = _coerce_message_content(message)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def compact_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        excluded_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(artifacts, dict):
            return {}
        excluded = set(excluded_keys or [])
        keys = sorted([key for key in artifacts.keys() if isinstance(key, str)])
        compact: dict[str, Any] = {"keys": keys}

        artifact_summary: dict[str, Any] = {}
        for key in keys:
            if key in excluded:
                continue
            artifact_summary[key] = self.summarize_artifact_value(artifacts.get(key))
        compact["artifact_summary"] = artifact_summary

        shared = artifacts.get("shared")
        if isinstance(shared, dict):
            shared_summary: list[dict[str, Any]] = []
            for agent_id, payload in list(shared.items())[-self.policy.shared_summary_items :]:
                if not isinstance(agent_id, str):
                    continue
                item: dict[str, Any] = {"agent_id": agent_id}
                if isinstance(payload, dict):
                    node = payload.get("node")
                    if isinstance(node, str):
                        item["node"] = node
                    output_text = payload.get("output_text")
                    if isinstance(output_text, str) and output_text:
                        item["output_preview"] = output_text[: self.policy.text_preview_chars]
                    parsed = payload.get("parsed")
                    if isinstance(parsed, dict):
                        item["parsed_keys"] = sorted(list(parsed.keys()))[:12]
                shared_summary.append(item)
            compact["shared_summary"] = shared_summary

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
