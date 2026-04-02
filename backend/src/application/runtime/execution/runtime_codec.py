from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.application.runtime.contracts.io_models import AgentTaskOutput


class RuntimeCodec:
    def __init__(self, *, logger: Any, decision_parser: Any) -> None:
        self._logger = logger
        self._decision_parser = decision_parser

    def extract_last_ai_text(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return self.coerce_text(message)
        return ""

    def extract_tool_outputs(self, messages: list[BaseMessage]) -> list[Any]:
        outputs: list[Any] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                outputs.append(message.content)
        return outputs

    def coerce_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, BaseMessage):
            content = raw.content
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        if hasattr(raw, "model_dump"):
            return json.dumps(raw.model_dump(), ensure_ascii=False, default=str)
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False, default=str)
        return str(raw)

    def try_parse_supervisor_decision_json(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = self._decision_parser.parse(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return self.try_parse_json(raw)

    def try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        decoder = json.JSONDecoder()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if fence_match:
            snippet = fence_match.group(1)
            try:
                parsed = json.loads(snippet)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text, idx)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue
        return None

    def normalize_agent_parsed_payload(
        self,
        text: str,
        parsed: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = dict(parsed or {})
        status = payload.get("status")
        if status not in {"success", "needs_clarification", "failed"}:
            status = "success"
        final_text = payload.get("final_text")
        if not isinstance(final_text, str) or not final_text.strip():
            final_text = text
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        confidence = payload.get("confidence", 0.5)
        errors = payload.get("errors", [])
        try:
            model = AgentTaskOutput.model_validate(
                {
                    "status": status,
                    "final_text": final_text,
                    "artifacts": artifacts,
                    "confidence": confidence,
                    "errors": errors,
                }
            )
        except Exception:
            model = AgentTaskOutput(status="failed", final_text=text, artifacts={}, confidence=0.0, errors=[])
        merged = dict(payload)
        merged.update(model.model_dump())
        return merged
