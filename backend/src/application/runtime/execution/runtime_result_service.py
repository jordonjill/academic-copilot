from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional

from src.application.runtime.contracts.state_types import RuntimeState


_PUBLIC_RUNTIME_KEYS = (
    "mode",
    "workflow_id",
    "current_node",
    "step_count",
    "max_steps",
    "loop_count",
    "max_loops",
    "status",
    "token_usage",
)
_INTERNAL_ARTIFACT_KEYS = {
    "shared",
    "execution_trace",
    "supervisor_instruction",
    "task_input",
    "workflow_runner_input",
}


class RuntimeResultService:
    """Applies agent outputs and builds final API result payloads."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def apply_agent_output(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        text: str,
        parsed: Optional[Dict[str, Any]],
        source_kind: str = "agent",
    ) -> None:
        artifacts_state = state.get("artifacts")
        if not isinstance(artifacts_state, dict):
            artifacts_state = {}
            state["artifacts"] = artifacts_state
        for key in _INTERNAL_ARTIFACT_KEYS:
            artifacts_state.pop(key, None)

        artifacts_patch: dict[str, Any] = {}
        record_text = text
        if parsed and isinstance(parsed, dict):
            raw_patch = parsed.get("artifacts")
            if isinstance(raw_patch, dict):
                artifacts_patch = self._public_artifacts_patch(raw_patch)
                artifacts_state.update(artifacts_patch)
            parsed_final_text = parsed.get("final_text")
            if isinstance(parsed_final_text, str) and parsed_final_text.strip():
                record_text = parsed_final_text
            final_structured = parsed.get("final_structured")
            if isinstance(final_structured, dict):
                state["output"]["final_structured"] = final_structured
        if isinstance(record_text, str) and record_text.strip():
            state["io"]["last_execution_output"] = record_text

        self._append_execution_record(
            state,
            {
                "source_kind": source_kind,
                "source_id": agent_id,
                "node": node_name,
                "output_text": record_text,
                "output_preview": record_text[:240] if isinstance(record_text, str) else "",
                "artifact_keys": sorted(artifacts_patch.keys()),
                "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
                **self._execution_status_fields(parsed),
            },
        )

    def best_available_final_text(self, state: RuntimeState) -> Optional[str]:
        final_text = state["output"].get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            return final_text

        for record in reversed(self._execution_records(state)):
            output_text = record.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text

        shared = state.get("artifacts", {}).get("shared", {})
        if isinstance(shared, dict):
            reporter = shared.get("reporter")
            if isinstance(reporter, dict):
                reporter_text = reporter.get("output_text")
                if isinstance(reporter_text, str) and reporter_text.strip():
                    return reporter_text

            for item in reversed(list(shared.values())):
                if not isinstance(item, dict):
                    continue
                output_text = item.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text
        return None

    def public_runtime(self, state: RuntimeState) -> dict[str, Any]:
        raw_runtime = state.get("runtime", {})
        if not isinstance(raw_runtime, Mapping):
            raw_runtime = {}

        runtime = {key: raw_runtime.get(key) for key in _PUBLIC_RUNTIME_KEYS if key in raw_runtime}
        budget = self._public_tool_budget(raw_runtime.get("tool_budget"))
        if budget:
            runtime["tool_budget"] = budget
        return runtime

    def public_outputs(self, state: RuntimeState) -> dict[str, Any]:
        artifacts = state.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            return {}

        outputs: dict[str, Any] = {}
        report_exports = self._find_report_exports(artifacts)
        if report_exports:
            outputs["report_exports"] = report_exports
        return outputs

    def build_result(self, state: RuntimeState) -> Dict[str, Any]:
        final_structured = state["output"].get("final_structured")
        if isinstance(final_structured, dict):
            result = {"success": True, "type": "structured", "data": final_structured}
            if "message" in final_structured and isinstance(final_structured["message"], str):
                result["message"] = final_structured["message"]
            return result

        final_text = (
            state["output"].get("final_text")
            or self.best_available_final_text(state)
            or state["io"].get("last_execution_output")
            or state["io"].get("last_model_output")
        )
        return {
            "success": bool(final_text),
            "type": "chat",
            "message": final_text or "No output produced.",
            "data": {
                "runtime": self.public_runtime(state),
                "outputs": self.public_outputs(state),
            },
        }

    @staticmethod
    def _public_tool_budget(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}

        limits = value.get("limits")
        counts = value.get("counts")
        public: dict[str, Any] = {
            "scope": value.get("scope"),
            "workflow_id": value.get("workflow_id"),
            "limits": dict(limits) if isinstance(limits, Mapping) else {},
            "counts": dict(counts) if isinstance(counts, Mapping) else {},
        }
        return public

    @classmethod
    def _find_report_exports(cls, artifacts: Mapping[str, Any]) -> dict[str, str]:
        direct = artifacts.get("report_exports")
        if isinstance(direct, Mapping):
            return cls._coerce_report_exports(direct)

        shared = artifacts.get("shared")
        if not isinstance(shared, Mapping):
            return {}

        for value in shared.values():
            if not isinstance(value, Mapping):
                continue
            parsed = value.get("parsed")
            if not isinstance(parsed, Mapping):
                continue
            parsed_artifacts = parsed.get("artifacts")
            if not isinstance(parsed_artifacts, Mapping):
                continue
            report_exports = parsed_artifacts.get("report_exports")
            if isinstance(report_exports, Mapping):
                coerced = cls._coerce_report_exports(report_exports)
                if coerced:
                    return coerced
        return {}

    @staticmethod
    def _public_artifacts_patch(raw_patch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(raw_patch).items()
            if isinstance(key, str) and key not in _INTERNAL_ARTIFACT_KEYS
        }

    @staticmethod
    def _execution_status_fields(parsed: Optional[Dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {}
        fields: dict[str, Any] = {}
        status = parsed.get("status")
        if isinstance(status, str) and status:
            fields["status"] = status
        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)):
            fields["confidence"] = float(confidence)
        errors = parsed.get("errors")
        if isinstance(errors, list):
            fields["errors"] = [item for item in errors if isinstance(item, str)]
        return fields

    @staticmethod
    def _execution_records(state: RuntimeState) -> list[dict[str, Any]]:
        records = state.get("executions")
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def _append_execution_record(state: RuntimeState, record: dict[str, Any]) -> None:
        records = state.get("executions")
        if not isinstance(records, list):
            records = []
            state["executions"] = records
        records.append(record)

    @staticmethod
    def _coerce_report_exports(value: Mapping[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key in ("docx_path", "pdf_path"):
            path = value.get(key)
            if isinstance(path, str) and path.strip():
                result[key] = path
        return result
