from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.application.runtime.contracts.io_models import AgentTaskInput, WorkflowRunnerInput, WorkflowRunnerOutput
from src.application.runtime.contracts.state_types import RuntimeState


class IsolationFacility:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        apply_agent_output: Callable[..., None],
    ) -> None:
        self._logger = logger
        self._apply_agent_output = apply_agent_output

    def build_isolated_subagent_state(
        self,
        parent_state: RuntimeState,
        agent_id: str,
        instruction: Any,
        *,
        input_artifact_keys: Optional[list[str]] = None,
        inline_input_artifacts: Optional[dict[str, Any]] = None,
        node_name: Optional[str] = None,
        runtime_mode: str = "subagent",
    ) -> RuntimeState:
        task_text = ""
        if isinstance(instruction, str) and instruction.strip():
            task_text = instruction.strip()
        if not task_text:
            task_text = parent_state["input"].get("user_text", "")
        selected_input_artifacts = self.compose_input_artifacts(
            parent_state,
            input_artifact_keys=input_artifact_keys,
            inline_input_artifacts=inline_input_artifacts,
        )
        task_input = AgentTaskInput(
            task_id=f"{agent_id}:{parent_state['runtime'].get('step_count', 0)}",
            instruction=task_text,
            input_artifacts=selected_input_artifacts,
        )
        artifacts_topic = None
        parent_artifacts = parent_state.get("artifacts")
        if isinstance(parent_artifacts, dict):
            artifacts_topic = parent_artifacts.get("topic")
        return {
            "input": {
                "user_text": task_text,
                "user_id": parent_state["input"].get("user_id", ""),
                "session_id": parent_state["input"].get("session_id", ""),
            },
            "context": {
                "messages": [],
                "memory_summary": "",
            },
            "runtime": {
                "mode": "subagent",
                "workflow_id": parent_state.get("runtime", {}).get("workflow_id"),
                "current_node": agent_id,
                "step_count": 0,
                "loop_count": 0,
                "status": "running",
            },
            "io": {
                "last_model_output": None,
                "last_execution_output": None,
                "last_tool_outputs": [],
            },
            "artifacts": {
                "topic": artifacts_topic,
                **selected_input_artifacts,
            },
            "task": {
                "instruction": task_text,
                "input_artifact_keys": list(input_artifact_keys or []),
                "input_artifacts": selected_input_artifacts,
                "task_input": task_input.model_dump(),
            },
            "executions": [],
            "output": {
                "final_text": None,
                "final_structured": None,
            },
            "errors": {
                "last_error": None,
            },
        }

    def select_input_artifacts(
        self,
        parent_state: RuntimeState,
        input_artifact_keys: Optional[list[str]],
    ) -> dict[str, Any]:
        parent_artifacts = parent_state.get("artifacts")
        if not isinstance(parent_artifacts, dict):
            return {}

        excluded = {
            "shared",
            "supervisor_instruction",
            "task_input",
            "workflow_runner_input",
            "execution_trace",
        }
        if input_artifact_keys:
            picked: dict[str, Any] = {}
            for key in input_artifact_keys:
                if not isinstance(key, str):
                    continue
                clean = key.strip()
                if not clean or clean in excluded:
                    continue
                if clean in parent_artifacts:
                    picked[clean] = parent_artifacts.get(clean)
            return picked

        return {
            key: value
            for key, value in parent_artifacts.items()
            if key not in excluded
        }

    def coerce_inline_input_artifacts(
        self,
        inline_input_artifacts: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(inline_input_artifacts, dict):
            return {}
        cleaned: dict[str, Any] = {}
        for key, value in inline_input_artifacts.items():
            if not isinstance(key, str):
                continue
            text = key.strip()
            if not text:
                continue
            cleaned[text] = value
        return cleaned

    def compose_input_artifacts(
        self,
        parent_state: RuntimeState,
        *,
        input_artifact_keys: Optional[list[str]],
        inline_input_artifacts: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        selected = self.select_input_artifacts(parent_state, input_artifact_keys)
        inline = self.coerce_inline_input_artifacts(inline_input_artifacts)
        if not inline:
            return selected
        merged = dict(selected)
        for key, value in inline.items():
            if self._is_effectively_empty_value(merged.get(key)):
                merged[key] = value
        return merged

    @staticmethod
    def _is_effectively_empty_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def build_isolated_workflow_state(
        self,
        parent_state: RuntimeState,
        workflow_id: str,
        runner_input: Optional[WorkflowRunnerInput] = None,
        *,
        inline_input_artifacts: Optional[dict[str, Any]] = None,
    ) -> RuntimeState:
        user_text = parent_state["input"].get("user_text", "")
        artifacts_topic = None
        parent_artifacts = parent_state.get("artifacts")
        if isinstance(parent_artifacts, dict):
            artifacts_topic = parent_artifacts.get("topic")
        if runner_input is None:
            runner_input = WorkflowRunnerInput(
                workflow_id=workflow_id,
                instruction=user_text,
                seed_artifacts=self.compose_input_artifacts(
                    parent_state,
                    input_artifact_keys=None,
                    inline_input_artifacts=inline_input_artifacts,
                ),
            )
        return {
            "input": {
                "user_text": user_text,
                "user_id": parent_state["input"].get("user_id", ""),
                "session_id": parent_state["input"].get("session_id", ""),
            },
            "context": {
                "messages": [],
                "memory_summary": "",
            },
            "runtime": {
                "mode": "workflow",
                "workflow_id": workflow_id,
                "current_node": None,
                "step_count": 0,
                "loop_count": 0,
                "status": "running",
            },
            "io": {
                "last_model_output": None,
                "last_execution_output": None,
                "last_tool_outputs": [],
            },
            "artifacts": {
                "topic": artifacts_topic,
                **runner_input.seed_artifacts,
            },
            "task": {
                "instruction": runner_input.instruction,
                "input_artifacts": runner_input.seed_artifacts,
                "workflow_runner_input": runner_input.model_dump(),
            },
            "executions": [],
            "output": {
                "final_text": None,
                "final_structured": None,
            },
            "errors": {
                "last_error": None,
            },
        }

    def collect_subagent_execution_result(
        self,
        isolated_state: RuntimeState,
        agent_id: str,
    ) -> Dict[str, Any]:
        isolated_io = isolated_state.get("io", {})
        record = self._latest_execution_record(isolated_state, source_id=agent_id)
        text = isolated_io.get("last_model_output") if isinstance(isolated_io, dict) else None
        if not isinstance(text, str) or not text.strip():
            output_state = isolated_state.get("output", {})
            candidate = output_state.get("final_text") if isinstance(output_state, dict) else None
            text = candidate if isinstance(candidate, str) else ""
        if isinstance(record, dict):
            entry_text = record.get("output_text")
            if isinstance(entry_text, str) and entry_text.strip():
                text = entry_text
        tool_outputs: list[Any] = []
        if isinstance(isolated_io, dict):
            raw_tool_outputs = isolated_io.get("last_tool_outputs", [])
            if isinstance(raw_tool_outputs, list):
                tool_outputs = list(raw_tool_outputs)
        if not tool_outputs and isinstance(record, dict):
            maybe_tools = record.get("tool_outputs")
            if isinstance(maybe_tools, list):
                tool_outputs = list(maybe_tools)
        artifacts_patch = self._artifact_patch_from_record(isolated_state, record)
        return {
            "source_kind": "subagent",
            "source_id": agent_id,
            "output_text": text or "",
            "parsed": None,
            "tool_outputs": tool_outputs,
            "artifacts_patch": artifacts_patch,
        }

    def collect_workflow_execution_result(
        self,
        isolated_state: RuntimeState,
        workflow_id: str,
    ) -> Dict[str, Any]:
        isolated_io = isolated_state.get("io", {})
        record = self._latest_execution_record(isolated_state)
        text = ""
        tool_outputs: list[Any] = []
        if isinstance(record, dict):
            maybe_text = record.get("output_text")
            if isinstance(maybe_text, str):
                text = maybe_text
            maybe_tools = record.get("tool_outputs")
            if isinstance(maybe_tools, list):
                tool_outputs = list(maybe_tools)
        if not text:
            text = (
                isolated_state.get("output", {}).get("final_text")
                or isolated_io.get("last_model_output")
                or ""
            )
        if not tool_outputs and isinstance(isolated_io, dict):
            maybe_tools = isolated_io.get("last_tool_outputs", [])
            if isinstance(maybe_tools, list):
                tool_outputs = list(maybe_tools)

        artifacts_patch = self._artifact_patch_from_all_execution_keys(isolated_state)
        workflow_output = WorkflowRunnerOutput(
            status="success" if text else "partial",
            final_text=text or "",
            artifacts=artifacts_patch,
            trace=[],
        )
        return {
            "source_kind": "workflow",
            "source_id": workflow_id,
            "output_text": workflow_output.final_text,
            "parsed": None,
            "tool_outputs": tool_outputs,
            "artifacts_patch": workflow_output.artifacts,
        }

    def deliver_execution_result_to_supervisor(
        self,
        supervisor_state: RuntimeState,
        result: Dict[str, Any],
    ) -> None:
        text = result.get("output_text", "")
        supervisor_state["io"]["last_model_output"] = text
        supervisor_state["io"]["last_execution_output"] = text
        supervisor_state["io"]["last_tool_outputs"] = list(result.get("tool_outputs", []))
        parsed_payload: Optional[Dict[str, Any]] = None
        base_parsed = result.get("parsed")
        if isinstance(base_parsed, dict):
            parsed_payload = dict(base_parsed)
        patch = result.get("artifacts_patch", {})
        if isinstance(patch, dict) and patch:
            if parsed_payload is None:
                parsed_payload = {}
            existing_patch = parsed_payload.get("artifacts")
            if isinstance(existing_patch, dict):
                merged_patch = dict(existing_patch)
                merged_patch.update(patch)
                parsed_payload["artifacts"] = merged_patch
            else:
                parsed_payload["artifacts"] = dict(patch)

        self._apply_agent_output(
            supervisor_state,
            str(result.get("source_id", "")),
            str(result.get("source_id", "")),
            text if isinstance(text, str) else "",
            parsed_payload,
            source_kind=str(result.get("source_kind", "agent")),
        )

    @staticmethod
    def _latest_execution_record(
        state: RuntimeState,
        *,
        source_id: str | None = None,
    ) -> dict[str, Any] | None:
        records = state.get("executions")
        if not isinstance(records, list):
            return None
        for record in reversed(records):
            if not isinstance(record, dict):
                continue
            if source_id is not None and record.get("source_id") != source_id:
                continue
            return record
        return None

    @staticmethod
    def _artifact_patch_from_record(
        state: RuntimeState,
        record: dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        keys = record.get("artifact_keys")
        if not isinstance(keys, list):
            return {}
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            return {}
        return {
            key: artifacts[key]
            for key in keys
            if isinstance(key, str) and key in artifacts
        }

    @classmethod
    def _artifact_patch_from_all_execution_keys(cls, state: RuntimeState) -> Dict[str, Any]:
        records = state.get("executions")
        if not isinstance(records, list):
            return {}
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            return {}
        keys: list[str] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            artifact_keys = record.get("artifact_keys")
            if not isinstance(artifact_keys, list):
                continue
            for key in artifact_keys:
                if isinstance(key, str) and key not in seen:
                    seen.add(key)
                    keys.append(key)
        return {key: artifacts[key] for key in keys if key in artifacts}
