from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.application.runtime.contracts.io_models import AgentTaskInput, WorkflowRunnerInput, WorkflowRunnerOutput
from src.application.runtime.contracts.state_types import RuntimeState


class IsolationFacility:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        apply_agent_output: Callable[
            [RuntimeState, str, str, str, Optional[Dict[str, Any]]],
            None,
        ],
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
        node_name: Optional[str] = None,
        runtime_mode: str = "subagent",
    ) -> RuntimeState:
        task_text = ""
        if isinstance(instruction, str) and instruction.strip():
            task_text = instruction.strip()
        if not task_text:
            task_text = parent_state["input"].get("user_text", "")
        selected_input_artifacts = self.select_input_artifacts(parent_state, input_artifact_keys)
        task_input = AgentTaskInput(
            task_id=f"{agent_id}:{parent_state['runtime'].get('step_count', 0)}",
            instruction=task_text,
            input_artifacts=selected_input_artifacts,
        )
        instruction_envelope = self.build_task_input_envelope(
            task_input,
            agent_id=agent_id,
            node_name=node_name or agent_id,
            runtime_mode=runtime_mode,
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
                "messages": [HumanMessage(content=instruction_envelope)],
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
                "shared": {},
                "execution_trace": [],
                "supervisor_instruction": task_text,
                "task_input": task_input.model_dump(),
                **selected_input_artifacts,
            },
            "output": {
                "final_text": None,
                "final_structured": None,
            },
            "errors": {
                "last_error": None,
            },
        }

    def build_task_input_envelope(
        self,
        task_input: AgentTaskInput,
        *,
        agent_id: str,
        node_name: str,
        runtime_mode: str,
    ) -> str:
        payload = {
            "protocol": "task_input_v1",
            "agent_id": agent_id,
            "node_name": node_name,
            "runtime_mode": runtime_mode,
            "task_input": task_input.model_dump(),
        }
        import json

        return (
            "[TASK_INPUT_V1]\n"
            "Use this structured input only; treat absent fields as unavailable.\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}"
        )

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

    def build_isolated_workflow_state(
        self,
        parent_state: RuntimeState,
        workflow_id: str,
        runner_input: Optional[WorkflowRunnerInput] = None,
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
                seed_artifacts=self.select_input_artifacts(parent_state, input_artifact_keys=None),
            )
        return {
            "input": {
                "user_text": user_text,
                "user_id": parent_state["input"].get("user_id", ""),
                "session_id": parent_state["input"].get("session_id", ""),
            },
            "context": {
                "messages": [HumanMessage(content=user_text)],
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
                "shared": {},
                "execution_trace": [],
                "workflow_runner_input": runner_input.model_dump(),
                **runner_input.seed_artifacts,
            },
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
        isolated_artifacts = isolated_state.get("artifacts", {})
        isolated_shared = (
            isolated_artifacts.get("shared", {})
            if isinstance(isolated_artifacts, dict)
            else {}
        )
        shared_entry = (
            isolated_shared.get(agent_id)
            if isinstance(isolated_shared, dict)
            else None
        )
        text = isolated_io.get("last_model_output") if isinstance(isolated_io, dict) else None
        if not isinstance(text, str) or not text.strip():
            output_state = isolated_state.get("output", {})
            candidate = output_state.get("final_text") if isinstance(output_state, dict) else None
            text = candidate if isinstance(candidate, str) else ""
        parsed: Optional[Dict[str, Any]] = None
        if isinstance(shared_entry, dict):
            entry_text = shared_entry.get("output_text")
            if isinstance(entry_text, str) and entry_text.strip():
                text = entry_text
            entry_parsed = shared_entry.get("parsed")
            if isinstance(entry_parsed, dict):
                parsed = entry_parsed
        tool_outputs: list[Any] = []
        if isinstance(isolated_io, dict):
            raw_tool_outputs = isolated_io.get("last_tool_outputs", [])
            if isinstance(raw_tool_outputs, list):
                tool_outputs = list(raw_tool_outputs)
        artifacts_patch: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            patch = parsed.get("artifacts")
            if isinstance(patch, dict):
                artifacts_patch = dict(patch)
        return {
            "source_kind": "subagent",
            "source_id": agent_id,
            "output_text": text or "",
            "parsed": parsed,
            "tool_outputs": tool_outputs,
            "artifacts_patch": artifacts_patch,
        }

    def collect_workflow_execution_result(
        self,
        isolated_state: RuntimeState,
        workflow_id: str,
    ) -> Dict[str, Any]:
        isolated_io = isolated_state.get("io", {})
        isolated_artifacts = isolated_state.get("artifacts", {})
        isolated_shared = (
            isolated_artifacts.get("shared", {})
            if isinstance(isolated_artifacts, dict)
            else {}
        )
        shared_entry: Any = None
        if isinstance(isolated_shared, dict):
            shared_entry = isolated_shared.get("reporter")
            if not isinstance(shared_entry, dict):
                for item in reversed(list(isolated_shared.values())):
                    if isinstance(item, dict):
                        shared_entry = item
                        break
        text = ""
        parsed: Optional[Dict[str, Any]] = None
        tool_outputs: list[Any] = []
        if isinstance(shared_entry, dict):
            maybe_text = shared_entry.get("output_text")
            if isinstance(maybe_text, str):
                text = maybe_text
            maybe_parsed = shared_entry.get("parsed")
            if isinstance(maybe_parsed, dict):
                parsed = maybe_parsed
            maybe_tools = shared_entry.get("tool_outputs")
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

        artifacts_patch: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            patch = parsed.get("artifacts")
            if isinstance(patch, dict):
                artifacts_patch = dict(patch)
        if not artifacts_patch and isinstance(isolated_artifacts, dict):
            artifacts_patch = {
                key: value
                for key, value in isolated_artifacts.items()
                if key not in {"topic", "shared", "supervisor_instruction"}
            }
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
            "parsed": parsed,
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
        if isinstance(text, str) and text.strip():
            supervisor_state["context"]["messages"].append(AIMessage(content=text))

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
        )
