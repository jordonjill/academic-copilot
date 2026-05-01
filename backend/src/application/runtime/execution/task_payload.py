from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from src.application.runtime.contracts.io_models import AgentTaskInput
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.providers.context_facility import ContextFacility

_INTERNAL_ARTIFACT_KEYS = {
    "shared",
    "execution_trace",
    "supervisor_instruction",
    "task_input",
    "workflow_runner_input",
}


def task_input_from_state(state: RuntimeState) -> AgentTaskInput | None:
    task = state.get("task")
    if not isinstance(task, dict):
        return None

    raw_task_input = task.get("task_input")
    if isinstance(raw_task_input, dict):
        try:
            return AgentTaskInput.model_validate(raw_task_input)
        except Exception:
            pass

    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return None

    input_artifacts = task.get("input_artifacts")
    if not isinstance(input_artifacts, dict):
        input_artifacts = _prompt_artifacts(state)

    return AgentTaskInput(
        task_id=f"task:{state.get('runtime', {}).get('step_count', 0)}",
        instruction=instruction.strip(),
        input_artifacts=dict(input_artifacts),
    )


def render_chain_payload(
    state: RuntimeState,
    context_facility: ContextFacility,
) -> dict[str, Any]:
    task_input = task_input_from_state(state)
    if task_input is not None:
        artifacts = dict(task_input.input_artifacts)
        instruction = task_input.instruction
        return {
            "user_text": state["input"].get("user_text", ""),
            "artifacts": json.dumps(artifacts, ensure_ascii=False, default=str),
            "supervisor_instruction": instruction,
        }
    else:
        messages_text = context_facility.messages_to_text(
            state["context"].get("messages", []),
            scope="default",
        )
        artifacts = _prompt_artifacts(state)
        instruction = ""

    return {
        "user_text": state["input"].get("user_text", ""),
        "messages": messages_text,
        "artifacts": json.dumps(artifacts, ensure_ascii=False, default=str),
        "supervisor_instruction": instruction,
    }


def render_react_messages(
    state: RuntimeState,
    *,
    agent_id: str,
    node_name: str,
    runtime_mode: str,
) -> list[BaseMessage]:
    task_input = task_input_from_state(state)
    if task_input is None:
        return list(state["context"].get("messages", []))
    return [
        HumanMessage(
            content=build_task_input_envelope(
                task_input,
                agent_id=agent_id,
                node_name=node_name,
                runtime_mode=runtime_mode,
            )
        )
    ]


def build_task_input_envelope(
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
    return (
        "[TASK_INPUT_V1]\n"
        "Use this structured input only; treat absent fields as unavailable.\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _prompt_artifacts(state: RuntimeState) -> dict[str, Any]:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    return {
        key: value
        for key, value in artifacts.items()
        if isinstance(key, str) and key not in _INTERNAL_ARTIFACT_KEYS
    }
