from __future__ import annotations

from typing import Any, NotRequired, Optional, TypedDict

from langchain_core.messages import BaseMessage


class RuntimeInputState(TypedDict):
    user_text: str
    user_id: str
    session_id: str


class RuntimeContextState(TypedDict):
    messages: list[BaseMessage]
    memory_summary: str


class RuntimeMetaState(TypedDict):
    mode: str
    workflow_id: Optional[str]
    current_node: Optional[str]
    step_count: int
    loop_count: int
    status: str


class RuntimeIOState(TypedDict):
    last_model_output: Optional[str]
    last_tool_outputs: list[Any]


class RuntimeArtifactsState(TypedDict):
    topic: Optional[str]
    shared: dict[str, Any]
    execution_trace: list[dict[str, Any]]
    supervisor_instruction: NotRequired[str]
    task_input: NotRequired[dict[str, Any]]
    workflow_runner_input: NotRequired[dict[str, Any]]


class RuntimeOutputState(TypedDict):
    final_text: Optional[str]
    final_structured: Optional[dict[str, Any]]


class RuntimeErrorsState(TypedDict):
    last_error: Optional[str]


class RuntimeState(TypedDict):
    input: RuntimeInputState
    context: RuntimeContextState
    runtime: RuntimeMetaState
    io: RuntimeIOState
    artifacts: RuntimeArtifactsState
    output: RuntimeOutputState
    errors: RuntimeErrorsState
