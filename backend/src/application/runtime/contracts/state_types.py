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
    max_steps: NotRequired[int]
    loop_count: int
    max_loops: NotRequired[int]
    status: str
    token_usage: NotRequired[dict[str, Any]]
    tool_budget: NotRequired[dict[str, Any]]


class RuntimeIOState(TypedDict):
    last_model_output: Optional[str]
    last_execution_output: Optional[str]
    last_tool_outputs: list[Any]


class RuntimeArtifactsState(TypedDict):
    topic: Optional[str]


class RuntimeTaskState(TypedDict):
    instruction: NotRequired[str]
    input_artifact_keys: NotRequired[list[str]]
    input_artifacts: NotRequired[dict[str, Any]]
    task_input: NotRequired[dict[str, Any]]
    workflow_runner_input: NotRequired[dict[str, Any]]


class RuntimeExecutionRecord(TypedDict):
    source_kind: str
    source_id: str
    node: NotRequired[str]
    action: NotRequired[str]
    target: NotRequired[Optional[str]]
    instruction: NotRequired[str]
    reason: NotRequired[str]
    step_count: NotRequired[int]
    output_text: NotRequired[str]
    output_preview: NotRequired[str]
    artifact_keys: NotRequired[list[str]]
    tool_outputs: NotRequired[list[Any]]
    status: NotRequired[str]
    confidence: NotRequired[float]
    errors: NotRequired[list[str]]


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
    task: NotRequired[RuntimeTaskState]
    executions: NotRequired[list[RuntimeExecutionRecord]]
    output: RuntimeOutputState
    errors: RuntimeErrorsState
