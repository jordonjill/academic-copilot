from __future__ import annotations

from time import perf_counter
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.spec_models import AgentSpec
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.utils.env_utils import read_env_float, read_env_int

StepCallback = Callable[[dict[str, Any]], Any]
SyncRunWorkflow = Callable[[RuntimeState, str, Optional[StepCallback]], None]
AsyncRunWorkflow = Callable[[RuntimeState, str, Optional[StepCallback]], Awaitable[None]]
SyncFinalize = Callable[[RuntimeState, Optional[str]], None]
AsyncFinalize = Callable[
    [RuntimeState, Optional[str], Optional[Callable[[str], Awaitable[None]]]],
    Awaitable[None],
]
SyncChitchat = Callable[[RuntimeState], None]
AsyncChitchat = Callable[[RuntimeState], Awaitable[None]]
SyncDecide = Callable[[RuntimeState, AgentSpec, Optional[str]], dict[str, Any]]
AsyncDecide = Callable[
    [RuntimeState, AgentSpec, Optional[str], Optional[Callable[[str], Awaitable[None]]]],
    Awaitable[dict[str, Any]],
]
ResolveWorkflowTarget = Callable[[dict[str, Any], RuntimeState], Optional[str]]
ResolveSubagentTarget = Callable[[dict[str, Any]], Optional[str]]
AppendExecutionTrace = Callable[..., None]
SyncExecuteWorkflowIsolated = Callable[[RuntimeState, str, Optional[StepCallback], Optional[dict[str, Any]]], None]
AsyncExecuteWorkflowIsolated = Callable[[RuntimeState, str, Optional[StepCallback], Optional[dict[str, Any]]], Awaitable[None]]
SyncExecuteSubagentIsolated = Callable[..., None]
AsyncExecuteSubagentIsolated = Callable[..., Awaitable[None]]
EnsureTurnBudget = Callable[[RuntimeState], dict[str, Any]]
AsyncEmitStepCallback = Callable[[Optional[StepCallback], dict[str, Any]], Awaitable[None]]
AsyncEmitRuntimeEvent = Callable[[dict[str, Any]], Awaitable[None]]


class SupervisorOrchestrator:
    def __init__(
        self,
        *,
        registry: ConfigRegistry,
        logger: Any,
        max_subagent_calls_env: str,
        default_wall_timeout_seconds: float,
    ) -> None:
        self._registry = registry
        self._logger = logger
        self._max_subagent_calls_env = max_subagent_calls_env
        self._default_wall_timeout_seconds = default_wall_timeout_seconds

    def run_sync(
        self,
        *,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
        supervisor_spec: Optional[AgentSpec],
        run_workflow: SyncRunWorkflow,
        finalize_with_supervisor: SyncFinalize,
        run_chitchat: SyncChitchat,
        decide_next_action: SyncDecide,
        resolve_workflow_target: ResolveWorkflowTarget,
        resolve_subagent_target: ResolveSubagentTarget,
        append_execution_trace: AppendExecutionTrace,
        execute_workflow_isolated: SyncExecuteWorkflowIsolated,
        execute_subagent_isolated: SyncExecuteSubagentIsolated,
        ensure_turn_tool_budget: EnsureTurnBudget,
    ) -> None:
        if requested_workflow_id:
            if requested_workflow_id not in self._registry.workflows:
                raise ValueError(f"Unknown workflow_id: {requested_workflow_id}")
            run_workflow(state, requested_workflow_id, step_callback)
            finalize_with_supervisor(state, requested_workflow_id)
            return

        if supervisor_spec is None or supervisor_spec.mode != "chain":
            run_chitchat(state)
            return

        max_steps = max(1, read_env_int("SUPERVISOR_MAX_STEPS", 8, minimum=1))
        max_subagent_calls_per_agent = read_env_int(self._max_subagent_calls_env, 2, minimum=0)
        max_subagent_calls_per_agent = max(0, max_subagent_calls_per_agent)
        max_wall_seconds = read_env_float(
            "SUPERVISOR_MAX_WALL_TIME_SECONDS",
            self._default_wall_timeout_seconds,
        )
        started = perf_counter()
        subagent_call_counts: dict[str, int] = {}
        workflow_calls_used = 0
        max_workflow_calls_per_turn = 1
        turn_tool_budget = ensure_turn_tool_budget(state)

        for _ in range(max_steps):
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Supervisor loop timeout after {max_wall_seconds:.1f} seconds")
            decision = decide_next_action(state, supervisor_spec, requested_workflow_id)
            action = decision.get("action")
            if not isinstance(action, str):
                action = "direct_reply"
            action = action.strip().lower()
            self._logger.info(
                "supervisor.decision action=%s target=%s done=%s step_count=%s",
                action,
                decision.get("target"),
                bool(decision.get("done")),
                state["runtime"].get("step_count", 0),
            )

            if action == "run_workflow":
                if workflow_calls_used >= max_workflow_calls_per_turn:
                    self._logger.warning(
                        "supervisor.workflow_limit_reached limit=%s",
                        max_workflow_calls_per_turn,
                    )
                    state["context"]["messages"].append(
                        SystemMessage(
                            content=(
                                f"Supervisor guardrail: workflow call limit reached "
                                f"({max_workflow_calls_per_turn}) in current turn. Choose another action."
                            )
                        )
                    )
                    continue
                workflow_id = resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self._registry.workflows:
                    append_execution_trace(
                        state,
                        action=action,
                        target=workflow_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(decision.get("instruction") or ""),
                    )
                    inline_input_artifacts = decision.get("inline_input_artifacts")
                    execute_workflow_isolated(
                        state,
                        workflow_id,
                        step_callback,
                        inline_input_artifacts
                        if isinstance(inline_input_artifacts, dict)
                        else None,
                    )
                    workflow_calls_used += 1
                    continue
                action = "direct_reply"

            if action == "run_agent":
                agent_id = resolve_subagent_target(decision)
                if agent_id and agent_id in self._registry.agents and agent_id != supervisor_spec.id:
                    calls_used = subagent_call_counts.get(agent_id, 0)
                    if calls_used >= max_subagent_calls_per_agent:
                        self._logger.warning(
                            "supervisor.subagent_limit_reached agent_id=%s limit=%s",
                            agent_id,
                            max_subagent_calls_per_agent,
                        )
                        state["context"]["messages"].append(
                            SystemMessage(
                                content=(
                                    f"Supervisor guardrail: subagent '{agent_id}' call limit reached "
                                    f"({max_subagent_calls_per_agent}) in current turn. Choose another action."
                                )
                            )
                        )
                        continue

                    instruction = decision.get("instruction")
                    if isinstance(instruction, str) and instruction.strip():
                        state["artifacts"]["supervisor_instruction"] = instruction
                        state["context"]["messages"].append(
                            HumanMessage(content=f"Supervisor task for {agent_id}: {instruction}")
                        )
                    input_artifact_keys = decision.get("input_artifact_keys")
                    inline_input_artifacts = decision.get("inline_input_artifacts")
                    execute_subagent_isolated(
                        state,
                        agent_id,
                        instruction,
                        input_artifact_keys if isinstance(input_artifact_keys, list) else None,
                        inline_input_artifacts if isinstance(inline_input_artifacts, dict) else None,
                        tool_budget=turn_tool_budget,
                    )
                    append_execution_trace(
                        state,
                        action=action,
                        target=agent_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(instruction or ""),
                    )
                    subagent_call_counts[agent_id] = calls_used + 1
                    state["runtime"]["step_count"] += 1
                    if step_callback is not None:
                        step_callback(
                            {
                                "node_name": agent_id,
                                "step_number": state["runtime"]["step_count"],
                                "agent_id": agent_id,
                                "next_node": None,
                                "supervisor_reason": "supervisor selected direct subagent execution",
                                "last_model_output": state["io"].get("last_model_output"),
                                "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
                            }
                        )
                    continue
                action = "direct_reply"

            if action == "direct_reply":
                final_text = decision.get("final_text")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = decision.get("message")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = state["io"].get("last_model_output") or "No output produced."
                append_execution_trace(
                    state,
                    action=action,
                    target=None,
                    reason=str(decision.get("reason") or ""),
                    instruction="",
                )
                state["io"]["last_model_output"] = final_text
                if bool(decision.get("done")):
                    state["output"]["final_text"] = final_text
                    state["context"]["messages"].append(AIMessage(content=final_text))
                    return
                continue

        if not state["output"].get("final_text"):
            fallback = state["io"].get("last_model_output") or "No output produced."
            state["output"]["final_text"] = fallback

    async def run_async(
        self,
        *,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
        supervisor_spec: Optional[AgentSpec],
        run_workflow_async: AsyncRunWorkflow,
        finalize_with_supervisor_async: AsyncFinalize,
        run_chitchat_async: AsyncChitchat,
        decide_next_action_async: AsyncDecide,
        resolve_workflow_target: ResolveWorkflowTarget,
        resolve_subagent_target: ResolveSubagentTarget,
        append_execution_trace: AppendExecutionTrace,
        execute_workflow_isolated_async: AsyncExecuteWorkflowIsolated,
        execute_subagent_isolated_async: AsyncExecuteSubagentIsolated,
        ensure_turn_tool_budget: EnsureTurnBudget,
        emit_step_callback_async: AsyncEmitStepCallback,
        emit_runtime_event_async: AsyncEmitRuntimeEvent,
    ) -> None:
        stream_event_index = 0

        async def _emit_supervisor_delta(delta: str) -> None:
            nonlocal stream_event_index
            if not isinstance(delta, str) or not delta:
                return
            stream_event_index += 1
            await emit_runtime_event_async(
                {
                    "type": "delta",
                    "source": "supervisor",
                    "delta": delta,
                    "index": stream_event_index,
                }
            )

        if requested_workflow_id:
            if requested_workflow_id not in self._registry.workflows:
                raise ValueError(f"Unknown workflow_id: {requested_workflow_id}")
            await run_workflow_async(state, requested_workflow_id, step_callback)
            await finalize_with_supervisor_async(state, requested_workflow_id, _emit_supervisor_delta)
            return

        if supervisor_spec is None or supervisor_spec.mode != "chain":
            await run_chitchat_async(state)
            return

        max_steps = max(1, read_env_int("SUPERVISOR_MAX_STEPS", 8, minimum=1))
        max_subagent_calls_per_agent = read_env_int(self._max_subagent_calls_env, 2, minimum=0)
        max_subagent_calls_per_agent = max(0, max_subagent_calls_per_agent)
        max_wall_seconds = read_env_float(
            "SUPERVISOR_MAX_WALL_TIME_SECONDS",
            self._default_wall_timeout_seconds,
        )
        started = perf_counter()
        subagent_call_counts: dict[str, int] = {}
        workflow_calls_used = 0
        max_workflow_calls_per_turn = 1
        turn_tool_budget = ensure_turn_tool_budget(state)

        for _ in range(max_steps):
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Supervisor loop timeout after {max_wall_seconds:.1f} seconds")
            decision = await decide_next_action_async(
                state,
                supervisor_spec,
                requested_workflow_id,
                _emit_supervisor_delta,
            )
            action = decision.get("action")
            if not isinstance(action, str):
                action = "direct_reply"
            action = action.strip().lower()
            self._logger.info(
                "supervisor.decision action=%s target=%s done=%s step_count=%s",
                action,
                decision.get("target"),
                bool(decision.get("done")),
                state["runtime"].get("step_count", 0),
            )

            if action == "run_workflow":
                if workflow_calls_used >= max_workflow_calls_per_turn:
                    self._logger.warning(
                        "supervisor.workflow_limit_reached limit=%s",
                        max_workflow_calls_per_turn,
                    )
                    state["context"]["messages"].append(
                        SystemMessage(
                            content=(
                                f"Supervisor guardrail: workflow call limit reached "
                                f"({max_workflow_calls_per_turn}) in current turn. Choose another action."
                            )
                        )
                    )
                    continue
                workflow_id = resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self._registry.workflows:
                    append_execution_trace(
                        state,
                        action=action,
                        target=workflow_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(decision.get("instruction") or ""),
                    )
                    inline_input_artifacts = decision.get("inline_input_artifacts")
                    await execute_workflow_isolated_async(
                        state,
                        workflow_id,
                        step_callback,
                        inline_input_artifacts
                        if isinstance(inline_input_artifacts, dict)
                        else None,
                    )
                    workflow_calls_used += 1
                    continue
                action = "direct_reply"

            if action == "run_agent":
                agent_id = resolve_subagent_target(decision)
                if agent_id and agent_id in self._registry.agents and agent_id != supervisor_spec.id:
                    calls_used = subagent_call_counts.get(agent_id, 0)
                    if calls_used >= max_subagent_calls_per_agent:
                        self._logger.warning(
                            "supervisor.subagent_limit_reached agent_id=%s limit=%s",
                            agent_id,
                            max_subagent_calls_per_agent,
                        )
                        state["context"]["messages"].append(
                            SystemMessage(
                                content=(
                                    f"Supervisor guardrail: subagent '{agent_id}' call limit reached "
                                    f"({max_subagent_calls_per_agent}) in current turn. Choose another action."
                                )
                            )
                        )
                        continue

                    instruction = decision.get("instruction")
                    if isinstance(instruction, str) and instruction.strip():
                        state["artifacts"]["supervisor_instruction"] = instruction
                        state["context"]["messages"].append(
                            HumanMessage(content=f"Supervisor task for {agent_id}: {instruction}")
                        )
                    input_artifact_keys = decision.get("input_artifact_keys")
                    inline_input_artifacts = decision.get("inline_input_artifacts")
                    await execute_subagent_isolated_async(
                        state,
                        agent_id,
                        instruction,
                        input_artifact_keys if isinstance(input_artifact_keys, list) else None,
                        inline_input_artifacts if isinstance(inline_input_artifacts, dict) else None,
                        tool_budget=turn_tool_budget,
                    )
                    append_execution_trace(
                        state,
                        action=action,
                        target=agent_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(instruction or ""),
                    )
                    subagent_call_counts[agent_id] = calls_used + 1
                    state["runtime"]["step_count"] += 1
                    await emit_step_callback_async(
                        step_callback,
                        {
                            "node_name": agent_id,
                            "step_number": state["runtime"]["step_count"],
                            "agent_id": agent_id,
                            "next_node": None,
                            "supervisor_reason": "supervisor selected direct subagent execution",
                            "last_model_output": state["io"].get("last_model_output"),
                            "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
                        },
                    )
                    continue
                action = "direct_reply"

            if action == "direct_reply":
                final_text = decision.get("final_text")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = decision.get("message")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = state["io"].get("last_model_output") or "No output produced."
                append_execution_trace(
                    state,
                    action=action,
                    target=None,
                    reason=str(decision.get("reason") or ""),
                    instruction="",
                )
                state["io"]["last_model_output"] = final_text
                if bool(decision.get("done")):
                    state["output"]["final_text"] = final_text
                    state["context"]["messages"].append(AIMessage(content=final_text))
                    return
                continue

        if not state["output"].get("final_text"):
            fallback = state["io"].get("last_model_output") or "No output produced."
            state["output"]["final_text"] = fallback
