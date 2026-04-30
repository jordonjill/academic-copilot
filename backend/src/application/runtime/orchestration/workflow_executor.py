from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Awaitable, Callable, Optional

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.orchestration.workflow_router import WorkflowRuntime
from src.application.runtime.utils.env_utils import read_env_float

StepCallback = Callable[[dict[str, Any]], Any]
NodeRunner = Callable[[RuntimeState, str, str, Optional[dict[str, Any]]], None]
AsyncNodeRunner = Callable[[RuntimeState, str, str, Optional[dict[str, Any]]], Awaitable[None]]
AsyncStepEmitter = Callable[[Optional[StepCallback], dict[str, Any]], Awaitable[None]]
BestTextResolver = Callable[[RuntimeState], Optional[str]]


class WorkflowExecutor:
    def __init__(
        self,
        *,
        registry: ConfigRegistry,
        tool_budget: Any,
        logger: logging.Logger,
        default_wall_timeout_seconds: float,
    ) -> None:
        self._registry = registry
        self._tool_budget = tool_budget
        self._logger = logger
        self._default_wall_timeout_seconds = default_wall_timeout_seconds

    def run_sync(
        self,
        *,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
        execute_node: NodeRunner,
        best_available_final_text: BestTextResolver,
    ) -> None:
        state["runtime"]["mode"] = "workflow"
        state["runtime"]["workflow_id"] = workflow_id
        spec = self._registry.workflows[workflow_id]
        max_steps = spec.resolved_max_steps()
        state["runtime"]["max_steps"] = max_steps
        state["runtime"]["max_loops"] = spec.resolved_max_loops(max_steps=max_steps)
        runtime = WorkflowRuntime(spec, agent_runner=None)
        tool_budget = self._tool_budget.ensure_workflow_tool_budget(state, workflow_id)

        current_node = spec.entry_node
        visit_counts: dict[str, int] = {current_node: 1}
        max_wall_seconds = read_env_float(
            "WORKFLOW_MAX_WALL_TIME_SECONDS",
            self._default_wall_timeout_seconds,
        )
        started = perf_counter()

        while True:
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Workflow timeout after {max_wall_seconds:.1f} seconds")
            state["runtime"]["current_node"] = current_node

            node_spec = spec.nodes.get(current_node)
            if not node_spec:
                raise RuntimeError(f"Node not found in workflow: {current_node}")

            node_type = node_spec.get("type")
            if node_type == "terminal":
                fallback = best_available_final_text(state)
                if fallback:
                    state["output"]["final_text"] = fallback
                break

            runtime.enforce_limits(
                {
                    "_step_count": state["runtime"]["step_count"],
                    "_loop_count": state["runtime"]["loop_count"],
                }
            )
            self._logger.info(
                "workflow.step workflow_id=%s current_node=%s step_count=%s loop_count=%s",
                workflow_id,
                current_node,
                state["runtime"].get("step_count", 0),
                state["runtime"].get("loop_count", 0),
            )

            if node_type != "agent":
                raise RuntimeError(f"Unsupported node type: {node_type}")

            agent_id = node_spec.get("agent_id")
            if not isinstance(agent_id, str):
                raise RuntimeError(f"Invalid agent_id for node: {current_node}")

            if runtime.is_node_visit_saturated(current_node, visit_counts):
                next_node = runtime.next_node_for_saturated_node(current_node)
                if next_node == current_node:
                    raise RuntimeError(
                        f"Node visit limit reached but no forward edge available for node: {current_node}"
                    )
                self._logger.warning(
                    "workflow.node_visit_saturated_fallback workflow_id=%s current_node=%s to=%s "
                    "visit_count=%s limit=%s",
                    workflow_id,
                    current_node,
                    next_node,
                    visit_counts.get(current_node, 0),
                    spec.limits.get(f"max_visits_{current_node}", spec.limits.get(f"max_{current_node}")),
                )
                runtime.assert_transition_allowed(current_node, next_node)
                state["runtime"]["current_node"] = next_node
                if visit_counts.get(next_node, 0) > 0:
                    state["runtime"]["loop_count"] += 1
                visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
                current_node = next_node
                continue

            execute_node(state, current_node, agent_id, tool_budget)
            state["runtime"]["step_count"] += 1

            next_node = runtime.next_node(current_node, state)
            runtime.assert_transition_allowed(current_node, next_node)
            state["runtime"]["current_node"] = next_node

            if step_callback is not None:
                step_callback(
                    {
                        "node_name": current_node,
                        "step_number": state["runtime"]["step_count"],
                        "agent_id": agent_id,
                        "next_node": next_node,
                        "supervisor_reason": "workflow auto transition",
                        "last_model_output": state["io"].get("last_model_output"),
                        "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
                    }
                )

            if visit_counts.get(next_node, 0) > 0:
                state["runtime"]["loop_count"] += 1
            visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
            current_node = next_node

    async def run_async(
        self,
        *,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
        execute_node_async: AsyncNodeRunner,
        emit_step_callback_async: AsyncStepEmitter,
        best_available_final_text: BestTextResolver,
    ) -> None:
        state["runtime"]["mode"] = "workflow"
        state["runtime"]["workflow_id"] = workflow_id
        spec = self._registry.workflows[workflow_id]
        max_steps = spec.resolved_max_steps()
        state["runtime"]["max_steps"] = max_steps
        state["runtime"]["max_loops"] = spec.resolved_max_loops(max_steps=max_steps)
        runtime = WorkflowRuntime(spec, agent_runner=None)
        tool_budget = self._tool_budget.ensure_workflow_tool_budget(state, workflow_id)

        current_node = spec.entry_node
        visit_counts: dict[str, int] = {current_node: 1}
        max_wall_seconds = read_env_float(
            "WORKFLOW_MAX_WALL_TIME_SECONDS",
            self._default_wall_timeout_seconds,
        )
        started = perf_counter()

        while True:
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Workflow timeout after {max_wall_seconds:.1f} seconds")
            state["runtime"]["current_node"] = current_node

            node_spec = spec.nodes.get(current_node)
            if not node_spec:
                raise RuntimeError(f"Node not found in workflow: {current_node}")

            node_type = node_spec.get("type")
            if node_type == "terminal":
                fallback = best_available_final_text(state)
                if fallback:
                    state["output"]["final_text"] = fallback
                break

            runtime.enforce_limits(
                {
                    "_step_count": state["runtime"]["step_count"],
                    "_loop_count": state["runtime"]["loop_count"],
                }
            )
            self._logger.info(
                "workflow.step workflow_id=%s current_node=%s step_count=%s loop_count=%s",
                workflow_id,
                current_node,
                state["runtime"].get("step_count", 0),
                state["runtime"].get("loop_count", 0),
            )

            if node_type != "agent":
                raise RuntimeError(f"Unsupported node type: {node_type}")

            agent_id = node_spec.get("agent_id")
            if not isinstance(agent_id, str):
                raise RuntimeError(f"Invalid agent_id for node: {current_node}")

            if runtime.is_node_visit_saturated(current_node, visit_counts):
                next_node = runtime.next_node_for_saturated_node(current_node)
                if next_node == current_node:
                    raise RuntimeError(
                        f"Node visit limit reached but no forward edge available for node: {current_node}"
                    )
                self._logger.warning(
                    "workflow.node_visit_saturated_fallback workflow_id=%s current_node=%s to=%s "
                    "visit_count=%s limit=%s",
                    workflow_id,
                    current_node,
                    next_node,
                    visit_counts.get(current_node, 0),
                    spec.limits.get(f"max_visits_{current_node}", spec.limits.get(f"max_{current_node}")),
                )
                runtime.assert_transition_allowed(current_node, next_node)
                state["runtime"]["current_node"] = next_node
                if visit_counts.get(next_node, 0) > 0:
                    state["runtime"]["loop_count"] += 1
                visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
                current_node = next_node
                continue

            await execute_node_async(state, current_node, agent_id, tool_budget)
            state["runtime"]["step_count"] += 1

            next_node = runtime.next_node(current_node, state)
            runtime.assert_transition_allowed(current_node, next_node)
            state["runtime"]["current_node"] = next_node

            await emit_step_callback_async(
                step_callback,
                {
                    "node_name": current_node,
                    "step_number": state["runtime"]["step_count"],
                    "agent_id": agent_id,
                    "next_node": next_node,
                    "supervisor_reason": "workflow auto transition",
                    "last_model_output": state["io"].get("last_model_output"),
                    "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
                },
            )

            if visit_counts.get(next_node, 0) > 0:
                state["runtime"]["loop_count"] += 1
            visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
            current_node = next_node
