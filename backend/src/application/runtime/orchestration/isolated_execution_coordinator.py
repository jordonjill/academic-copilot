from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.io_models import WorkflowRunnerInput
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.execution.agent_execution_service import AgentExecutionService
from src.application.runtime.execution.isolation_facility import IsolationFacility

StepCallback = Callable[[dict[str, Any]], Any]
SyncWorkflowRunner = Callable[[RuntimeState, str, Optional[StepCallback]], None]
AsyncWorkflowRunner = Callable[[RuntimeState, str, Optional[StepCallback]], Awaitable[None]]


class IsolatedExecutionCoordinator:
    def __init__(
        self,
        *,
        registry: ConfigRegistry,
        isolation: IsolationFacility,
        agent_execution: AgentExecutionService,
        run_workflow_sync: SyncWorkflowRunner,
        run_workflow_async: AsyncWorkflowRunner,
    ) -> None:
        self._registry = registry
        self._isolation = isolation
        self._agent_execution = agent_execution
        self._run_workflow_sync = run_workflow_sync
        self._run_workflow_async = run_workflow_async

    def execute_workflow_agent_isolated_for_executor(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        tool_budget: Optional[dict[str, Any]],
    ) -> None:
        self.execute_workflow_agent_isolated(
            state,
            node_name,
            agent_id,
            tool_budget=tool_budget,
        )

    async def execute_workflow_agent_isolated_async_for_executor(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        tool_budget: Optional[dict[str, Any]],
    ) -> None:
        await self.execute_workflow_agent_isolated_async(
            state,
            node_name,
            agent_id,
            tool_budget=tool_budget,
        )

    def execute_workflow_agent_isolated(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        instruction = self._build_workflow_node_instruction(state, node_name, agent_id)
        isolated_state = self._isolation.build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=None,
            node_name=node_name,
            runtime_mode=str(state.get("runtime", {}).get("mode", "subagent")),
        )
        self._agent_execution.execute_agent(
            isolated_state,
            node_name,
            agent_id,
            tool_budget=tool_budget,
        )
        result = self._isolation.collect_subagent_execution_result(isolated_state, agent_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    async def execute_workflow_agent_isolated_async(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        instruction = self._build_workflow_node_instruction(state, node_name, agent_id)
        isolated_state = self._isolation.build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=None,
            node_name=node_name,
            runtime_mode=str(state.get("runtime", {}).get("mode", "subagent")),
        )
        await self._agent_execution.execute_agent_async(
            isolated_state,
            node_name,
            agent_id,
            tool_budget=tool_budget,
        )
        result = self._isolation.collect_subagent_execution_result(isolated_state, agent_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    def execute_subagent_isolated(
        self,
        state: RuntimeState,
        agent_id: str,
        instruction: Any,
        input_artifact_keys: Optional[list[str]] = None,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        isolated_state = self._isolation.build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=input_artifact_keys,
            node_name=agent_id,
            runtime_mode=str(state.get("runtime", {}).get("mode", "subagent")),
        )
        self._agent_execution.execute_agent(
            isolated_state,
            agent_id,
            agent_id,
            tool_budget=tool_budget,
        )
        result = self._isolation.collect_subagent_execution_result(isolated_state, agent_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    async def execute_subagent_isolated_async(
        self,
        state: RuntimeState,
        agent_id: str,
        instruction: Any,
        input_artifact_keys: Optional[list[str]] = None,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        isolated_state = self._isolation.build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=input_artifact_keys,
            node_name=agent_id,
            runtime_mode=str(state.get("runtime", {}).get("mode", "subagent")),
        )
        await self._agent_execution.execute_agent_async(
            isolated_state,
            agent_id,
            agent_id,
            tool_budget=tool_budget,
        )
        result = self._isolation.collect_subagent_execution_result(isolated_state, agent_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    def execute_workflow_isolated(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        workflow_spec = self._registry.workflows[workflow_id]
        max_steps = workflow_spec.resolved_max_steps()
        runner_input = WorkflowRunnerInput(
            workflow_id=workflow_id,
            instruction=state["input"].get("user_text", ""),
            seed_artifacts=self._isolation.select_input_artifacts(state, input_artifact_keys=None),
            limits={
                "max_steps": max_steps,
                "max_loops": workflow_spec.resolved_max_loops(max_steps=max_steps),
            },
        )
        isolated_state = self._isolation.build_isolated_workflow_state(
            state,
            workflow_id,
            runner_input=runner_input,
        )
        self._run_workflow_sync(isolated_state, workflow_id, step_callback)
        state["runtime"]["step_count"] += int(isolated_state.get("runtime", {}).get("step_count", 0))
        state["runtime"]["loop_count"] += int(isolated_state.get("runtime", {}).get("loop_count", 0))
        result = self._isolation.collect_workflow_execution_result(isolated_state, workflow_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    async def execute_workflow_isolated_async(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        workflow_spec = self._registry.workflows[workflow_id]
        max_steps = workflow_spec.resolved_max_steps()
        runner_input = WorkflowRunnerInput(
            workflow_id=workflow_id,
            instruction=state["input"].get("user_text", ""),
            seed_artifacts=self._isolation.select_input_artifacts(state, input_artifact_keys=None),
            limits={
                "max_steps": max_steps,
                "max_loops": workflow_spec.resolved_max_loops(max_steps=max_steps),
            },
        )
        isolated_state = self._isolation.build_isolated_workflow_state(
            state,
            workflow_id,
            runner_input=runner_input,
        )
        await self._run_workflow_async(isolated_state, workflow_id, step_callback)
        state["runtime"]["step_count"] += int(isolated_state.get("runtime", {}).get("step_count", 0))
        state["runtime"]["loop_count"] += int(isolated_state.get("runtime", {}).get("loop_count", 0))
        result = self._isolation.collect_workflow_execution_result(isolated_state, workflow_id)
        self._isolation.deliver_execution_result_to_supervisor(state, result)

    def _build_workflow_node_instruction(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
    ) -> str:
        user_text = state.get("input", {}).get("user_text", "")
        return f"[workflow_node={node_name} agent={agent_id}] User request: {user_text}"
