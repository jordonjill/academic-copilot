from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from typing import Any, Callable, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.application.runtime.agent_factory import build_agent_from_spec
from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.spec_models import AgentSpec
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.execution.agent_execution_service import AgentExecutionService
from src.application.runtime.execution.isolation_facility import IsolationFacility
from src.application.runtime.execution.runtime_codec import RuntimeCodec
from src.application.runtime.execution.runtime_result_service import RuntimeResultService
from src.application.runtime.execution.tool_budget import ToolBudgetManager
from src.application.runtime.orchestration.isolated_execution_coordinator import (
    IsolatedExecutionCoordinator,
)
from src.application.runtime.orchestration.supervisor_decision_service import SupervisorDecisionService
from src.application.runtime.orchestration.supervisor_orchestrator import SupervisorOrchestrator
from src.application.runtime.orchestration.supervisor_payload_builder import SupervisorPayloadBuilder
from src.application.runtime.orchestration.workflow_executor import WorkflowExecutor
from src.application.runtime.providers.context_facility import ContextFacility
from src.application.runtime.providers.llm_provider import LLMProvider
from src.infrastructure.memory.ltm import load_ltm_profile_for_supervisor
from src.infrastructure.tools.registry import get_tool


CHITCHAT_SYSTEM = """You are Academic Copilot, an assistant for academic research tasks.
Respond concisely and in the same language as the user."""

StepCallback = Callable[[Dict[str, Any]], Any]
_SUPERVISOR_AGENT_ENV = "SUPERVISOR_AGENT_ID"
_DEFAULT_SUPERVISOR_AGENT_ID = "supervisor"
_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV = "SUPERVISOR_MAX_SUBAGENT_CALLS_PER_AGENT"
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{\w+\}")
logger = logging.getLogger(__name__)
_SUPERVISOR_DECISION_PARSER = JsonOutputParser()
_REACT_MAX_INTERNAL_STEPS_WORKFLOW = 8
_REACT_MAX_INTERNAL_STEPS_TURN = 6
_REACT_MAX_INTERNAL_STEPS_DEFAULT = 8
_REACT_MAX_INTERNAL_STEPS_BUDGET_EXHAUSTED = 2
_DEFAULT_SUPERVISOR_WALL_TIMEOUT_SECONDS = 900.0
_DEFAULT_WORKFLOW_WALL_TIMEOUT_SECONDS = 900.0


class RuntimeEngine:
    """Single-path runtime: supervisor -> config workflow (if available) -> final output."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self.registry = registry
        self._context_facility = ContextFacility.from_env()
        self._llm_provider = LLMProvider(
            registry=self.registry,
            env_placeholder_pattern=_ENV_PLACEHOLDER_PATTERN,
            create_chat_openai=lambda **kwargs: ChatOpenAI(**kwargs),
        )
        self._codec = RuntimeCodec(
            logger=logger,
            decision_parser=_SUPERVISOR_DECISION_PARSER,
        )
        self._result_service = RuntimeResultService(logger=logger)
        self._tool_budget = ToolBudgetManager(
            registry=self.registry,
            resolve_tool=self._resolve_tool,
            logger=logger,
        )
        self._workflow_executor = WorkflowExecutor(
            registry=self.registry,
            tool_budget=self._tool_budget,
            logger=logger,
            default_wall_timeout_seconds=_DEFAULT_WORKFLOW_WALL_TIMEOUT_SECONDS,
        )
        self._supervisor_payload = SupervisorPayloadBuilder(
            registry=self.registry,
            context_facility=self._context_facility,
            load_ltm_profile=lambda user_id: load_ltm_profile_for_supervisor(user_id),
        )
        self._agent_execution = AgentExecutionService(
            registry=self.registry,
            tool_budget=self._tool_budget,
            context_facility=self._context_facility,
            build_agent_from_spec_fn=lambda spec, llm, tool_resolver: build_agent_from_spec(
                spec,
                llm,
                tool_resolver,
            ),
            resolve_llm=lambda spec: self._resolve_llm(spec),
            apply_agent_output=self._result_service.apply_agent_output,
            coerce_text=lambda raw: self._codec.coerce_text(raw),
            try_parse_json=lambda text: self._codec.try_parse_json(text),
            normalize_agent_parsed_payload=lambda text, parsed: self._codec.normalize_agent_parsed_payload(
                text, parsed
            ),
            invoke_async=lambda runnable, payload, config: self._invoke_async(runnable, payload, config=config),
            extract_last_ai_text=lambda messages: self._codec.extract_last_ai_text(messages),
            extract_tool_outputs=lambda messages: self._codec.extract_tool_outputs(messages),
            react_max_internal_steps_default=_REACT_MAX_INTERNAL_STEPS_DEFAULT,
            react_max_internal_steps_workflow=_REACT_MAX_INTERNAL_STEPS_WORKFLOW,
            react_max_internal_steps_turn=_REACT_MAX_INTERNAL_STEPS_TURN,
            react_max_internal_steps_budget_exhausted=_REACT_MAX_INTERNAL_STEPS_BUDGET_EXHAUSTED,
        )
        self._supervisor_orchestrator = SupervisorOrchestrator(
            registry=self.registry,
            logger=logger,
            max_subagent_calls_env=_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV,
            default_wall_timeout_seconds=_DEFAULT_SUPERVISOR_WALL_TIMEOUT_SECONDS,
        )
        self._supervisor_decision = SupervisorDecisionService(
            registry=self.registry,
            logger=logger,
            build_agent_from_spec_fn=lambda spec, llm, tool_resolver: build_agent_from_spec(
                spec,
                llm,
                tool_resolver,
            ),
            resolve_supervisor_spec=lambda: self._resolve_supervisor_spec(),
            resolve_llm=lambda spec: self._resolve_llm(spec),
            resolve_tool=lambda tool_id: self._resolve_tool(tool_id),
            build_supervisor_payload=lambda state, supervisor_spec, requested_workflow_id, workflow_completed: self._supervisor_payload.build_supervisor_payload(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
                workflow_completed=workflow_completed,
            ),
            coerce_text=lambda raw: self._codec.coerce_text(raw),
            try_parse_supervisor_decision_json=lambda text: self._codec.try_parse_supervisor_decision_json(text),
            invoke_async=lambda runnable, payload, config: self._invoke_async(runnable, payload, config=config),
        )
        self._isolation = IsolationFacility(
            logger=logger,
            apply_agent_output=self._result_service.apply_agent_output,
        )
        self._isolation_coordinator = IsolatedExecutionCoordinator(
            registry=self.registry,
            isolation=self._isolation,
            agent_execution=self._agent_execution,
            run_workflow_sync=lambda state, workflow_id, step_callback: self._run_workflow(
                state,
                workflow_id,
                step_callback,
            ),
            run_workflow_async=lambda state, workflow_id, step_callback: self._run_workflow_async(
                state,
                workflow_id,
                step_callback,
            ),
        )

    def run_turn(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str] = None,
        step_callback: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        self._ensure_registry_loaded()
        state["runtime"]["status"] = "running"
        try:
            self._run_supervisor_loop(
                state=state,
                requested_workflow_id=requested_workflow_id,
                step_callback=step_callback,
            )
        except Exception as exc:
            state["runtime"]["status"] = "failed"
            state["errors"]["last_error"] = str(exc)
            raise

        state["runtime"]["status"] = "completed"
        return self._result_service.build_result(state)

    async def run_turn_async(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str] = None,
        step_callback: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        self._ensure_registry_loaded()
        state["runtime"]["status"] = "running"
        try:
            await self._run_supervisor_loop_async(
                state=state,
                requested_workflow_id=requested_workflow_id,
                step_callback=step_callback,
            )
        except Exception as exc:
            state["runtime"]["status"] = "failed"
            state["errors"]["last_error"] = str(exc)
            raise

        state["runtime"]["status"] = "completed"
        return self._result_service.build_result(state)

    def _ensure_registry_loaded(self) -> None:
        if not self.registry.agents and not self.registry.workflows:
            self.registry.reload()

    def _run_chitchat(self, state: RuntimeState) -> None:
        llm = self._resolve_default_llm()
        messages = list(state["context"].get("messages", []))
        response = llm.invoke([SystemMessage(content=CHITCHAT_SYSTEM)] + messages)
        text = self._codec.coerce_text(response)
        state["io"]["last_model_output"] = text
        state["output"]["final_text"] = text
        state["context"]["messages"].append(AIMessage(content=text))

    async def _run_chitchat_async(self, state: RuntimeState) -> None:
        llm = self._resolve_default_llm()
        messages = list(state["context"].get("messages", []))
        response = await self._invoke_async(llm, [SystemMessage(content=CHITCHAT_SYSTEM)] + messages)
        text = self._codec.coerce_text(response)
        state["io"]["last_model_output"] = text
        state["output"]["final_text"] = text
        state["context"]["messages"].append(AIMessage(content=text))

    def _run_supervisor_loop(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
    ) -> None:
        self._supervisor_orchestrator.run_sync(
            state=state,
            requested_workflow_id=requested_workflow_id,
            step_callback=step_callback,
            supervisor_spec=self._resolve_supervisor_spec(),
            run_workflow=self._run_workflow,
            finalize_with_supervisor=lambda s, wf: self._supervisor_decision.finalize_with_supervisor(
                state=s,
                requested_workflow_id=wf,
            ),
            run_chitchat=self._run_chitchat,
            decide_next_action=lambda s, spec, wf: self._supervisor_decision.decide_next_action(
                state=s,
                supervisor_spec=spec,
                requested_workflow_id=wf,
            ),
            resolve_workflow_target=self._supervisor_decision.resolve_workflow_target,
            resolve_subagent_target=self._supervisor_decision.resolve_subagent_target,
            append_execution_trace=self._supervisor_payload.append_execution_trace,
            execute_workflow_isolated=self._isolation_coordinator.execute_workflow_isolated,
            execute_subagent_isolated=self._isolation_coordinator.execute_subagent_isolated,
            ensure_turn_tool_budget=self._tool_budget.ensure_turn_tool_budget,
        )

    async def _run_supervisor_loop_async(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
    ) -> None:
        await self._supervisor_orchestrator.run_async(
            state=state,
            requested_workflow_id=requested_workflow_id,
            step_callback=step_callback,
            supervisor_spec=self._resolve_supervisor_spec(),
            run_workflow_async=self._run_workflow_async,
            finalize_with_supervisor_async=lambda s, wf: self._supervisor_decision.finalize_with_supervisor_async(
                state=s,
                requested_workflow_id=wf,
            ),
            run_chitchat_async=self._run_chitchat_async,
            decide_next_action_async=lambda s, spec, wf: self._supervisor_decision.decide_next_action_async(
                state=s,
                supervisor_spec=spec,
                requested_workflow_id=wf,
            ),
            resolve_workflow_target=self._supervisor_decision.resolve_workflow_target,
            resolve_subagent_target=self._supervisor_decision.resolve_subagent_target,
            append_execution_trace=self._supervisor_payload.append_execution_trace,
            execute_workflow_isolated_async=self._isolation_coordinator.execute_workflow_isolated_async,
            execute_subagent_isolated_async=self._isolation_coordinator.execute_subagent_isolated_async,
            ensure_turn_tool_budget=self._tool_budget.ensure_turn_tool_budget,
            emit_step_callback_async=self._emit_step_callback_async,
        )

    def _run_workflow(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        self._workflow_executor.run_sync(
            state=state,
            workflow_id=workflow_id,
            step_callback=step_callback,
            execute_node=self._isolation_coordinator.execute_workflow_agent_isolated_for_executor,
            best_available_final_text=self._result_service.best_available_final_text,
        )

    async def _run_workflow_async(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        await self._workflow_executor.run_async(
            state=state,
            workflow_id=workflow_id,
            step_callback=step_callback,
            execute_node_async=self._isolation_coordinator.execute_workflow_agent_isolated_async_for_executor,
            emit_step_callback_async=self._emit_step_callback_async,
            best_available_final_text=self._result_service.best_available_final_text,
        )

    # Backward-compatible seam used by tests; logic lives in RuntimeResultService.
    def _apply_agent_output(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        text: str,
        parsed: Optional[Dict[str, Any]],
    ) -> None:
        self._result_service.apply_agent_output(
            state=state,
            node_name=node_name,
            agent_id=agent_id,
            text=text,
            parsed=parsed,
        )

    def _resolve_llm(self, spec: AgentSpec) -> BaseLanguageModel:
        return self._llm_provider.resolve_llm(spec)

    def _resolve_supervisor_spec(self) -> Optional[AgentSpec]:
        preferred = os.getenv(_SUPERVISOR_AGENT_ENV, _DEFAULT_SUPERVISOR_AGENT_ID)
        system_agents = getattr(self.registry, "system_agents", None)
        if isinstance(system_agents, dict):
            if preferred in system_agents:
                return system_agents[preferred]
            if _DEFAULT_SUPERVISOR_AGENT_ID in system_agents:
                return system_agents[_DEFAULT_SUPERVISOR_AGENT_ID]
        if preferred in self.registry.agents:
            return self.registry.agents[preferred]
        if "supervisor" in self.registry.agents:
            return self.registry.agents["supervisor"]
        return None

    def _resolve_default_llm(self) -> BaseLanguageModel:
        return self._llm_provider.resolve_default_llm(self._resolve_supervisor_spec)

    def resolve_default_llm(self) -> BaseLanguageModel:
        """Public accessor for the runtime default model (used by memory pipeline)."""
        self._ensure_registry_loaded()
        return self._resolve_default_llm()

    def health_probe(self) -> Dict[str, Any]:
        self._ensure_registry_loaded()
        supervisor = self._resolve_supervisor_spec()
        errors: list[str] = []

        default_llm_type: Optional[str] = None
        try:
            default_llm = self._resolve_default_llm()
            default_llm_type = default_llm.__class__.__name__
        except Exception as exc:
            errors.append(f"default_llm_error: {exc}")

        tool = get_tool("scholar_search")
        if tool is None:
            errors.append("tool_error: scholar_search unavailable")

        return {
            "ok": len(errors) == 0,
            "loaded_agents": len(self.registry.agents),
            "loaded_subagents": len(getattr(self.registry, "subagents", {})),
            "loaded_system_agents": len(getattr(self.registry, "system_agents", {})),
            "loaded_workflows": len(self.registry.workflows),
            "supervisor_agent_id": supervisor.id if supervisor else None,
            "default_llm_type": default_llm_type,
            "tool_manager_initialized": tool is not None,
            "llm_cache": self._cache_metrics(),
            "errors": errors,
        }

    def _resolve_tool(self, tool_id: str):
        tool = get_tool(tool_id)
        if tool is None:
            raise ValueError(f"Tool unavailable: {tool_id}")
        return tool

    async def _emit_step_callback_async(
        self,
        step_callback: Optional[StepCallback],
        payload: Dict[str, Any],
    ) -> None:
        if step_callback is None:
            return
        result = step_callback(payload)
        if inspect.isawaitable(result):
            await result

    async def _invoke_async(
        self,
        runnable: Any,
        payload: Any,
        config: Optional[dict[str, Any]] = None,
    ) -> Any:
        ainvoke = getattr(runnable, "ainvoke", None)
        if callable(ainvoke):
            if config is not None:
                try:
                    return await ainvoke(payload, config=config)
                except TypeError:
                    pass
            return await ainvoke(payload)
        invoke = getattr(runnable, "invoke", None)
        if callable(invoke):
            if config is not None:
                try:
                    return await asyncio.to_thread(invoke, payload, config)
                except TypeError:
                    pass
            return await asyncio.to_thread(invoke, payload)
        raise TypeError(f"Runnable {type(runnable).__name__} has neither ainvoke nor invoke")

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        return self._codec.try_parse_json(text)

    def _cache_metrics(self) -> Dict[str, int]:
        return self._llm_provider.cache_metrics()
