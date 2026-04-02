from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from time import perf_counter
from typing import Any, Callable, Dict, Literal, Optional, TypedDict
from urllib.parse import urlparse

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.application.runtime.agent_factory import build_agent_from_spec
from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.context_facility import ContextFacility
from src.application.runtime.env_utils import read_env_float, read_env_int
from src.application.runtime.io_models import (
    AgentTaskInput,
    AgentTaskOutput,
    SupervisorDecision,
    WorkflowRunnerInput,
    WorkflowRunnerOutput,
)
from src.application.runtime.state_types import RuntimeState
from src.application.runtime.spec_models import AgentSpec
from src.application.runtime.workflow_router import WorkflowRuntime
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
_SUPERVISOR_ACTION_MAP = {
    "run_subagent": "run_agent",
    "start_workflow": "run_workflow",
}


class ExecutionResult(TypedDict):
    source_kind: Literal["subagent", "workflow"]
    source_id: str
    output_text: str
    parsed: Optional[Dict[str, Any]]
    tool_outputs: list[Any]
    artifacts_patch: Dict[str, Any]


class RuntimeEngine:
    """Single-path runtime: supervisor -> config workflow (if available) -> final output."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self.registry = registry
        self._llm_cache: OrderedDict[
            tuple[str, str, str, str, str, str], BaseLanguageModel
        ] = OrderedDict()
        self._llm_cache_max_size = max(1, read_env_int("LLM_CACHE_MAX_SIZE", 128, minimum=1))
        self._context_facility = ContextFacility.from_env()
        self._llm_cache_lock = threading.Lock()
        self._llm_cache_hits = 0
        self._llm_cache_misses = 0

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
        return self._build_result(state)

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
        return self._build_result(state)

    def _ensure_registry_loaded(self) -> None:
        if not self.registry.agents and not self.registry.workflows:
            self.registry.reload()

    def _run_chitchat(self, state: RuntimeState) -> None:
        llm = self._resolve_default_llm()
        messages = list(state["context"].get("messages", []))
        response = llm.invoke([SystemMessage(content=CHITCHAT_SYSTEM)] + messages)
        text = self._coerce_text(response)
        state["io"]["last_model_output"] = text
        state["output"]["final_text"] = text
        state["context"]["messages"].append(AIMessage(content=text))

    async def _run_chitchat_async(self, state: RuntimeState) -> None:
        llm = self._resolve_default_llm()
        messages = list(state["context"].get("messages", []))
        response = await self._invoke_async(llm, [SystemMessage(content=CHITCHAT_SYSTEM)] + messages)
        text = self._coerce_text(response)
        state["io"]["last_model_output"] = text
        state["output"]["final_text"] = text
        state["context"]["messages"].append(AIMessage(content=text))

    def _run_supervisor_loop(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
    ) -> None:
        if requested_workflow_id:
            if requested_workflow_id not in self.registry.workflows:
                raise ValueError(f"Unknown workflow_id: {requested_workflow_id}")
            self._run_workflow(state, requested_workflow_id, step_callback)
            self._finalize_with_supervisor(state, requested_workflow_id=requested_workflow_id)
            return

        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None:
            self._run_chitchat(state)
            return
        if supervisor_spec.mode != "chain":
            self._run_chitchat(state)
            return

        max_steps = max(1, read_env_int("SUPERVISOR_MAX_STEPS", 8, minimum=1))
        max_subagent_calls_per_agent = read_env_int(_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV, 2, minimum=0)
        max_subagent_calls_per_agent = max(0, max_subagent_calls_per_agent)
        max_wall_seconds = read_env_float("SUPERVISOR_MAX_WALL_TIME_SECONDS", 180.0)
        started = perf_counter()
        subagent_call_counts: dict[str, int] = {}

        for _ in range(max_steps):
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Supervisor loop timeout after {max_wall_seconds:.1f} seconds")
            decision = self._decide_next_action(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
            )
            action = decision.get("action")
            if not isinstance(action, str):
                action = "direct_reply"
            action = action.strip().lower()
            logger.info(
                "supervisor.decision action=%s target=%s done=%s step_count=%s",
                action,
                decision.get("target"),
                bool(decision.get("done")),
                state["runtime"].get("step_count", 0),
            )

            if action == "run_workflow":
                workflow_id = self._resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self.registry.workflows:
                    self._append_execution_trace(
                        state,
                        action=action,
                        target=workflow_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(decision.get("instruction") or ""),
                    )
                    self._execute_workflow_isolated(state, workflow_id, step_callback)
                    self._finalize_with_supervisor(state, requested_workflow_id=workflow_id)
                    return
                # Invalid target: degrade gracefully.
                action = "direct_reply"

            if action == "run_agent":
                agent_id = self._resolve_subagent_target(decision)
                if agent_id and agent_id in self.registry.agents and agent_id != supervisor_spec.id:
                    calls_used = subagent_call_counts.get(agent_id, 0)
                    if calls_used >= max_subagent_calls_per_agent:
                        logger.warning(
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
                        if decision.get("done"):
                            final_text = decision.get("final_text")
                            if not isinstance(final_text, str) or not final_text.strip():
                                final_text = decision.get("message")
                            if isinstance(final_text, str) and final_text.strip():
                                state["output"]["final_text"] = final_text
                                state["io"]["last_model_output"] = final_text
                                state["context"]["messages"].append(AIMessage(content=final_text))
                                return
                        continue

                    instruction = decision.get("instruction")
                    if isinstance(instruction, str) and instruction.strip():
                        state["artifacts"]["supervisor_instruction"] = instruction
                        state["context"]["messages"].append(
                            HumanMessage(content=f"Supervisor task for {agent_id}: {instruction}")
                        )
                    input_artifact_keys = decision.get("input_artifact_keys")
                    self._execute_subagent_isolated(
                        state,
                        agent_id,
                        instruction,
                        input_artifact_keys if isinstance(input_artifact_keys, list) else None,
                    )
                    self._append_execution_trace(
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
                    if decision.get("done") and state["output"].get("final_text"):
                        return
                    continue
                action = "direct_reply"

            if action == "direct_reply":
                final_text = decision.get("final_text")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = decision.get("message")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = state["io"].get("last_model_output") or "No output produced."
                self._append_execution_trace(
                    state,
                    action=action,
                    target=None,
                    reason=str(decision.get("reason") or ""),
                    instruction="",
                )
                state["output"]["final_text"] = final_text
                state["io"]["last_model_output"] = final_text
                state["context"]["messages"].append(AIMessage(content=final_text))
                return

        # Max supervisor steps reached: return the best available output.
        if not state["output"].get("final_text"):
            fallback = state["io"].get("last_model_output") or "No output produced."
            state["output"]["final_text"] = fallback

    async def _run_supervisor_loop_async(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
        step_callback: Optional[StepCallback],
    ) -> None:
        if requested_workflow_id:
            if requested_workflow_id not in self.registry.workflows:
                raise ValueError(f"Unknown workflow_id: {requested_workflow_id}")
            await self._run_workflow_async(state, requested_workflow_id, step_callback)
            await self._finalize_with_supervisor_async(state, requested_workflow_id=requested_workflow_id)
            return

        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None:
            await self._run_chitchat_async(state)
            return
        if supervisor_spec.mode != "chain":
            await self._run_chitchat_async(state)
            return

        max_steps = max(1, read_env_int("SUPERVISOR_MAX_STEPS", 8, minimum=1))
        max_subagent_calls_per_agent = read_env_int(_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV, 2, minimum=0)
        max_subagent_calls_per_agent = max(0, max_subagent_calls_per_agent)
        max_wall_seconds = read_env_float("SUPERVISOR_MAX_WALL_TIME_SECONDS", 180.0)
        started = perf_counter()
        subagent_call_counts: dict[str, int] = {}

        for _ in range(max_steps):
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Supervisor loop timeout after {max_wall_seconds:.1f} seconds")
            decision = await self._decide_next_action_async(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
            )
            action = decision.get("action")
            if not isinstance(action, str):
                action = "direct_reply"
            action = action.strip().lower()
            logger.info(
                "supervisor.decision action=%s target=%s done=%s step_count=%s",
                action,
                decision.get("target"),
                bool(decision.get("done")),
                state["runtime"].get("step_count", 0),
            )

            if action == "run_workflow":
                workflow_id = self._resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self.registry.workflows:
                    self._append_execution_trace(
                        state,
                        action=action,
                        target=workflow_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(decision.get("instruction") or ""),
                    )
                    await self._execute_workflow_isolated_async(state, workflow_id, step_callback)
                    await self._finalize_with_supervisor_async(state, requested_workflow_id=workflow_id)
                    return
                action = "direct_reply"

            if action == "run_agent":
                agent_id = self._resolve_subagent_target(decision)
                if agent_id and agent_id in self.registry.agents and agent_id != supervisor_spec.id:
                    calls_used = subagent_call_counts.get(agent_id, 0)
                    if calls_used >= max_subagent_calls_per_agent:
                        logger.warning(
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
                        if decision.get("done"):
                            final_text = decision.get("final_text")
                            if not isinstance(final_text, str) or not final_text.strip():
                                final_text = decision.get("message")
                            if isinstance(final_text, str) and final_text.strip():
                                state["output"]["final_text"] = final_text
                                state["io"]["last_model_output"] = final_text
                                state["context"]["messages"].append(AIMessage(content=final_text))
                                return
                        continue

                    instruction = decision.get("instruction")
                    if isinstance(instruction, str) and instruction.strip():
                        state["artifacts"]["supervisor_instruction"] = instruction
                        state["context"]["messages"].append(
                            HumanMessage(content=f"Supervisor task for {agent_id}: {instruction}")
                        )
                    input_artifact_keys = decision.get("input_artifact_keys")
                    await self._execute_subagent_isolated_async(
                        state,
                        agent_id,
                        instruction,
                        input_artifact_keys if isinstance(input_artifact_keys, list) else None,
                    )
                    self._append_execution_trace(
                        state,
                        action=action,
                        target=agent_id,
                        reason=str(decision.get("reason") or ""),
                        instruction=str(instruction or ""),
                    )
                    subagent_call_counts[agent_id] = calls_used + 1
                    state["runtime"]["step_count"] += 1
                    await self._emit_step_callback_async(
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
                    if decision.get("done") and state["output"].get("final_text"):
                        return
                    continue
                action = "direct_reply"

            if action == "direct_reply":
                final_text = decision.get("final_text")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = decision.get("message")
                if not isinstance(final_text, str) or not final_text.strip():
                    final_text = state["io"].get("last_model_output") or "No output produced."
                self._append_execution_trace(
                    state,
                    action=action,
                    target=None,
                    reason=str(decision.get("reason") or ""),
                    instruction="",
                )
                state["output"]["final_text"] = final_text
                state["io"]["last_model_output"] = final_text
                state["context"]["messages"].append(AIMessage(content=final_text))
                return

        if not state["output"].get("final_text"):
            fallback = state["io"].get("last_model_output") or "No output produced."
            state["output"]["final_text"] = fallback

    def _decide_next_action(
        self,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
    ) -> Dict[str, Any]:
        llm = self._resolve_llm(supervisor_spec)
        runnable = build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = runnable.invoke(
            self._build_supervisor_payload(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
                workflow_completed=False,
            )
        )
        text = self._coerce_text(raw)
        if not text.strip():
            logger.warning(
                "supervisor.empty_decision_output agent_id=%s step_count=%s workflow_id=%s",
                supervisor_spec.id,
                state["runtime"].get("step_count", 0),
                requested_workflow_id,
            )
        parsed = self._try_parse_supervisor_decision_json(text)
        state["io"]["last_model_output"] = text
        if isinstance(parsed, dict):
            return self._normalize_supervisor_decision(parsed, text, state)
        return {"action": "direct_reply", "final_text": text}

    async def _decide_next_action_async(
        self,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
    ) -> Dict[str, Any]:
        llm = self._resolve_llm(supervisor_spec)
        runnable = build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = await self._invoke_async(
            runnable,
            self._build_supervisor_payload(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
                workflow_completed=False,
            ),
        )
        text = self._coerce_text(raw)
        if not text.strip():
            logger.warning(
                "supervisor.empty_decision_output agent_id=%s step_count=%s workflow_id=%s",
                supervisor_spec.id,
                state["runtime"].get("step_count", 0),
                requested_workflow_id,
            )
        parsed = self._try_parse_supervisor_decision_json(text)
        state["io"]["last_model_output"] = text
        if isinstance(parsed, dict):
            return self._normalize_supervisor_decision(parsed, text, state)
        return {"action": "direct_reply", "final_text": text}

    def _normalize_supervisor_decision(
        self,
        parsed: Dict[str, Any],
        raw_text: str,
        state: RuntimeState,
    ) -> Dict[str, Any]:
        decision_model = self._coerce_supervisor_decision_model(parsed, raw_text)
        final_text = decision_model.final_text or raw_text
        if decision_model.done and not final_text.strip():
            final_text = state["io"].get("last_model_output") or "No output produced."

        normalized: Dict[str, Any] = {
            "action": decision_model.action,
            "target": decision_model.target,
            "final_text": final_text,
            "done": decision_model.done,
            "reason": decision_model.reason,
            "input_artifact_keys": list(decision_model.input_artifact_keys),
        }
        if isinstance(decision_model.instruction, str) and decision_model.instruction.strip():
            normalized["instruction"] = decision_model.instruction
        return normalized

    def _coerce_supervisor_decision_model(
        self,
        parsed: Dict[str, Any],
        raw_text: str,
    ) -> SupervisorDecision:
        raw_payload = dict(parsed or {})
        action = raw_payload.get("action")
        if not isinstance(action, str):
            action = "direct_reply"
        action = action.strip().lower()
        action = _SUPERVISOR_ACTION_MAP.get(action, action)
        if action not in {"direct_reply", "run_agent", "run_workflow"}:
            action = "direct_reply"
        target = raw_payload.get("target")
        if not isinstance(target, str):
            if action == "run_workflow":
                target = raw_payload.get("workflow_id")
            elif action == "run_agent":
                target = raw_payload.get("agent_id")
        final_text = raw_payload.get("final_text")
        if not isinstance(final_text, str):
            final_text = raw_payload.get("message")
        if not isinstance(final_text, str):
            final_text = raw_text

        cleaned_payload: Dict[str, Any] = {
            "action": action,
            "target": target if isinstance(target, str) and target.strip() else None,
            "instruction": raw_payload.get("instruction") if isinstance(raw_payload.get("instruction"), str) else None,
            "input_artifact_keys": raw_payload.get("input_artifact_keys", []),
            "done": bool(raw_payload.get("done", False)),
            "final_text": final_text,
            "reason": raw_payload.get("reason") if isinstance(raw_payload.get("reason"), str) else "",
        }

        try:
            return SupervisorDecision.model_validate(cleaned_payload)
        except Exception:
            return SupervisorDecision(
                action="direct_reply",
                final_text=raw_text,
                done=False,
                reason="invalid supervisor decision payload, degraded to direct_reply",
            )

    def _resolve_workflow_target(self, decision: Dict[str, Any], state: RuntimeState) -> Optional[str]:
        del state
        target = decision.get("target")
        if isinstance(target, str) and target in self.registry.workflows:
            return target

        workflow_hint = decision.get("workflow_id")
        if isinstance(workflow_hint, str) and workflow_hint in self.registry.workflows:
            return workflow_hint

        return None

    def _resolve_subagent_target(self, decision: Dict[str, Any]) -> Optional[str]:
        target = decision.get("target")
        if isinstance(target, str):
            return target
        agent_hint = decision.get("agent_id")
        if isinstance(agent_hint, str):
            return agent_hint
        return None

    def _finalize_with_supervisor(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
    ) -> None:
        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None:
            return
        if supervisor_spec.mode != "chain":
            return

        llm = self._resolve_llm(supervisor_spec)
        runnable = build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = runnable.invoke(
            self._build_supervisor_payload(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
                workflow_completed=True,
            )
        )
        text = self._coerce_text(raw)
        parsed = self._try_parse_supervisor_decision_json(text)
        decision = self._normalize_supervisor_decision(parsed, text, state) if isinstance(parsed, dict) else {
            "action": "direct_reply",
            "final_text": text,
        }
        if decision.get("action") != "direct_reply":
            return
        final_text = decision.get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            state["output"]["final_text"] = final_text
            state["io"]["last_model_output"] = final_text
            state["context"]["messages"].append(AIMessage(content=final_text))

    async def _finalize_with_supervisor_async(
        self,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
    ) -> None:
        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None:
            return
        if supervisor_spec.mode != "chain":
            return

        llm = self._resolve_llm(supervisor_spec)
        runnable = build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = await self._invoke_async(
            runnable,
            self._build_supervisor_payload(
                state=state,
                supervisor_spec=supervisor_spec,
                requested_workflow_id=requested_workflow_id,
                workflow_completed=True,
            ),
        )
        text = self._coerce_text(raw)
        parsed = self._try_parse_supervisor_decision_json(text)
        decision = self._normalize_supervisor_decision(parsed, text, state) if isinstance(parsed, dict) else {
            "action": "direct_reply",
            "final_text": text,
        }
        if decision.get("action") != "direct_reply":
            return
        final_text = decision.get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            state["output"]["final_text"] = final_text
            state["io"]["last_model_output"] = final_text
            state["context"]["messages"].append(AIMessage(content=final_text))

    def _run_workflow(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        state["runtime"]["mode"] = "workflow"
        state["runtime"]["workflow_id"] = workflow_id
        spec = self.registry.workflows[workflow_id]
        runtime = WorkflowRuntime(spec, agent_runner=None)

        current_node = spec.entry_node
        visit_counts: dict[str, int] = {current_node: 1}
        max_wall_seconds = read_env_float("WORKFLOW_MAX_WALL_TIME_SECONDS", 300.0)
        started = perf_counter()

        while True:
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Workflow timeout after {max_wall_seconds:.1f} seconds")
            runtime.enforce_limits(
                {
                    "_step_count": state["runtime"]["step_count"],
                    "_loop_count": state["runtime"]["loop_count"],
                }
            )
            state["runtime"]["current_node"] = current_node
            logger.info(
                "workflow.step workflow_id=%s current_node=%s step_count=%s loop_count=%s",
                workflow_id,
                current_node,
                state["runtime"].get("step_count", 0),
                state["runtime"].get("loop_count", 0),
            )

            node_spec = spec.nodes.get(current_node)
            if not node_spec:
                raise RuntimeError(f"Node not found in workflow: {current_node}")

            node_type = node_spec.get("type")
            if node_type == "terminal":
                fallback = self._best_available_final_text(state)
                if fallback:
                    state["output"]["final_text"] = fallback
                break

            if node_type != "agent":
                raise RuntimeError(f"Unsupported node type: {node_type}")

            agent_id = node_spec.get("agent_id")
            if not isinstance(agent_id, str):
                raise RuntimeError(f"Invalid agent_id for node: {current_node}")

            self._execute_workflow_agent_isolated(state, current_node, agent_id)
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

    async def _run_workflow_async(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        state["runtime"]["mode"] = "workflow"
        state["runtime"]["workflow_id"] = workflow_id
        spec = self.registry.workflows[workflow_id]
        runtime = WorkflowRuntime(spec, agent_runner=None)

        current_node = spec.entry_node
        visit_counts: dict[str, int] = {current_node: 1}
        max_wall_seconds = read_env_float("WORKFLOW_MAX_WALL_TIME_SECONDS", 300.0)
        started = perf_counter()

        while True:
            if perf_counter() - started > max_wall_seconds:
                raise TimeoutError(f"Workflow timeout after {max_wall_seconds:.1f} seconds")
            runtime.enforce_limits(
                {
                    "_step_count": state["runtime"]["step_count"],
                    "_loop_count": state["runtime"]["loop_count"],
                }
            )
            state["runtime"]["current_node"] = current_node
            logger.info(
                "workflow.step workflow_id=%s current_node=%s step_count=%s loop_count=%s",
                workflow_id,
                current_node,
                state["runtime"].get("step_count", 0),
                state["runtime"].get("loop_count", 0),
            )

            node_spec = spec.nodes.get(current_node)
            if not node_spec:
                raise RuntimeError(f"Node not found in workflow: {current_node}")

            node_type = node_spec.get("type")
            if node_type == "terminal":
                fallback = self._best_available_final_text(state)
                if fallback:
                    state["output"]["final_text"] = fallback
                break

            if node_type != "agent":
                raise RuntimeError(f"Unsupported node type: {node_type}")

            agent_id = node_spec.get("agent_id")
            if not isinstance(agent_id, str):
                raise RuntimeError(f"Invalid agent_id for node: {current_node}")

            await self._execute_workflow_agent_isolated_async(state, current_node, agent_id)
            state["runtime"]["step_count"] += 1

            next_node = runtime.next_node(current_node, state)
            runtime.assert_transition_allowed(current_node, next_node)
            state["runtime"]["current_node"] = next_node

            await self._emit_step_callback_async(
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

    def _execute_workflow_agent_isolated(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
    ) -> None:
        instruction = self._build_workflow_node_instruction(state, node_name, agent_id)
        isolated_state = self._build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=None,
        )
        self._execute_agent(isolated_state, node_name, agent_id)
        result = self._collect_subagent_execution_result(isolated_state, agent_id)
        self._deliver_execution_result_to_supervisor(state, result)

    async def _execute_workflow_agent_isolated_async(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
    ) -> None:
        instruction = self._build_workflow_node_instruction(state, node_name, agent_id)
        isolated_state = self._build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=None,
        )
        await self._execute_agent_async(isolated_state, node_name, agent_id)
        result = self._collect_subagent_execution_result(isolated_state, agent_id)
        self._deliver_execution_result_to_supervisor(state, result)

    def _build_workflow_node_instruction(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
    ) -> str:
        user_text = state.get("input", {}).get("user_text", "")
        artifacts = self._select_input_artifacts(state, input_artifact_keys=None)
        artifact_text = json.dumps(artifacts, ensure_ascii=False, default=str)
        return (
            f"[workflow_node={node_name} agent={agent_id}] "
            f"User request: {user_text}\n"
            f"Current artifacts: {artifact_text}"
        )

    def _execute_agent(self, state: RuntimeState, node_name: str, agent_id: str) -> None:
        if agent_id not in self.registry.agents:
            raise RuntimeError(f"Agent not found: {agent_id}")

        spec = self.registry.agents[agent_id]
        llm = self._resolve_llm(spec)
        runnable = build_agent_from_spec(spec, llm, self._resolve_tool)

        if spec.mode == "react":
            self._execute_react_agent(state, node_name, agent_id, runnable)
            return

        payload = self._build_chain_payload(state, node_name, agent_id)
        raw = runnable.invoke(payload)
        text = self._coerce_text(raw)
        parsed = self._normalize_agent_parsed_payload(text, self._try_parse_json(text))

        state["io"]["last_model_output"] = text
        state["io"]["last_tool_outputs"] = []
        state["context"]["messages"].append(AIMessage(content=text))

        self._apply_agent_output(state, node_name, agent_id, text, parsed)

    async def _execute_agent_async(self, state: RuntimeState, node_name: str, agent_id: str) -> None:
        if agent_id not in self.registry.agents:
            raise RuntimeError(f"Agent not found: {agent_id}")

        spec = self.registry.agents[agent_id]
        llm = self._resolve_llm(spec)
        runnable = build_agent_from_spec(spec, llm, self._resolve_tool)

        if spec.mode == "react":
            await self._execute_react_agent_async(state, node_name, agent_id, runnable)
            return

        payload = self._build_chain_payload(state, node_name, agent_id)
        raw = await self._invoke_async(runnable, payload)
        text = self._coerce_text(raw)
        parsed = self._normalize_agent_parsed_payload(text, self._try_parse_json(text))

        state["io"]["last_model_output"] = text
        state["io"]["last_tool_outputs"] = []
        state["context"]["messages"].append(AIMessage(content=text))

        self._apply_agent_output(state, node_name, agent_id, text, parsed)

    def _execute_react_agent(self, state: RuntimeState, node_name: str, agent_id: str, runnable: Any) -> None:
        messages = list(state["context"].get("messages", []))
        raw = runnable.invoke({"messages": messages})

        next_messages = raw.get("messages") if isinstance(raw, dict) else None
        if isinstance(next_messages, list) and next_messages:
            state["context"]["messages"] = next_messages
            ai_text = self._extract_last_ai_text(next_messages)
            tool_outputs = self._extract_tool_outputs(next_messages)
        else:
            ai_text = self._coerce_text(raw)
            tool_outputs = []
            state["context"]["messages"].append(AIMessage(content=ai_text))

        parsed = self._normalize_agent_parsed_payload(ai_text, self._try_parse_json(ai_text))
        state["io"]["last_model_output"] = ai_text
        state["io"]["last_tool_outputs"] = tool_outputs

        self._apply_agent_output(state, node_name, agent_id, ai_text, parsed)

    async def _execute_react_agent_async(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        runnable: Any,
    ) -> None:
        messages = list(state["context"].get("messages", []))
        raw = await self._invoke_async(runnable, {"messages": messages})

        next_messages = raw.get("messages") if isinstance(raw, dict) else None
        if isinstance(next_messages, list) and next_messages:
            state["context"]["messages"] = next_messages
            ai_text = self._extract_last_ai_text(next_messages)
            tool_outputs = self._extract_tool_outputs(next_messages)
        else:
            ai_text = self._coerce_text(raw)
            tool_outputs = []
            state["context"]["messages"].append(AIMessage(content=ai_text))

        parsed = self._normalize_agent_parsed_payload(ai_text, self._try_parse_json(ai_text))
        state["io"]["last_model_output"] = ai_text
        state["io"]["last_tool_outputs"] = tool_outputs

        self._apply_agent_output(state, node_name, agent_id, ai_text, parsed)

    def _execute_subagent_isolated(
        self,
        state: RuntimeState,
        agent_id: str,
        instruction: Any,
        input_artifact_keys: Optional[list[str]] = None,
    ) -> None:
        isolated_state = self._build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=input_artifact_keys,
        )
        self._execute_agent(isolated_state, agent_id, agent_id)
        result = self._collect_subagent_execution_result(isolated_state, agent_id)
        self._deliver_execution_result_to_supervisor(state, result)

    async def _execute_subagent_isolated_async(
        self,
        state: RuntimeState,
        agent_id: str,
        instruction: Any,
        input_artifact_keys: Optional[list[str]] = None,
    ) -> None:
        isolated_state = self._build_isolated_subagent_state(
            state,
            agent_id,
            instruction,
            input_artifact_keys=input_artifact_keys,
        )
        await self._execute_agent_async(isolated_state, agent_id, agent_id)
        result = self._collect_subagent_execution_result(isolated_state, agent_id)
        self._deliver_execution_result_to_supervisor(state, result)

    def _execute_workflow_isolated(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        runner_input = WorkflowRunnerInput(
            workflow_id=workflow_id,
            instruction=state["input"].get("user_text", ""),
            seed_artifacts=self._select_input_artifacts(state, input_artifact_keys=None),
            limits={
                "max_steps": self.registry.workflows[workflow_id].limits.get("max_steps", 20),
                "max_loops": self.registry.workflows[workflow_id].limits.get("max_loops", 4),
            },
        )
        isolated_state = self._build_isolated_workflow_state(state, workflow_id, runner_input=runner_input)
        self._run_workflow(isolated_state, workflow_id, step_callback)
        state["runtime"]["step_count"] += int(isolated_state.get("runtime", {}).get("step_count", 0))
        state["runtime"]["loop_count"] += int(isolated_state.get("runtime", {}).get("loop_count", 0))
        result = self._collect_workflow_execution_result(isolated_state, workflow_id)
        self._deliver_execution_result_to_supervisor(state, result)

    async def _execute_workflow_isolated_async(
        self,
        state: RuntimeState,
        workflow_id: str,
        step_callback: Optional[StepCallback],
    ) -> None:
        runner_input = WorkflowRunnerInput(
            workflow_id=workflow_id,
            instruction=state["input"].get("user_text", ""),
            seed_artifacts=self._select_input_artifacts(state, input_artifact_keys=None),
            limits={
                "max_steps": self.registry.workflows[workflow_id].limits.get("max_steps", 20),
                "max_loops": self.registry.workflows[workflow_id].limits.get("max_loops", 4),
            },
        )
        isolated_state = self._build_isolated_workflow_state(state, workflow_id, runner_input=runner_input)
        await self._run_workflow_async(isolated_state, workflow_id, step_callback)
        state["runtime"]["step_count"] += int(isolated_state.get("runtime", {}).get("step_count", 0))
        state["runtime"]["loop_count"] += int(isolated_state.get("runtime", {}).get("loop_count", 0))
        result = self._collect_workflow_execution_result(isolated_state, workflow_id)
        self._deliver_execution_result_to_supervisor(state, result)

    def _build_isolated_subagent_state(
        self,
        parent_state: RuntimeState,
        agent_id: str,
        instruction: Any,
        input_artifact_keys: Optional[list[str]] = None,
    ) -> RuntimeState:
        task_text = ""
        if isinstance(instruction, str) and instruction.strip():
            task_text = instruction.strip()
        if not task_text:
            task_text = parent_state["input"].get("user_text", "")
        selected_input_artifacts = self._select_input_artifacts(parent_state, input_artifact_keys)
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
                "messages": [HumanMessage(content=task_text)],
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

    def _select_input_artifacts(
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

    def _build_isolated_workflow_state(
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
                seed_artifacts=self._select_input_artifacts(parent_state, input_artifact_keys=None),
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

    def _collect_subagent_execution_result(
        self,
        isolated_state: RuntimeState,
        agent_id: str,
    ) -> ExecutionResult:
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

    def _collect_workflow_execution_result(
        self,
        isolated_state: RuntimeState,
        workflow_id: str,
    ) -> ExecutionResult:
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

    def _deliver_execution_result_to_supervisor(
        self,
        supervisor_state: RuntimeState,
        result: ExecutionResult,
    ) -> None:
        text = result.get("output_text", "")
        supervisor_state["io"]["last_model_output"] = text
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
            result.get("source_id", ""),
            result.get("source_id", ""),
            text if isinstance(text, str) else "",
            parsed_payload,
        )

    def _apply_agent_output(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        text: str,
        parsed: Optional[Dict[str, Any]],
    ) -> None:
        artifacts_state = state.get("artifacts")
        if not isinstance(artifacts_state, dict):
            artifacts_state = {}
            state["artifacts"] = artifacts_state

        shared = artifacts_state.get("shared")
        if not isinstance(shared, dict):
            if shared is not None:
                logger.warning(
                    "runtime.artifacts.shared_reset_invalid_type agent_id=%s type=%s",
                    agent_id,
                    type(shared).__name__,
                )
            shared = {}
            artifacts_state["shared"] = shared

        shared.pop(agent_id, None)
        shared[agent_id] = {
            "node": node_name,
            "output_text": text,
            "parsed": parsed,
            "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
        }

        if parsed and isinstance(parsed, dict):
            artifacts_patch = parsed.get("artifacts")
            if isinstance(artifacts_patch, dict):
                patch = dict(artifacts_patch)
                shared_patch = patch.pop("shared", None) if "shared" in patch or "shared" in artifacts_patch else None
                artifacts_state.update(patch)
                if "shared" in artifacts_patch:
                    if isinstance(shared_patch, dict):
                        target_shared = artifacts_state.get("shared")
                        if not isinstance(target_shared, dict):
                            target_shared = {}
                            artifacts_state["shared"] = target_shared
                        target_shared.update(shared_patch)
                    else:
                        logger.warning(
                            "runtime.artifacts.shared_patch_ignored agent_id=%s type=%s",
                            agent_id,
                            type(shared_patch).__name__,
                        )

            final_text = parsed.get("final_text")
            if isinstance(final_text, str) and final_text.strip():
                state["output"]["final_text"] = final_text

            final_structured = parsed.get("final_structured")
            if isinstance(final_structured, dict):
                state["output"]["final_structured"] = final_structured

        if node_name == "reporter" and not state["output"].get("final_text"):
            state["output"]["final_text"] = text

    def _build_chain_payload(self, state: RuntimeState, node_name: str, agent_id: str) -> Dict[str, Any]:
        artifacts = state.get("artifacts", {})
        return {
            "user_text": state["input"].get("user_text", ""),
            "messages": self._context_facility.messages_to_text(
                state["context"].get("messages", []),
                scope="default",
            ),
            "memory_summary": state["context"].get("memory_summary", ""),
            "last_model_output": state["io"].get("last_model_output", ""),
            "last_tool_outputs": json.dumps(state["io"].get("last_tool_outputs", []), ensure_ascii=False),
            "artifacts": json.dumps(artifacts, ensure_ascii=False, default=str),
            "agent_id": agent_id,
            "node_name": node_name,
            "supervisor_instruction": artifacts.get("supervisor_instruction", ""),
        }

    def _resolve_llm(self, spec: AgentSpec) -> BaseLanguageModel:
        llm_ref = spec.llm
        profile_name = llm_ref.name.strip()
        profile = self.registry.llms.get(profile_name)
        if profile is None:
            raise RuntimeError(f"Unknown llm profile name: {profile_name}")

        model = profile.model_name
        base_url = profile.base_url or ""
        api_key_env = (profile.api_key_env or "").strip()
        if _ENV_PLACEHOLDER_PATTERN.search(model):
            raise RuntimeError(
                f"Unresolved env placeholder in model_name for llm profile '{profile_name}': {model}"
            )
        if base_url and _ENV_PLACEHOLDER_PATTERN.search(base_url):
            raise RuntimeError(
                f"Unresolved env placeholder in base_url for llm profile '{profile_name}': {base_url}"
            )
        if api_key_env and _ENV_PLACEHOLDER_PATTERN.search(api_key_env):
            raise RuntimeError(
                f"Unresolved env placeholder in api_key_env for llm profile '{profile_name}': {api_key_env}"
            )
        api_key = ""
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"Missing API key env '{api_key_env}' for llm profile '{profile_name}'"
                )
        temperature = (
            float(llm_ref.temperature)
            if llm_ref.temperature is not None
            else float(profile.temperature)
        )
        cache_temperature = f"{round(temperature, 3):.3f}"
        llm_timeout_seconds = read_env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0)
        cache_timeout = f"{llm_timeout_seconds:.3f}"
        user_agent = _resolve_openai_compat_user_agent(base_url)
        cache_user_agent = user_agent or ""

        key = (
            profile_name,
            model,
            base_url,
            cache_temperature,
            cache_timeout,
            cache_user_agent,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": llm_timeout_seconds,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        if user_agent:
            kwargs["default_headers"] = {"User-Agent": user_agent}

        with self._llm_cache_lock:
            cached = self._llm_cache.get(key)
            if cached is not None:
                self._llm_cache_hits += 1
                self._llm_cache.move_to_end(key)
                return cached

            self._llm_cache_misses += 1
            llm = ChatOpenAI(**kwargs)
            while len(self._llm_cache) >= self._llm_cache_max_size:
                self._llm_cache.popitem(last=False)
            self._llm_cache[key] = llm
            self._llm_cache.move_to_end(key)
            return llm

    def _available_agents(self, supervisor_spec: AgentSpec) -> list[str]:
        subagents = getattr(self.registry, "subagents", None)
        if isinstance(subagents, dict) and subagents:
            return sorted(
                [
                    agent_id
                    for agent_id in subagents.keys()
                    if agent_id != supervisor_spec.id and not agent_id.endswith("_router")
                ]
            )
        return sorted(
            [
                agent_id
                for agent_id in self.registry.agents.keys()
                if agent_id != supervisor_spec.id and not agent_id.endswith("_router")
            ]
        )

    def _available_workflows(self) -> list[str]:
        return sorted(self.registry.workflows.keys())

    def _agent_capabilities(self, supervisor_spec: AgentSpec) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for agent_id in self._available_agents(supervisor_spec):
            spec = self.registry.agents.get(agent_id)
            if spec is None:
                continue
            rows.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "mode": spec.mode,
                    "tools": list(spec.tools),
                }
            )
        return rows

    def _workflow_capabilities(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for workflow_id in self._available_workflows():
            spec = self.registry.workflows.get(workflow_id)
            if spec is None:
                continue
            rows.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "entry_node": spec.entry_node,
                    "node_count": len(spec.nodes),
                }
            )
        return rows

    def _append_execution_trace(
        self,
        state: RuntimeState,
        *,
        action: str,
        target: Optional[str],
        reason: str,
        instruction: str,
    ) -> None:
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            state["artifacts"] = artifacts

        last_model_output = state.get("io", {}).get("last_model_output")
        tool_outputs = state.get("io", {}).get("last_tool_outputs")
        tool_count = len(tool_outputs) if isinstance(tool_outputs, list) else 0
        self._context_facility.append_trace(
            artifacts,
            entry={
                "step_count": int(state.get("runtime", {}).get("step_count", 0)),
                "action": action,
                "target": target,
                "reason": (reason or "")[: self._context_facility.policy.trace_reason_chars],
                "instruction": (instruction or "")[: self._context_facility.policy.trace_instruction_chars],
                "tool_outputs_count": tool_count,
                "last_model_output_preview": (last_model_output or "")[
                    : self._context_facility.policy.trace_output_preview_chars
                ]
                if isinstance(last_model_output, str)
                else "",
            },
            trace_key="execution_trace",
        )

    def _build_supervisor_payload(
        self,
        *,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
        workflow_completed: bool,
    ) -> Dict[str, Any]:
        available_agents = self._available_agents(supervisor_spec)
        available_workflows = self._available_workflows()
        agent_capabilities = self._agent_capabilities(supervisor_spec)
        workflow_capabilities = self._workflow_capabilities()
        artifacts = state.get("artifacts", {})
        compact_artifacts = self._context_facility.compact_artifacts(
            artifacts if isinstance(artifacts, dict) else {},
            excluded_keys={"shared", "execution_trace", "task_input", "workflow_runner_input"},
        )
        recent_steps = self._context_facility.recent_trace(
            artifacts if isinstance(artifacts, dict) else {},
            trace_key="execution_trace",
        )
        return {
            "user_text": state["input"].get("user_text", ""),
            "messages": self._context_facility.messages_to_text(
                state["context"].get("messages", []),
                scope="supervisor",
            ),
            "memory_summary": state["context"].get("memory_summary", ""),
            "artifacts": json.dumps(artifacts, ensure_ascii=False, default=str),
            "artifacts_compact": json.dumps(compact_artifacts, ensure_ascii=False, default=str),
            "last_model_output": state["io"].get("last_model_output", ""),
            "last_tool_outputs": json.dumps(state["io"].get("last_tool_outputs", []), ensure_ascii=False, default=str),
            "available_agents": json.dumps(available_agents, ensure_ascii=False),
            "available_workflows": json.dumps(available_workflows, ensure_ascii=False),
            "agent_capabilities": json.dumps(agent_capabilities, ensure_ascii=False, default=str),
            "workflow_capabilities": json.dumps(workflow_capabilities, ensure_ascii=False, default=str),
            "recent_steps": json.dumps(recent_steps, ensure_ascii=False, default=str),
            "requested_workflow_id": requested_workflow_id or "",
            "step_count": state["runtime"].get("step_count", 0),
            "loop_count": state["runtime"].get("loop_count", 0),
            "workflow_completed": workflow_completed,
        }

    def _best_available_final_text(self, state: RuntimeState) -> Optional[str]:
        final_text = state["output"].get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            return final_text

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
        supervisor = self._resolve_supervisor_spec()
        if supervisor is not None:
            return self._resolve_llm(supervisor)
        if self.registry.agents:
            first_spec = next(iter(self.registry.agents.values()))
            return self._resolve_llm(first_spec)
        raise RuntimeError("No agent specs loaded; cannot resolve default llm")

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

        tool = get_tool("web_search")
        if tool is None:
            errors.append("tool_error: web_search unavailable")

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

    async def _invoke_async(self, runnable: Any, payload: Any) -> Any:
        ainvoke = getattr(runnable, "ainvoke", None)
        if callable(ainvoke):
            return await ainvoke(payload)
        invoke = getattr(runnable, "invoke", None)
        if callable(invoke):
            return await asyncio.to_thread(invoke, payload)
        raise TypeError(f"Runnable {type(runnable).__name__} has neither ainvoke nor invoke")

    def _extract_last_ai_text(self, messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return self._coerce_text(message)
        return ""

    def _extract_tool_outputs(self, messages: list[BaseMessage]) -> list[Any]:
        outputs: list[Any] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                outputs.append(message.content)
        return outputs

    def _coerce_text(self, raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, BaseMessage):
            content = raw.content
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        if hasattr(raw, "model_dump"):
            return json.dumps(raw.model_dump(), ensure_ascii=False, default=str)
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False, default=str)
        return str(raw)

    def _try_parse_supervisor_decision_json(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = _SUPERVISOR_DECISION_PARSER.parse(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return self._try_parse_json(raw)

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None
        decoder = json.JSONDecoder()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if fence_match:
            snippet = fence_match.group(1)
            try:
                parsed = json.loads(snippet)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text, idx)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue

        return None

    def _normalize_agent_parsed_payload(
        self,
        text: str,
        parsed: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = dict(parsed or {})
        status = payload.get("status")
        if status not in {"success", "needs_clarification", "failed"}:
            status = "success"
        final_text = payload.get("final_text")
        if not isinstance(final_text, str) or not final_text.strip():
            final_text = text
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        confidence = payload.get("confidence", 0.5)
        errors = payload.get("errors", [])
        try:
            model = AgentTaskOutput.model_validate(
                {
                    "status": status,
                    "final_text": final_text,
                    "artifacts": artifacts,
                    "confidence": confidence,
                    "errors": errors,
                }
            )
        except Exception:
            model = AgentTaskOutput(status="failed", final_text=text, artifacts={}, confidence=0.0, errors=[])
        merged = dict(payload)
        merged.update(model.model_dump())
        return merged

    def _cache_metrics(self) -> Dict[str, int]:
        with self._llm_cache_lock:
            return {
                "size": len(self._llm_cache),
                "max_size": self._llm_cache_max_size,
                "hits": self._llm_cache_hits,
                "misses": self._llm_cache_misses,
            }

    def _build_result(self, state: RuntimeState) -> Dict[str, Any]:
        final_structured = state["output"].get("final_structured")
        if isinstance(final_structured, dict):
            result = {"success": True, "type": "structured", "data": final_structured}
            if "message" in final_structured and isinstance(final_structured["message"], str):
                result["message"] = final_structured["message"]
            return result

        final_text = (
            state["output"].get("final_text")
            or self._best_available_final_text(state)
            or state["io"].get("last_model_output")
        )
        return {
            "success": bool(final_text),
            "type": "chat",
            "message": final_text or "No output produced.",
            "data": {
                "runtime": state.get("runtime", {}),
                "artifacts": state.get("artifacts", {}),
            },
        }
def _resolve_openai_compat_user_agent(base_url: str) -> str:
    normalized_base_url = (base_url or "").strip()
    if not normalized_base_url:
        return ""
    hostname = (urlparse(normalized_base_url).hostname or "").lower()
    if hostname == "api.openai.com":
        return ""
    return os.getenv("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/1.0").strip()
