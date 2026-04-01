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
from typing import Any, Callable, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.application.runtime.agent_factory import build_agent_from_spec
from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.env_utils import read_env_float
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


class RuntimeEngine:
    """Single-path runtime: supervisor -> config workflow (if available) -> final output."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self.registry = registry
        self._llm_cache: OrderedDict[tuple[str, str, str, str, str], BaseLanguageModel] = OrderedDict()
        self._llm_cache_max_size = max(1, _read_int_env("LLM_CACHE_MAX_SIZE", 128))
        self._chain_context_messages_window = max(1, _read_int_env("CHAIN_CONTEXT_MESSAGES_WINDOW", 12))
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

        max_steps = max(1, _read_int_env("SUPERVISOR_MAX_STEPS", 8))
        max_subagent_calls_per_agent = _read_int_env(_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV, 2)
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

            if action == "start_workflow":
                workflow_id = self._resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self.registry.workflows:
                    self._run_workflow(state, workflow_id, step_callback)
                    self._finalize_with_supervisor(state, requested_workflow_id=workflow_id)
                    return
                # Invalid target: degrade gracefully.
                action = "direct_reply"

            if action == "run_subagent":
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
                    self._execute_agent(state, agent_id, agent_id)
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

        max_steps = max(1, _read_int_env("SUPERVISOR_MAX_STEPS", 8))
        max_subagent_calls_per_agent = _read_int_env(_SUPERVISOR_MAX_SUBAGENT_CALLS_ENV, 2)
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

            if action == "start_workflow":
                workflow_id = self._resolve_workflow_target(decision, state)
                if workflow_id and workflow_id in self.registry.workflows:
                    await self._run_workflow_async(state, workflow_id, step_callback)
                    await self._finalize_with_supervisor_async(state, requested_workflow_id=workflow_id)
                    return
                action = "direct_reply"

            if action == "run_subagent":
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
                    await self._execute_agent_async(state, agent_id, agent_id)
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
        action = parsed.get("action")
        if not isinstance(action, str):
            action = "direct_reply"

        target = parsed.get("target")
        if not isinstance(target, str):
            target = parsed.get("workflow_id") if action == "start_workflow" else parsed.get("agent_id")
        if not isinstance(target, str):
            target = None

        final_text = parsed.get("final_text")
        if not isinstance(final_text, str):
            final_text = parsed.get("message")
        if not isinstance(final_text, str):
            final_text = raw_text

        done = bool(parsed.get("done", False))
        if done and not final_text.strip():
            final_text = state["io"].get("last_model_output") or "No output produced."

        normalized: Dict[str, Any] = {
            "action": action.strip().lower(),
            "target": target,
            "final_text": final_text,
            "done": done,
        }
        if isinstance(parsed.get("instruction"), str):
            normalized["instruction"] = parsed["instruction"]
        if isinstance(parsed.get("reason"), str):
            normalized["reason"] = parsed["reason"]
        return normalized

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

            self._execute_agent(state, current_node, agent_id)
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

            await self._execute_agent_async(state, current_node, agent_id)
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
        parsed = self._try_parse_json(text)

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
        parsed = self._try_parse_json(text)

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

        parsed = self._try_parse_json(ai_text)
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

        parsed = self._try_parse_json(ai_text)
        state["io"]["last_model_output"] = ai_text
        state["io"]["last_tool_outputs"] = tool_outputs

        self._apply_agent_output(state, node_name, agent_id, ai_text, parsed)

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
        payload = {
            "user_text": state["input"].get("user_text", ""),
            "messages": self._messages_to_text(state["context"].get("messages", [])),
            "memory_summary": state["context"].get("memory_summary", ""),
            "last_model_output": state["io"].get("last_model_output", ""),
            "last_tool_outputs": json.dumps(state["io"].get("last_tool_outputs", []), ensure_ascii=False),
            "artifacts": json.dumps(artifacts, ensure_ascii=False, default=str),
            "agent_id": agent_id,
            "node_name": node_name,
            # common shortcuts for current workflow prompts
            "initial_topic": artifacts.get("topic") or state["input"].get("user_text", ""),
            "retrieved_resources": json.dumps(artifacts.get("retrieved_resources", []), ensure_ascii=False, default=str),
            "feedback_section": artifacts.get("feedback_section", ""),
            "all_resources": json.dumps(artifacts.get("retrieved_resources", []), ensure_ascii=False, default=str),
            "research_gap": artifacts.get("research_gap", ""),
            "research_idea": artifacts.get("research_idea", ""),
            "supervisor_instruction": artifacts.get("supervisor_instruction", ""),
        }
        return payload

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

        key = (profile_name, model, base_url, cache_temperature, cache_timeout)
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": llm_timeout_seconds,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

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
        return sorted(
            [
                agent_id
                for agent_id in self.registry.agents.keys()
                if agent_id != supervisor_spec.id and not agent_id.endswith("_router")
            ]
        )

    def _available_workflows(self) -> list[str]:
        return sorted(self.registry.workflows.keys())

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
        return {
            "user_text": state["input"].get("user_text", ""),
            "messages": self._messages_to_text(state["context"].get("messages", [])),
            "artifacts": json.dumps(state.get("artifacts", {}), ensure_ascii=False, default=str),
            "last_model_output": state["io"].get("last_model_output", ""),
            "last_tool_outputs": json.dumps(state["io"].get("last_tool_outputs", []), ensure_ascii=False, default=str),
            "available_agents": json.dumps(available_agents, ensure_ascii=False),
            "available_workflows": json.dumps(available_workflows, ensure_ascii=False),
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

    def _messages_to_text(self, messages: list[BaseMessage]) -> str:
        lines: list[str] = []
        for message in messages[-self._chain_context_messages_window:]:
            role = message.__class__.__name__.replace("Message", "").lower() or "message"
            content = self._coerce_text(message)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

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


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default
