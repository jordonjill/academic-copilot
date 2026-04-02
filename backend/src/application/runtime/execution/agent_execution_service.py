from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, Optional

from langchain_core.messages import AIMessage

from src.application.runtime.contracts.spec_models import AgentSpec
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.providers.context_facility import ContextFacility


class AgentExecutionService:
    def __init__(
        self,
        *,
        registry: Any,
        tool_budget: Any,
        context_facility: ContextFacility,
        build_agent_from_spec_fn: Callable[[AgentSpec, Any, Callable[[str], Any]], Any],
        resolve_llm: Callable[[AgentSpec], Any],
        apply_agent_output: Callable[[RuntimeState, str, str, str, Optional[Dict[str, Any]]], None],
        coerce_text: Callable[[Any], str],
        try_parse_json: Callable[[str], Optional[Dict[str, Any]]],
        normalize_agent_parsed_payload: Callable[[str, Optional[Dict[str, Any]]], Optional[Dict[str, Any]]],
        invoke_async: Callable[[Any, Any, Optional[dict[str, Any]]], Awaitable[Any]],
        extract_last_ai_text: Callable[[list[Any]], str],
        extract_tool_outputs: Callable[[list[Any]], list[Any]],
        react_max_internal_steps_default: int,
        react_max_internal_steps_workflow: int,
        react_max_internal_steps_turn: int,
        react_max_internal_steps_budget_exhausted: int,
    ) -> None:
        self._registry = registry
        self._tool_budget = tool_budget
        self._context_facility = context_facility
        self._build_agent_from_spec = build_agent_from_spec_fn
        self._resolve_llm = resolve_llm
        self._apply_agent_output = apply_agent_output
        self._coerce_text = coerce_text
        self._try_parse_json = try_parse_json
        self._normalize_agent_parsed_payload = normalize_agent_parsed_payload
        self._invoke_async = invoke_async
        self._extract_last_ai_text = extract_last_ai_text
        self._extract_tool_outputs = extract_tool_outputs
        self._react_max_internal_steps_default = react_max_internal_steps_default
        self._react_max_internal_steps_workflow = react_max_internal_steps_workflow
        self._react_max_internal_steps_turn = react_max_internal_steps_turn
        self._react_max_internal_steps_budget_exhausted = react_max_internal_steps_budget_exhausted

    def execute_agent(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        if agent_id not in self._registry.agents:
            raise RuntimeError(f"Agent not found: {agent_id}")

        spec = self._registry.agents[agent_id]
        llm = self._resolve_llm(spec)
        tool_resolver = self._tool_budget.build_tool_resolver(tool_budget, node_name=node_name)
        runnable = self._build_agent_from_spec(spec, llm, tool_resolver)

        if spec.mode == "react":
            self._execute_react_agent(state, node_name, agent_id, spec, runnable)
            return

        payload = self.build_chain_payload(state, node_name, agent_id)
        raw = runnable.invoke(payload)
        text = self._coerce_text(raw)
        parsed = self._normalize_agent_parsed_payload(text, self._try_parse_json(text))

        state["io"]["last_model_output"] = text
        state["io"]["last_tool_outputs"] = []
        state["context"]["messages"].append(AIMessage(content=text))
        self._apply_agent_output(state, node_name, agent_id, text, parsed)

    async def execute_agent_async(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        *,
        tool_budget: Optional[dict[str, Any]] = None,
    ) -> None:
        if agent_id not in self._registry.agents:
            raise RuntimeError(f"Agent not found: {agent_id}")

        spec = self._registry.agents[agent_id]
        llm = self._resolve_llm(spec)
        tool_resolver = self._tool_budget.build_tool_resolver(tool_budget, node_name=node_name)
        runnable = self._build_agent_from_spec(spec, llm, tool_resolver)

        if spec.mode == "react":
            await self._execute_react_agent_async(state, node_name, agent_id, spec, runnable)
            return

        payload = self.build_chain_payload(state, node_name, agent_id)
        raw = await self._invoke_async(runnable, payload, None)
        text = self._coerce_text(raw)
        parsed = self._normalize_agent_parsed_payload(text, self._try_parse_json(text))

        state["io"]["last_model_output"] = text
        state["io"]["last_tool_outputs"] = []
        state["context"]["messages"].append(AIMessage(content=text))
        self._apply_agent_output(state, node_name, agent_id, text, parsed)

    def execute_react_agent(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        spec: AgentSpec,
        runnable: Any,
    ) -> None:
        self._execute_react_agent(state, node_name, agent_id, spec, runnable)

    async def execute_react_agent_async(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        spec: AgentSpec,
        runnable: Any,
    ) -> None:
        await self._execute_react_agent_async(state, node_name, agent_id, spec, runnable)

    def build_chain_payload(self, state: RuntimeState, node_name: str, agent_id: str) -> Dict[str, Any]:
        del node_name, agent_id
        artifacts = state.get("artifacts", {})
        artifacts_dict = artifacts if isinstance(artifacts, dict) else {}
        return {
            "user_text": state["input"].get("user_text", ""),
            "messages": self._context_facility.messages_to_text(
                state["context"].get("messages", []),
                scope="default",
            ),
            "artifacts": json.dumps(artifacts_dict, ensure_ascii=False, default=str),
            "supervisor_instruction": artifacts_dict.get("supervisor_instruction", ""),
        }

    def react_max_internal_steps_for_state(self, state: RuntimeState, spec: AgentSpec) -> int:
        base = self._react_max_internal_steps_default
        runtime_state = state.get("runtime", {})
        if isinstance(runtime_state, dict):
            budget = runtime_state.get("tool_budget")
            if isinstance(budget, dict):
                scope = budget.get("scope")
                if scope == "workflow":
                    base = self._react_max_internal_steps_workflow
                if scope == "turn":
                    base = self._react_max_internal_steps_turn

        remaining = self.remaining_tool_budget_for_agent(state, spec)
        if remaining is None:
            return base
        if remaining <= 0:
            return self._react_max_internal_steps_budget_exhausted
        return min(base, max(2, remaining + 2))

    def remaining_tool_budget_for_agent(self, state: RuntimeState, spec: AgentSpec) -> Optional[int]:
        runtime_state = state.get("runtime")
        if not isinstance(runtime_state, dict):
            return None
        budget = runtime_state.get("tool_budget")
        if not isinstance(budget, dict):
            return None
        limits = budget.get("limits")
        counts = budget.get("counts")
        if not isinstance(limits, dict) or not isinstance(counts, dict):
            return None

        remaining_values: list[int] = []
        for tool_id in spec.tools:
            if tool_id not in limits:
                continue
            try:
                limit = max(0, int(limits.get(tool_id, 0)))
                used = max(0, int(counts.get(tool_id, 0)))
            except Exception:
                continue
            remaining_values.append(max(0, limit - used))

        if not remaining_values:
            return None
        return sum(remaining_values)

    def invoke_react_sync(
        self,
        runnable: Any,
        payload: dict[str, Any],
        *,
        max_internal_steps: int,
    ) -> Any:
        invoke = getattr(runnable, "invoke", None)
        if not callable(invoke):
            raise TypeError(f"Runnable {type(runnable).__name__} has no invoke for react execution")
        config = {"recursion_limit": max(1, int(max_internal_steps))}
        try:
            return invoke(payload, config=config)
        except TypeError:
            return invoke(payload)

    async def invoke_react_async(
        self,
        runnable: Any,
        payload: dict[str, Any],
        *,
        max_internal_steps: int,
    ) -> Any:
        config = {"recursion_limit": max(1, int(max_internal_steps))}
        return await self._invoke_async(runnable, payload, config)

    def _execute_react_agent(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        spec: AgentSpec,
        runnable: Any,
    ) -> None:
        messages = list(state["context"].get("messages", []))
        max_internal_steps = self.react_max_internal_steps_for_state(state, spec)
        raw = self.invoke_react_sync(
            runnable,
            {"messages": messages},
            max_internal_steps=max_internal_steps,
        )

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
        spec: AgentSpec,
        runnable: Any,
    ) -> None:
        messages = list(state["context"].get("messages", []))
        max_internal_steps = self.react_max_internal_steps_for_state(state, spec)
        raw = await self.invoke_react_async(
            runnable,
            {"messages": messages},
            max_internal_steps=max_internal_steps,
        )

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
