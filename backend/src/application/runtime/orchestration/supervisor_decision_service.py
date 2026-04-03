from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from langchain_core.messages import AIMessage

from src.application.runtime.contracts.io_models import SupervisorDecision
from src.application.runtime.contracts.spec_models import AgentSpec
from src.application.runtime.contracts.state_types import RuntimeState

_SUPERVISOR_ACTION_MAP = {
    "run_subagent": "run_agent",
    "start_workflow": "run_workflow",
}


class SupervisorDecisionService:
    def __init__(
        self,
        *,
        registry: Any,
        logger: Any,
        build_agent_from_spec_fn: Callable[[AgentSpec, Any, Callable[[str], Any]], Any],
        resolve_supervisor_spec: Callable[[], Optional[AgentSpec]],
        resolve_llm: Callable[[AgentSpec], Any],
        resolve_tool: Callable[[str], Any],
        build_supervisor_payload: Callable[[RuntimeState, AgentSpec, Optional[str], bool], Dict[str, Any]],
        coerce_text: Callable[[Any], str],
        try_parse_supervisor_decision_json: Callable[[str], Optional[Dict[str, Any]]],
        invoke_async: Callable[[Any, Any, Optional[dict[str, Any]]], Awaitable[Any]],
    ) -> None:
        self._registry = registry
        self._logger = logger
        self._build_agent_from_spec = build_agent_from_spec_fn
        self._resolve_supervisor_spec = resolve_supervisor_spec
        self._resolve_llm = resolve_llm
        self._resolve_tool = resolve_tool
        self._build_supervisor_payload = build_supervisor_payload
        self._coerce_text = coerce_text
        self._try_parse_supervisor_decision_json = try_parse_supervisor_decision_json
        self._invoke_async = invoke_async

    def decide_next_action(
        self,
        *,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
    ) -> Dict[str, Any]:
        llm = self._resolve_llm(supervisor_spec)
        runnable = self._build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = runnable.invoke(
            self._build_supervisor_payload(
                state,
                supervisor_spec,
                requested_workflow_id,
                False,
            )
        )
        text = self._coerce_text(raw)
        if not text.strip():
            self._logger.warning(
                "supervisor.empty_decision_output agent_id=%s step_count=%s workflow_id=%s",
                supervisor_spec.id,
                state["runtime"].get("step_count", 0),
                requested_workflow_id,
            )
        parsed = self._try_parse_supervisor_decision_json(text)
        state["io"]["last_model_output"] = text
        if isinstance(parsed, dict):
            return self.normalize_supervisor_decision(parsed=parsed, raw_text=text, state=state)
        return {"action": "direct_reply", "final_text": text}

    async def decide_next_action_async(
        self,
        *,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
    ) -> Dict[str, Any]:
        llm = self._resolve_llm(supervisor_spec)
        runnable = self._build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = await self._invoke_async(
            runnable,
            self._build_supervisor_payload(
                state,
                supervisor_spec,
                requested_workflow_id,
                False,
            ),
            None,
        )
        text = self._coerce_text(raw)
        if not text.strip():
            self._logger.warning(
                "supervisor.empty_decision_output agent_id=%s step_count=%s workflow_id=%s",
                supervisor_spec.id,
                state["runtime"].get("step_count", 0),
                requested_workflow_id,
            )
        parsed = self._try_parse_supervisor_decision_json(text)
        state["io"]["last_model_output"] = text
        if isinstance(parsed, dict):
            return self.normalize_supervisor_decision(parsed=parsed, raw_text=text, state=state)
        return {"action": "direct_reply", "final_text": text}

    def finalize_with_supervisor(
        self,
        *,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
    ) -> None:
        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None or supervisor_spec.mode != "chain":
            return

        llm = self._resolve_llm(supervisor_spec)
        runnable = self._build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = runnable.invoke(
            self._build_supervisor_payload(
                state,
                supervisor_spec,
                requested_workflow_id,
                True,
            )
        )
        text = self._coerce_text(raw)
        parsed = self._try_parse_supervisor_decision_json(text)
        decision = (
            self.normalize_supervisor_decision(parsed=parsed, raw_text=text, state=state)
            if isinstance(parsed, dict)
            else {"action": "direct_reply", "final_text": text}
        )
        if decision.get("action") != "direct_reply":
            return
        final_text = decision.get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            state["output"]["final_text"] = final_text
            state["io"]["last_model_output"] = final_text
            state["context"]["messages"].append(AIMessage(content=final_text))

    async def finalize_with_supervisor_async(
        self,
        *,
        state: RuntimeState,
        requested_workflow_id: Optional[str],
    ) -> None:
        supervisor_spec = self._resolve_supervisor_spec()
        if supervisor_spec is None or supervisor_spec.mode != "chain":
            return

        llm = self._resolve_llm(supervisor_spec)
        runnable = self._build_agent_from_spec(supervisor_spec, llm, self._resolve_tool)
        raw = await self._invoke_async(
            runnable,
            self._build_supervisor_payload(
                state,
                supervisor_spec,
                requested_workflow_id,
                True,
            ),
            None,
        )
        text = self._coerce_text(raw)
        parsed = self._try_parse_supervisor_decision_json(text)
        decision = (
            self.normalize_supervisor_decision(parsed=parsed, raw_text=text, state=state)
            if isinstance(parsed, dict)
            else {"action": "direct_reply", "final_text": text}
        )
        if decision.get("action") != "direct_reply":
            return
        final_text = decision.get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            state["output"]["final_text"] = final_text
            state["io"]["last_model_output"] = final_text
            state["context"]["messages"].append(AIMessage(content=final_text))

    def normalize_supervisor_decision(
        self,
        *,
        parsed: Dict[str, Any],
        raw_text: str,
        state: RuntimeState,
    ) -> Dict[str, Any]:
        decision_model = self.coerce_supervisor_decision_model(parsed=parsed, raw_text=raw_text)
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
            "inline_input_artifacts": dict(decision_model.inline_input_artifacts),
        }
        if isinstance(decision_model.instruction, str) and decision_model.instruction.strip():
            normalized["instruction"] = decision_model.instruction
        return normalized

    def coerce_supervisor_decision_model(
        self,
        *,
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
            "inline_input_artifacts": (
                raw_payload.get("inline_input_artifacts")
                if isinstance(raw_payload.get("inline_input_artifacts"), dict)
                else (
                    raw_payload.get("input_artifacts")
                    if isinstance(raw_payload.get("input_artifacts"), dict)
                    else {}
                )
            ),
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

    def resolve_workflow_target(self, decision: Dict[str, Any], state: RuntimeState) -> Optional[str]:
        del state
        target = decision.get("target")
        if isinstance(target, str) and target in self._registry.workflows:
            return target

        workflow_hint = decision.get("workflow_id")
        if isinstance(workflow_hint, str) and workflow_hint in self._registry.workflows:
            return workflow_hint

        return None

    def resolve_subagent_target(self, decision: Dict[str, Any]) -> Optional[str]:
        target = decision.get("target")
        if isinstance(target, str):
            return target
        agent_hint = decision.get("agent_id")
        if isinstance(agent_hint, str):
            return agent_hint
        return None
