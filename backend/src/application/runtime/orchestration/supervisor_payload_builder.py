from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.spec_models import AgentSpec
from src.application.runtime.contracts.state_types import RuntimeState
from src.application.runtime.providers.context_facility import ContextFacility


class SupervisorPayloadBuilder:
    def __init__(
        self,
        *,
        registry: ConfigRegistry,
        context_facility: ContextFacility,
        load_ltm_profile: Callable[[str], str],
    ) -> None:
        self._registry = registry
        self._context_facility = context_facility
        self._load_ltm_profile = load_ltm_profile

    def available_agents(self, supervisor_spec: AgentSpec) -> list[str]:
        subagents = getattr(self._registry, "subagents", None)
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
                for agent_id in self._registry.agents.keys()
                if agent_id != supervisor_spec.id and not agent_id.endswith("_router")
            ]
        )

    def available_workflows(self) -> list[str]:
        return sorted(self._registry.workflows.keys())

    def agent_capabilities(self, supervisor_spec: AgentSpec) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for agent_id in self.available_agents(supervisor_spec):
            spec = self._registry.agents.get(agent_id)
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

    def workflow_capabilities(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for workflow_id in self.available_workflows():
            spec = self._registry.workflows.get(workflow_id)
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

    def append_execution_trace(
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

    def build_supervisor_payload(
        self,
        *,
        state: RuntimeState,
        supervisor_spec: AgentSpec,
        requested_workflow_id: Optional[str],
        workflow_completed: bool,
    ) -> Dict[str, Any]:
        available_agents = self.available_agents(supervisor_spec)
        available_workflows = self.available_workflows()
        agent_capabilities = self.agent_capabilities(supervisor_spec)
        workflow_capabilities = self.workflow_capabilities()
        artifacts = state.get("artifacts", {})
        compact_artifacts = self._context_facility.compact_artifacts(
            artifacts if isinstance(artifacts, dict) else {},
            excluded_keys={"shared", "execution_trace", "task_input", "workflow_runner_input"},
        )
        ltm_profile = self._load_ltm_profile(str(state["input"].get("user_id", "")))
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
            "ltm_profile": ltm_profile,
            "recent_steps": json.dumps(recent_steps, ensure_ascii=False, default=str),
            "requested_workflow_id": requested_workflow_id or "",
            "step_count": state["runtime"].get("step_count", 0),
            "loop_count": state["runtime"].get("loop_count", 0),
            "workflow_completed": workflow_completed,
        }
