from __future__ import annotations

from typing import Any

from src.application.runtime.spec_models import WorkflowSpec


class WorkflowRuntime:
    def __init__(self, spec: WorkflowSpec, agent_runner: Any):
        self.spec = spec
        self.agent_runner = agent_runner

    def next_node(self, current_node: str, state: dict[str, Any]) -> str:
        edges = [edge for edge in self.spec.edges if edge["from"] == current_node]
        if not edges:
            raise RuntimeError(f"No valid transition from node: {current_node}")

        route_key = self._resolve_route_key(state)
        conditional_edges = [edge for edge in edges if edge.get("condition") is not None]

        for edge in conditional_edges:
            if self._condition_matches(edge.get("condition"), state, route_key):
                return edge["to"]

        for edge in edges:
            if edge.get("condition") is None:
                return edge["to"]

        raise RuntimeError(f"No valid transition from node: {current_node}")

    def enforce_limits(self, state: dict[str, Any]) -> None:
        max_steps = self.spec.limits.get("max_steps", 30)
        max_loops = self.spec.limits.get("max_loops", 6)
        if state.get("_step_count", 0) >= max_steps:
            raise RuntimeError("max_steps exceeded")
        if state.get("_loop_count", 0) >= max_loops:
            raise RuntimeError("max_loops exceeded")

    def _resolve_route_key(self, state: dict[str, Any]) -> str | None:
        route_key = state.get("route_key")
        if route_key:
            return route_key
        return None

    def _condition_matches(
        self, condition: Any, state: dict[str, Any], route_key: str | None
    ) -> bool:
        if isinstance(condition, str):
            if route_key:
                return route_key == condition
            return condition in self._string_condition_candidates(state)

        if isinstance(condition, dict):
            field = condition.get("field")
            equals = condition.get("equals")
            if not field:
                return False
            value = self._extract_field(state, field)
            return value == equals

        return False

    def _extract_field(self, state: dict[str, Any], field_path: str) -> Any:
        value: Any = state
        for part in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                return None
        return value

    def _string_condition_candidates(self, state: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        research_plan = state.get("research_plan")
        if research_plan is not None:
            plan_step_type = getattr(research_plan, "step_type", None)
            if plan_step_type:
                candidates.append(plan_step_type)
            elif isinstance(research_plan, dict):
                step_type = research_plan.get("step_type")
                if step_type:
                    candidates.append(step_type)

        research_critic = state.get("research_critic")
        if research_critic is not None:
            critic_valid = getattr(research_critic, "is_valid", None)
            if critic_valid is not None:
                candidates.append("valid" if critic_valid else "revise")
            elif isinstance(research_critic, dict) and "is_valid" in research_critic:
                candidates.append("valid" if research_critic["is_valid"] else "revise")

        return candidates
