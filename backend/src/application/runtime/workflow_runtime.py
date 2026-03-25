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

        for edge in edges:
            condition = edge.get("condition")
            if condition is None:
                return edge["to"]
            if self._condition_matches(condition, state, route_key):
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

        research_plan = state.get("research_plan")
        if research_plan is None:
            return None

        plan_step_type = getattr(research_plan, "step_type", None)
        if plan_step_type:
            return plan_step_type

        if isinstance(research_plan, dict):
            return research_plan.get("step_type")

        return None

    def _condition_matches(
        self, condition: Any, state: dict[str, Any], route_key: str | None
    ) -> bool:
        if isinstance(condition, str):
            return bool(route_key and route_key == condition)

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
