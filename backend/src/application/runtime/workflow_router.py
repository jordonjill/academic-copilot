from __future__ import annotations

import json
import re
from typing import Any

from src.application.runtime.spec_models import WorkflowSpec


class WorkflowRuntime:
    """Workflow guardrail layer: relation constraints + transition + limits."""

    def __init__(self, spec: WorkflowSpec, agent_runner: Any):
        self.spec = spec
        self.agent_runner = agent_runner

    def allowed_next_nodes(self, current_node: str) -> list[str]:
        return [edge["to"] for edge in self.spec.edges if edge["from"] == current_node]

    def assert_transition_allowed(self, current_node: str, next_node: str) -> None:
        allowed = self.allowed_next_nodes(current_node)
        if next_node not in allowed:
            raise RuntimeError(
                f"Invalid transition: {current_node} -> {next_node}; allowed: {allowed}"
            )

    def next_node(self, current_node: str, state: dict[str, Any] | None = None) -> str:
        edges = [edge for edge in self.spec.edges if edge["from"] == current_node]
        if not edges:
            raise RuntimeError(f"No valid transition from node: {current_node}")

        current_state = state or {}
        conditional_edges = [edge for edge in edges if edge.get("condition") is not None]

        for edge in conditional_edges:
            if self._condition_matches(edge.get("condition"), current_state):
                return edge["to"]

        for edge in edges:
            if edge.get("condition") is None:
                return edge["to"]

        # If all edges are conditional and none matches, fallback to first edge.
        return edges[0]["to"]

    def enforce_limits(self, state: dict[str, Any]) -> None:
        max_steps = self.spec.limits.get("max_steps", 30)
        max_loops = self.spec.limits.get("max_loops", 6)
        if state.get("_step_count", 0) >= max_steps:
            raise RuntimeError("max_steps exceeded")
        if state.get("_loop_count", 0) >= max_loops:
            raise RuntimeError("max_loops exceeded")

    def _condition_matches(self, condition: Any, state: dict[str, Any]) -> bool:
        if isinstance(condition, dict):
            field = condition.get("field")
            if not field:
                return False
            value = self._extract_field(state, field)
            op = str(condition.get("op") or "eq").strip().lower()
            expected = condition.get("value", condition.get("equals"))
            return self._evaluate_condition(value, expected, op)

        if isinstance(condition, str):
            condition_value = condition.strip().lower()
            expr_match = re.match(
                r"^\s*([A-Za-z0-9_.]+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$",
                condition,
            )
            if expr_match:
                field, symbol, raw_expected = expr_match.groups()
                value = self._extract_field(state, field)
                expected = self._parse_literal(raw_expected)
                op = {
                    "==": "eq",
                    "!=": "ne",
                    ">": "gt",
                    ">=": "gte",
                    "<": "lt",
                    "<=": "lte",
                }[symbol]
                return self._evaluate_condition(value, expected, op)
            for candidate in self._string_condition_candidates(state):
                if candidate == condition_value:
                    return True
            return False

        return False

    def _evaluate_condition(self, value: Any, expected: Any, op: str) -> bool:
        if op in {"eq", "=="}:
            return value == expected
        if op in {"ne", "!="}:
            return value != expected
        if op == "in":
            if isinstance(expected, (list, tuple, set)):
                return value in expected
            return False
        if op == "not_in":
            if isinstance(expected, (list, tuple, set)):
                return value not in expected
            return False
        if op == "contains":
            if isinstance(value, (list, tuple, set, str)):
                return expected in value
            return False
        if op == "exists":
            return value is not None
        if op == "truthy":
            return bool(value)
        if op == "falsy":
            return not bool(value)
        if op in {"gt", "gte", "lt", "lte"}:
            try:
                left = float(value)
                right = float(expected)
            except (TypeError, ValueError):
                return False
            if op == "gt":
                return left > right
            if op == "gte":
                return left >= right
            if op == "lt":
                return left < right
            return left <= right
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
        artifacts = state.get("artifacts", {})
        if isinstance(artifacts, dict):
            research_plan = artifacts.get("research_plan")
            if isinstance(research_plan, dict):
                step_type = research_plan.get("step_type")
                if isinstance(step_type, str):
                    candidates.append(step_type.lower())

            research_critic = artifacts.get("research_critic")
            if isinstance(research_critic, dict) and "is_valid" in research_critic:
                candidates.append("valid" if research_critic["is_valid"] else "revise")

        return candidates

    def _parse_literal(self, raw: str) -> Any:
        text = raw.strip()
        if not text:
            return ""
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return json.loads(text)
        except Exception:
            return text
