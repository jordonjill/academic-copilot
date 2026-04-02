from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.application.runtime.contracts.spec_models import WorkflowSpec

logger = logging.getLogger(__name__)


class WorkflowRuntime:
    """Workflow guardrail layer: relation constraints + transition + limits."""

    def __init__(self, spec: WorkflowSpec, agent_runner: Any):
        self.spec = spec
        self.agent_runner = agent_runner
        self._node_visit_limits = self.spec.node_visit_limits()

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

        if self._loop_saturated(current_state):
            fallback = self._preferred_fallback_edge(edges)
            logger.warning(
                "workflow.loop_saturated_fallback workflow_id=%s current_node=%s to=%s",
                self.spec.id,
                current_node,
                fallback["to"],
            )
            return fallback["to"]

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
        max_steps = self.spec.resolved_max_steps()
        max_loops = self.spec.resolved_max_loops(max_steps=max_steps)
        if state.get("_step_count", 0) >= max_steps:
            raise RuntimeError("max_steps exceeded")
        if state.get("_loop_count", 0) >= max_loops:
            logger.warning(
                "workflow.max_loops_reached workflow_id=%s loop_count=%s max_loops=%s continue=true",
                self.spec.id,
                state.get("_loop_count", 0),
                max_loops,
            )

    def is_node_visit_saturated(self, node_name: str, visit_counts: dict[str, int]) -> bool:
        limit = self._node_visit_limits.get(node_name)
        if limit is None:
            return False
        return int(visit_counts.get(node_name, 0)) > int(limit)

    def next_node_for_saturated_node(self, current_node: str) -> str:
        edges = [edge for edge in self.spec.edges if edge["from"] == current_node]
        if not edges:
            raise RuntimeError(f"No valid transition from node: {current_node}")
        fallback = self._preferred_fallback_edge(edges, avoid_node=current_node)
        return fallback["to"]

    def _loop_saturated(self, state: dict[str, Any]) -> bool:
        max_steps = self.spec.resolved_max_steps()
        max_loops = self.spec.resolved_max_loops(max_steps=max_steps)
        if max_loops < 0:
            return False
        loop_count = state.get("_loop_count")
        if not isinstance(loop_count, int):
            runtime = state.get("runtime")
            if isinstance(runtime, dict):
                loop_count = runtime.get("loop_count")
        if not isinstance(loop_count, int):
            return False
        return loop_count >= max_loops

    @staticmethod
    def _preferred_fallback_edge(
        edges: list[dict[str, Any]],
        avoid_node: str | None = None,
    ) -> dict[str, Any]:
        if avoid_node:
            for edge in edges:
                if edge.get("condition") is None and edge.get("to") != avoid_node:
                    return edge
            for edge in edges:
                if edge.get("to") != avoid_node:
                    return edge
        for edge in edges:
            if edge.get("condition") is None:
                return edge
        return edges[0]

    def _condition_matches(self, condition: Any, state: dict[str, Any]) -> bool:
        if condition is None:
            logger.debug(
                "workflow.condition.none workflow_id=%s state_keys=%s",
                self.spec.id,
                sorted(state.keys()),
            )
            return False
        if isinstance(condition, dict):
            field = condition.get("field")
            if not field:
                return False
            value, exists = self._extract_field_with_presence(state, field)
            if not exists:
                logger.debug(
                    "workflow.condition.field_missing workflow_id=%s field=%s condition=%s",
                    self.spec.id,
                    field,
                    condition,
                )
                return False
            op = str(condition.get("op") or "eq").strip().lower()
            has_expected = ("value" in condition) or ("equals" in condition)
            expected = condition.get("value", condition.get("equals"))
            if op not in {"exists", "truthy", "falsy"} and not has_expected:
                logger.warning(
                    "workflow.condition.missing_expected workflow_id=%s field=%s op=%s condition=%s",
                    self.spec.id,
                    field,
                    op,
                    condition,
                )
                return False
            return self._evaluate_condition(value, expected, op)

        if isinstance(condition, str):
            condition_value = condition.strip().lower()
            expr_match = re.match(
                r"^\s*([A-Za-z0-9_.]+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$",
                condition,
            )
            if expr_match:
                field, symbol, raw_expected = expr_match.groups()
                value, exists = self._extract_field_with_presence(state, field)
                if not exists:
                    logger.debug(
                        "workflow.expression.field_missing workflow_id=%s field=%s condition=%s",
                        self.spec.id,
                        field,
                        condition,
                    )
                    return False
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
                logger.debug(
                    "workflow.condition.numeric_coerce_failed workflow_id=%s value=%r expected=%r op=%s",
                    self.spec.id,
                    value,
                    expected,
                    op,
                )
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
        value, _ = self._extract_field_with_presence(state, field_path)
        return value

    def _extract_field_with_presence(self, state: dict[str, Any], field_path: str) -> tuple[Any, bool]:
        value: Any = state
        for part in field_path.split("."):
            if isinstance(value, dict):
                if part not in value:
                    return None, False
                value = value.get(part)
            else:
                if not hasattr(value, part):
                    return None, False
                value = getattr(value, part)
            if value is None:
                return None, True
        return value, True

    def _string_condition_candidates(self, state: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        artifacts = state.get("artifacts", {})
        if isinstance(artifacts, dict):
            research_plan = artifacts.get("research_plan")
            if isinstance(research_plan, dict):
                step_type = research_plan.get("step_type")
                if isinstance(step_type, str):
                    candidates.append(step_type.strip().lower())

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
