from __future__ import annotations

import heapq
from typing import Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_RESERVED_LIMIT_KEYS = {"max_steps", "max_loops"}
_NODE_VISIT_PREFIX = "max_visits_"
_GENERIC_MAX_PREFIX = "max_"
_DEFAULT_DERIVED_MAX_STEPS = 30


class LLMConfig(BaseModel):
    # Strict mode: agent must reference a named LLM profile from config/llms.yaml.
    name: str
    temperature: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class LLMProfileSpec(BaseModel):
    name: str
    model_name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0

    model_config = ConfigDict(extra="forbid")


class HooksConfig(BaseModel):
    pre_run: Optional[str] = None
    post_run: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class AgentSpec(BaseModel):
    id: str
    description: str = Field(
        validation_alias=AliasChoices("description", "name"),
        serialization_alias="description",
    )
    input_requirements: List[str] = Field(default_factory=list)
    mode: Literal["chain", "react"]
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    llm: LLMConfig
    hooks: Optional[HooksConfig] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def name(self) -> str:
        # Backward-compatible alias for legacy call sites/tests.
        return self.description

    @field_validator("input_requirements", mode="before")
    @classmethod
    def _normalize_input_requirements(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
        return result

    @model_validator(mode="after")
    def _validate_tools_for_mode(self) -> "AgentSpec":
        if self.mode == "chain" and self.tools:
            raise ValueError(
                "AgentSpec.tools invalid: chain mode does not support tools; set tools to [] "
                "or switch mode to 'react'"
            )
        return self


class WorkflowSpec(BaseModel):
    id: str
    description: str = Field(
        validation_alias=AliasChoices("description", "name"),
        serialization_alias="description",
    )
    input_requirements: List[str] = Field(default_factory=list)
    entry_node: str
    nodes: Dict[str, dict]
    edges: List[dict]
    limits: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def name(self) -> str:
        # Backward-compatible alias for legacy call sites/tests.
        return self.description

    @field_validator("input_requirements", mode="before")
    @classmethod
    def _normalize_workflow_input_requirements(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
        return result

    @model_validator(mode="after")
    def _validate_graph(self) -> "WorkflowSpec":
        if self.entry_node not in self.nodes:
            raise ValueError(f"entry_node '{self.entry_node}' is not defined in nodes")

        for idx, edge in enumerate(self.edges):
            if not isinstance(edge, dict):
                raise ValueError(f"edge[{idx}] must be a mapping")
            source = edge.get("from")
            target = edge.get("to")
            if not isinstance(source, str) or not source:
                raise ValueError(f"edge[{idx}].from must be a non-empty string")
            if not isinstance(target, str) or not target:
                raise ValueError(f"edge[{idx}].to must be a non-empty string")
            if source not in self.nodes:
                raise ValueError(f"edge[{idx}].from references unknown node '{source}'")
            if target not in self.nodes:
                raise ValueError(f"edge[{idx}].to references unknown node '{target}'")

        self._validate_limits()
        return self

    def node_visit_limits(self) -> Dict[str, int]:
        return self._extract_node_visit_limits()

    def resolved_max_steps(self) -> int:
        if "max_steps" in self.limits:
            return int(self.limits.get("max_steps", _DEFAULT_DERIVED_MAX_STEPS))

        min_steps_to_terminal = self._min_steps_to_terminal()
        node_limits_sum = sum(self._extract_node_visit_limits().values())
        if min_steps_to_terminal is None:
            return max(1, _DEFAULT_DERIVED_MAX_STEPS, node_limits_sum)
        max_loops = self.resolved_max_loops(max_steps=max(1, min_steps_to_terminal))
        per_loop_budget = self._max_loop_cycle_agent_steps()
        derived = min_steps_to_terminal + (max_loops * per_loop_budget)
        return max(1, derived, node_limits_sum)

    def resolved_max_loops(self, *, max_steps: Optional[int] = None) -> int:
        if "max_loops" in self.limits:
            return int(self.limits.get("max_loops", 6))
        resolved_steps = max_steps if max_steps is not None else self.resolved_max_steps()
        return min(6, max(1, int(resolved_steps)))

    def _validate_limits(self) -> None:
        max_steps_explicit = "max_steps" in self.limits
        max_steps = self.resolved_max_steps()
        if max_steps < 1:
            raise ValueError("limits.max_steps must be >= 1")

        max_loops_explicit = "max_loops" in self.limits
        max_loops = self.resolved_max_loops(max_steps=max_steps)
        if max_loops < 0:
            raise ValueError("limits.max_loops must be >= 0")
        if max_loops > max_steps:
            raise ValueError("limits.max_loops must be <= limits.max_steps")

        node_limits = self._extract_node_visit_limits()
        for node_name, limit in node_limits.items():
            if limit < 1:
                raise ValueError(f"limits for node '{node_name}' must be >= 1")
            node_spec = self.nodes.get(node_name)
            if not isinstance(node_spec, dict) or node_spec.get("type") != "agent":
                raise ValueError(f"limits for node '{node_name}' only support agent nodes")

        if node_limits and sum(node_limits.values()) > max_steps:
            raise ValueError(
                "sum of node visit limits must be <= limits.max_steps "
                f"(sum={sum(node_limits.values())}, max_steps={max_steps})"
            )

        min_steps_to_terminal = self._min_steps_to_terminal()
        if min_steps_to_terminal is not None and max_steps < min_steps_to_terminal:
            raise ValueError(
                "limits.max_steps is too small to reach any terminal node "
                f"(required>={min_steps_to_terminal}, got={max_steps})"
            )
        if (
            max_steps_explicit
            and max_loops_explicit
            and min_steps_to_terminal is not None
            and (
                min_steps_to_terminal + (max_loops * self._max_loop_cycle_agent_steps())
            ) > max_steps
        ):
            required = min_steps_to_terminal + (max_loops * self._max_loop_cycle_agent_steps())
            raise ValueError(
                "limits are inconsistent: max_steps must cover baseline path plus "
                "max_loops * per-loop cycle budget "
                f"(required>={required}, got={max_steps})"
            )

    def _extract_node_visit_limits(self) -> Dict[str, int]:
        node_limits: Dict[str, int] = {}
        source_keys: Dict[str, str] = {}
        for raw_key, raw_value in self.limits.items():
            key = str(raw_key)
            node_name = self._node_limit_key_to_node_name(key)
            if node_name is None:
                continue
            value = int(raw_value)
            previous_key = source_keys.get(node_name)
            if previous_key is not None and previous_key != key:
                raise ValueError(
                    f"duplicate node visit limit keys for '{node_name}': '{previous_key}' and '{key}'"
                )
            source_keys[node_name] = key
            node_limits[node_name] = value
        return node_limits

    def _node_limit_key_to_node_name(self, key: str) -> Optional[str]:
        if key in _RESERVED_LIMIT_KEYS:
            return None
        if key.startswith(_NODE_VISIT_PREFIX):
            node_name = key[len(_NODE_VISIT_PREFIX):].strip()
            if not node_name:
                raise ValueError(f"invalid limit key: '{key}'")
            if node_name not in self.nodes:
                raise ValueError(f"limit key '{key}' references unknown node '{node_name}'")
            return node_name
        if key.startswith(_GENERIC_MAX_PREFIX):
            node_name = key[len(_GENERIC_MAX_PREFIX):].strip()
            if not node_name:
                raise ValueError(f"invalid limit key: '{key}'")
            if node_name in self.nodes:
                return node_name
            return None
        return None

    def _min_steps_to_terminal(self) -> Optional[int]:
        terminal_nodes = [
            name
            for name, spec in self.nodes.items()
            if isinstance(spec, dict) and spec.get("type") == "terminal"
        ]
        if not terminal_nodes:
            return None

        edge_map: Dict[str, list[str]] = {}
        for edge in self.edges:
            edge_map.setdefault(str(edge["from"]), []).append(str(edge["to"]))

        start_cost = 1 if self._is_agent_node(self.entry_node) else 0
        distances: Dict[str, int] = {self.entry_node: start_cost}
        heap: list[tuple[int, str]] = [(start_cost, self.entry_node)]

        while heap:
            cost, node = heapq.heappop(heap)
            if cost != distances.get(node):
                continue
            for nxt in edge_map.get(node, []):
                next_cost = cost + (1 if self._is_agent_node(nxt) else 0)
                if next_cost < distances.get(nxt, 10**9):
                    distances[nxt] = next_cost
                    heapq.heappush(heap, (next_cost, nxt))

        reachable_costs = [distances[name] for name in terminal_nodes if name in distances]
        if not reachable_costs:
            return None
        return min(reachable_costs)

    def _max_loop_cycle_agent_steps(self) -> int:
        edge_map: Dict[str, list[str]] = {}
        for edge in self.edges:
            source = str(edge["from"])
            target = str(edge["to"])
            edge_map.setdefault(source, []).append(target)

        reachable = self._reachable_from_entry(edge_map)
        terminal_reachable = self._nodes_reaching_terminal(edge_map)

        max_cycle_budget = 1
        for edge in self.edges:
            source = str(edge["from"])
            target = str(edge["to"])
            if source not in reachable or target not in reachable:
                continue
            if source not in terminal_reachable:
                continue
            cycle_cost = self._shortest_agent_steps_between(
                start_node=target,
                target_node=source,
                edge_map=edge_map,
            )
            if cycle_cost is None:
                continue
            max_cycle_budget = max(max_cycle_budget, max(1, cycle_cost))
        return max_cycle_budget

    def _shortest_agent_steps_between(
        self,
        *,
        start_node: str,
        target_node: str,
        edge_map: Dict[str, list[str]],
    ) -> Optional[int]:
        start_cost = 1 if self._is_agent_node(start_node) else 0
        distances: Dict[str, int] = {start_node: start_cost}
        heap: list[tuple[int, str]] = [(start_cost, start_node)]

        while heap:
            cost, node = heapq.heappop(heap)
            if cost != distances.get(node):
                continue
            if node == target_node:
                return cost
            for nxt in edge_map.get(node, []):
                next_cost = cost + (1 if self._is_agent_node(nxt) else 0)
                if next_cost < distances.get(nxt, 10**9):
                    distances[nxt] = next_cost
                    heapq.heappush(heap, (next_cost, nxt))
        return None

    def _reachable_from_entry(self, edge_map: Dict[str, list[str]]) -> set[str]:
        stack = [self.entry_node]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edge_map.get(node, []))
        return seen

    def _nodes_reaching_terminal(self, edge_map: Dict[str, list[str]]) -> set[str]:
        reverse_map: Dict[str, list[str]] = {}
        for src, targets in edge_map.items():
            for dst in targets:
                reverse_map.setdefault(dst, []).append(src)
        terminal_nodes = [
            name
            for name, spec in self.nodes.items()
            if isinstance(spec, dict) and spec.get("type") == "terminal"
        ]
        stack = list(terminal_nodes)
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(reverse_map.get(node, []))
        return seen

    def _is_agent_node(self, node_name: str) -> bool:
        node = self.nodes.get(node_name)
        return isinstance(node, dict) and node.get("type") == "agent"
