from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.contracts.state_types import RuntimeState

_TURN_TOOL_LIMITS_DEFAULT: dict[str, int] = {
    "scholar_search": 2,
    "paper_fetch": 4,
}
_DEFAULT_DERIVED_BUDGET_LOOPS = 1


class ToolBudgetManager:
    def __init__(
        self,
        *,
        registry: ConfigRegistry,
        resolve_tool: Callable[[str], BaseTool],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._registry = registry
        self._resolve_tool = resolve_tool
        self._logger = logger or logging.getLogger(__name__)

    def build_tool_resolver(
        self,
        tool_budget: Optional[dict[str, Any]] = None,
        *,
        node_name: Optional[str] = None,
    ) -> Callable[[str], Optional[BaseTool]]:
        if not isinstance(tool_budget, dict):
            return self._resolve_tool

        limits = tool_budget.get("limits", {})
        if not isinstance(limits, dict) or not limits:
            limits = {}
        counts = tool_budget.setdefault("counts", {})
        if not isinstance(counts, dict):
            counts = {}
            tool_budget["counts"] = counts
        hidden_logged = tool_budget.setdefault("_hidden_exhausted_tools", {})
        if not isinstance(hidden_logged, dict):
            hidden_logged = {}
            tool_budget["_hidden_exhausted_tools"] = hidden_logged
        node_visit_limits = tool_budget.get("node_visit_limits")
        if not isinstance(node_visit_limits, dict):
            node_visit_limits = {}
        node_limits = {}
        if isinstance(node_name, str) and node_name:
            maybe_node_limits = node_visit_limits.get(node_name)
            if isinstance(maybe_node_limits, dict):
                node_limits = maybe_node_limits
        local_counts: dict[str, int] = {}
        local_exceeded_logged: dict[str, bool] = {}

        def _resolver(tool_id: str) -> Optional[BaseTool]:
            base_tool = self._resolve_tool(tool_id)
            global_limit_value = limits.get(tool_id)
            local_limit_value = node_limits.get(tool_id)
            if global_limit_value is None and local_limit_value is None:
                return base_tool
            global_limit: Optional[int] = None
            if global_limit_value is not None:
                global_limit = max(0, int(global_limit_value))
            local_limit: Optional[int] = None
            if local_limit_value is not None:
                local_limit = max(0, int(local_limit_value))
            if global_limit is not None:
                used = max(0, int(counts.get(tool_id, 0)))
                if used >= global_limit:
                    if not hidden_logged.get(tool_id):
                        self._logger.info(
                            "tool.hidden_exhausted tool_id=%s used=%s limit=%s workflow_id=%s",
                            tool_id,
                            used,
                            global_limit,
                            tool_budget.get("workflow_id"),
                        )
                        hidden_logged[tool_id] = True
                    return None
            if local_limit is not None:
                local_used = max(0, int(local_counts.get(tool_id, 0)))
                if local_used >= local_limit:
                    local_key = f"{node_name}:{tool_id}"
                    if not local_exceeded_logged.get(local_key):
                        self._logger.info(
                            "tool.hidden_exhausted_per_visit node=%s tool_id=%s used=%s limit=%s workflow_id=%s",
                            node_name,
                            tool_id,
                            local_used,
                            local_limit,
                            tool_budget.get("workflow_id"),
                        )
                        local_exceeded_logged[local_key] = True
                    return None
            return self._wrap_tool_with_budget(
                tool_id=tool_id,
                tool=base_tool,
                global_limit=global_limit,
                tool_budget=tool_budget,
                local_limit=local_limit,
                local_counts=local_counts,
                local_exceeded_logged=local_exceeded_logged,
                node_name=node_name,
            )

        return _resolver

    def ensure_workflow_tool_budget(self, state: RuntimeState, workflow_id: str) -> dict[str, Any]:
        runtime_state = state.get("runtime")
        if not isinstance(runtime_state, dict):
            runtime_state = {}
            state["runtime"] = runtime_state

        existing = runtime_state.get("tool_budget")
        if isinstance(existing, dict) and existing.get("workflow_id") == workflow_id:
            counts = existing.get("counts")
            limits = existing.get("limits")
            node_visit_limits = existing.get("node_visit_limits")
            if isinstance(counts, dict) and isinstance(limits, dict) and isinstance(node_visit_limits, dict):
                return existing

        limits = self._workflow_tool_limits(workflow_id)
        node_visit_limits = self._workflow_node_tool_limits(workflow_id)
        budget = {
            "scope": "workflow",
            "workflow_id": workflow_id,
            "limits": limits,
            "node_visit_limits": node_visit_limits,
            "counts": {},
        }
        runtime_state["tool_budget"] = budget
        return budget

    def ensure_turn_tool_budget(self, state: RuntimeState) -> dict[str, Any]:
        runtime_state = state.get("runtime")
        if not isinstance(runtime_state, dict):
            runtime_state = {}
            state["runtime"] = runtime_state

        existing = runtime_state.get("tool_budget")
        if isinstance(existing, dict) and existing.get("scope") == "turn":
            counts = existing.get("counts")
            limits = existing.get("limits")
            if isinstance(counts, dict) and isinstance(limits, dict):
                return existing

        budget = {
            "scope": "turn",
            "workflow_id": None,
            "limits": dict(_TURN_TOOL_LIMITS_DEFAULT),
            "node_visit_limits": {},
            "counts": {},
        }
        runtime_state["tool_budget"] = budget
        return budget

    def _workflow_tool_limits(self, workflow_id: str) -> dict[str, int]:
        spec = self._registry.workflows.get(workflow_id)
        if spec is None:
            return {}

        raw_limits = spec.limits if isinstance(spec.limits, dict) else {}
        limits: dict[str, int] = {}

        for raw_key, raw_value in raw_limits.items():
            if not isinstance(raw_key, str) or not raw_key.startswith("max_tool_"):
                continue
            tool_id = raw_key[len("max_tool_"):].strip()
            if not tool_id:
                continue
            limits[tool_id] = max(0, int(raw_value))

        if not limits:
            node_limits = self._workflow_node_tool_limits(workflow_id)
            derived = self._derive_global_tool_limits_from_node_limits(spec, node_limits)
            limits.update(derived)

        return limits

    def _workflow_node_tool_limits(self, workflow_id: str) -> dict[str, dict[str, int]]:
        spec = self._registry.workflows.get(workflow_id)
        if spec is None:
            return {}

        raw_limits = spec.limits if isinstance(spec.limits, dict) else {}
        node_tool_limits: dict[str, dict[str, int]] = {}
        prefix = "max_node_tool_"
        for raw_key, raw_value in raw_limits.items():
            if not isinstance(raw_key, str) or not raw_key.startswith(prefix):
                continue
            suffix = raw_key[len(prefix):].strip()
            if not suffix:
                continue
            node_name = ""
            tool_id = ""
            if "__" in suffix:
                node_name, tool_id = suffix.split("__", 1)
            elif "_" in suffix:
                maybe_node, maybe_tool = suffix.split("_", 1)
                if maybe_node in spec.nodes:
                    node_name, tool_id = maybe_node, maybe_tool
            node_name = node_name.strip()
            tool_id = tool_id.strip()
            if not node_name or not tool_id or node_name not in spec.nodes:
                continue
            node_limits = node_tool_limits.setdefault(node_name, {})
            node_limits[tool_id] = max(0, int(raw_value))
        return node_tool_limits

    def _derive_global_tool_limits_from_node_limits(
        self,
        spec: Any,
        node_limits: dict[str, dict[str, int]],
    ) -> dict[str, int]:
        if not isinstance(node_limits, dict) or not node_limits:
            return {}
        raw_limits = spec.limits if isinstance(getattr(spec, "limits", None), dict) else {}
        loops_raw = raw_limits.get("max_loops", _DEFAULT_DERIVED_BUDGET_LOOPS)
        loops = max(1, int(loops_raw))

        derived: dict[str, int] = {}
        for per_node in node_limits.values():
            if not isinstance(per_node, dict):
                continue
            for tool_id, per_visit_limit in per_node.items():
                if not isinstance(tool_id, str) or not tool_id:
                    continue
                per_visit = max(0, int(per_visit_limit))
                candidate = loops * per_visit + 1
                derived[tool_id] = max(derived.get(tool_id, 0), candidate)
        return derived

    def _wrap_tool_with_budget(
        self,
        tool_id: str,
        tool: BaseTool,
        *,
        global_limit: Optional[int],
        tool_budget: dict[str, Any],
        local_limit: Optional[int] = None,
        local_counts: Optional[dict[str, int]] = None,
        local_exceeded_logged: Optional[dict[str, bool]] = None,
        node_name: Optional[str] = None,
    ) -> BaseTool:
        counts = tool_budget.setdefault("counts", {})
        if not isinstance(counts, dict):
            counts = {}
            tool_budget["counts"] = counts
        exceeded_logged = tool_budget.setdefault("_exceeded_logged", {})
        if not isinstance(exceeded_logged, dict):
            exceeded_logged = {}
            tool_budget["_exceeded_logged"] = exceeded_logged
        if local_limit is not None and local_counts is None:
            local_counts = {}
        if local_limit is not None and local_exceeded_logged is None:
            local_exceeded_logged = {}

        def _consume_or_reject() -> tuple[bool, int, str]:
            if local_limit is not None and isinstance(local_counts, dict):
                local_used = int(local_counts.get(tool_id, 0))
                if local_used >= local_limit:
                    local_key = f"{node_name}:{tool_id}"
                    if isinstance(local_exceeded_logged, dict) and not local_exceeded_logged.get(local_key):
                        self._logger.warning(
                            "tool.budget_exceeded_per_visit node=%s tool_id=%s used=%s limit=%s workflow_id=%s",
                            node_name,
                            tool_id,
                            local_used,
                            local_limit,
                            tool_budget.get("workflow_id"),
                        )
                        local_exceeded_logged[local_key] = True
                    return False, local_used, "local"

            if global_limit is not None:
                used = int(counts.get(tool_id, 0))
                if used >= global_limit:
                    if not exceeded_logged.get(tool_id):
                        self._logger.warning(
                            "tool.budget_exceeded tool_id=%s used=%s limit=%s workflow_id=%s",
                            tool_id,
                            used,
                            global_limit,
                            tool_budget.get("workflow_id"),
                        )
                        exceeded_logged[tool_id] = True
                    return False, used, "global"

            if local_limit is not None and isinstance(local_counts, dict):
                local_counts[tool_id] = int(local_counts.get(tool_id, 0)) + 1
            if global_limit is not None:
                used = int(counts.get(tool_id, 0))
                counts[tool_id] = used + 1
                return True, used + 1, "global"
            return True, int(local_counts.get(tool_id, 0)) if isinstance(local_counts, dict) else 0, "local"

        def _budget_error(used: int, *, scope: str) -> dict[str, Any]:
            if scope == "local":
                return {
                    "ok": False,
                    "error_code": "NODE_TOOL_BUDGET_EXCEEDED",
                    "error_message": (
                        f"tool '{tool_id}' per-visit budget exceeded "
                        f"(used={used}, limit={local_limit}, node={node_name})"
                    ),
                    "tool_id": tool_id,
                    "node_name": node_name,
                    "used": used,
                    "limit": local_limit,
                    "workflow_id": tool_budget.get("workflow_id"),
                }
            return {
                "ok": False,
                "error_code": "TOOL_BUDGET_EXCEEDED",
                "error_message": f"tool '{tool_id}' budget exceeded (used={used}, limit={global_limit})",
                "tool_id": tool_id,
                "used": used,
                "limit": global_limit,
                "workflow_id": tool_budget.get("workflow_id"),
            }

        def _wrapped_func(**kwargs: Any) -> Any:
            allowed, used, scope = _consume_or_reject()
            if not allowed:
                return _budget_error(used, scope=scope)
            return tool.invoke(kwargs)

        async def _wrapped_coro(**kwargs: Any) -> Any:
            allowed, used, scope = _consume_or_reject()
            if not allowed:
                return _budget_error(used, scope=scope)
            return await tool.ainvoke(kwargs)

        wrapped = StructuredTool.from_function(
            func=_wrapped_func,
            coroutine=_wrapped_coro,
            name=tool.name,
            description=tool.description or f"Budgeted wrapper for {tool.name}",
            return_direct=bool(getattr(tool, "return_direct", False)),
            args_schema=getattr(tool, "args_schema", None),
            infer_schema=False,
            response_format=getattr(tool, "response_format", "content"),
        )
        wrapped.tags = list(getattr(tool, "tags", []) or [])
        wrapped.metadata = dict(getattr(tool, "metadata", {}) or {})
        return wrapped
