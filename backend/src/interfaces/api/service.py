"""
Academic Copilot 服务层。

职责：
  - 初始化/重载配置注册表
  - 执行简化版 YAML Runtime（supervisor + subagent/workflow）
  - 提供 chat_async() 统一对话接口
  - health_check()

不包含任何 HTTP/WebSocket 逻辑，与框架无关。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.env_utils import read_env_float
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.state_types import RuntimeState
from src.infrastructure.memory import MemoryAdapter
from src.infrastructure.memory.sqlite_store import SQLiteStore
from src.infrastructure.tools.tool_manager import get_tool_manager
from src.infrastructure.tools.loader import reload_tools

load_dotenv()

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"
_CONFIG_REGISTRY = ConfigRegistry(config_root=_CONFIG_ROOT)
logger = logging.getLogger(__name__)
_APP_LOCK = threading.Lock()
_COPILOT_BY_MODEL: dict[str, "AcademicCopilotApp"] = {}
_DEFAULT_CHAT_TURN_TIMEOUT_SECONDS = 120.0
_MAX_LAST_STATES = 128
_TIMEOUT_RELATION_WARNED_FOR: tuple[float, float, float, float] | None = None
_TIMEOUT_RELATION_LOCK = threading.Lock()
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{\w+\}")
_LOG_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "access_key",
    "authorization",
    "token",
    "secret",
    "password",
    "cookie",
)
_LOG_SENSITIVE_SUFFIX_PATTERN = re.compile(r"(?:^|[_-])(key|token|secret|password)$", re.I)
_LOG_MAX_STRING_LEN = 256


def get_config_registry() -> ConfigRegistry:
    return _CONFIG_REGISTRY


def reload_runtime_config() -> Dict[str, Any]:
    report = _CONFIG_REGISTRY.reload()
    validation_errors = validate_runtime_bindings()
    failed = list(report["failed_objects"]) + validation_errors
    return {
        "config_version": report["config_version"],
        "loaded": {
            "llms": report.get("loaded_llms", []),
            "agents": report["loaded_agents"],
            "subagents": report.get("loaded_subagents", report["loaded_agents"]),
            "system_agents": report.get("loaded_system_agents", []),
            "workflows": report["loaded_workflows"],
        },
        "failed": failed,
    }


async def reload_tools_config() -> Dict[str, Any]:
    return await reload_tools()


async def reload_all_config() -> Dict[str, Any]:
    tools_report = await reload_tools_config()
    runtime_report = reload_runtime_config()
    return {
        "runtime": runtime_report,
        "tools": tools_report,
    }


def validate_runtime_bindings() -> list[Dict[str, str]]:
    errors: list[Dict[str, str]] = []
    tool_manager = get_tool_manager()
    known_llms = set(getattr(_CONFIG_REGISTRY, "llms", {}).keys())

    for agent_id, spec in _CONFIG_REGISTRY.agents.items():
        llm_name = spec.llm.name
        if isinstance(llm_name, str) and llm_name.strip():
            if llm_name not in known_llms:
                errors.append(
                    {
                        "type": "binding",
                        "path": f"agent:{agent_id}",
                        "error": f"Unknown llm.name: {llm_name}",
                    }
                )
            else:
                llm_spec = _CONFIG_REGISTRY.llms.get(llm_name)
                for field_name in ("model_name", "base_url", "api_key_env"):
                    raw_value = getattr(llm_spec, field_name, None)
                    if _has_unresolved_env_placeholder(raw_value):
                        errors.append(
                            {
                                "type": "binding",
                                "path": f"agent:{agent_id}",
                                "error": (
                                    f"LLM profile '{llm_name}' has unresolved env placeholder "
                                    f"in field '{field_name}': {raw_value}"
                                ),
                            }
                        )
                api_key_env = getattr(llm_spec, "api_key_env", None)
                if isinstance(api_key_env, str) and api_key_env.strip():
                    env_name = api_key_env.strip()
                    env_value = os.getenv(env_name, "").strip()
                    if not env_value:
                        errors.append(
                            {
                                "type": "binding",
                                "path": f"agent:{agent_id}",
                                "error": (
                                    f"LLM profile '{llm_name}' requires missing env var: {env_name}"
                                ),
                            }
                        )

        if spec.mode == "chain" and spec.tools:
            errors.append(
                {
                    "type": "binding",
                    "path": f"agent:{agent_id}",
                    "error": "chain mode does not support tools; set tools to [] or switch mode to react",
                }
            )

        if spec.mode != "react":
            continue
        for tool_id in spec.tools:
            if tool_manager.get_tool(tool_id) is not None:
                continue
            errors.append(
                {
                    "type": "binding",
                    "path": f"agent:{agent_id}",
                    "error": f"Unresolvable tool_id: {tool_id}",
                }
            )

    known_agents = set(_CONFIG_REGISTRY.agents.keys())
    for workflow_id, spec in _CONFIG_REGISTRY.workflows.items():
        for node_name, node in spec.nodes.items():
            if not isinstance(node, dict):
                errors.append(
                    {
                        "type": "binding",
                        "path": f"workflow:{workflow_id}.{node_name}",
                        "error": "Node must be a mapping",
                    }
                )
                continue
            if node.get("type") != "agent":
                continue
            agent_id = node.get("agent_id")
            if not isinstance(agent_id, str) or agent_id not in known_agents:
                errors.append(
                    {
                        "type": "binding",
                        "path": f"workflow:{workflow_id}.{node_name}",
                        "error": f"Unknown agent_id: {agent_id}",
                    }
                )

    return errors


class AcademicCopilotApp:
    """Academic Copilot 应用实例，封装简化 YAML Runtime。"""

    def __init__(self, model_type: str = "ollama", temperature: float = 0) -> None:
        del temperature
        self.model_type = model_type
        self.runtime = RuntimeEngine(registry=_CONFIG_REGISTRY)
        self.memory = MemoryAdapter()
        self._last_states: OrderedDict[str, RuntimeState] = OrderedDict()
        self._last_states_lock = threading.Lock()

    async def chat_async(
        self,
        user_message: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        websocket_send: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        started = perf_counter()
        _log_event(
            "chat.turn.start",
            session_id=sid,
            user_id=user_id,
            workflow_id=workflow_id,
        )
        history_messages, memory_summary = self.memory.load_context(sid)
        state: RuntimeState = self._build_state(
            user_message,
            user_id,
            sid,
            workflow_id,
            history_messages=history_messages,
            memory_summary=memory_summary,
        )

        if websocket_send:
            await websocket_send(
                {
                    "type": "status",
                    "message": f"Processing: {user_message[:80]}...",
                    "session_id": sid,
                    "timestamp": _ts(),
                }
            )

        captured_steps: list[Dict[str, Any]] = []

        def _capture_step(step: Dict[str, Any]) -> None:
            captured_steps.append(step)
            _log_event(
                "chat.turn.step",
                session_id=sid,
                step_number=step.get("step_number"),
                node_name=step.get("node_name"),
                agent_id=step.get("agent_id"),
                next_node=step.get("next_node"),
            )

        timeout_seconds = _chat_turn_timeout_seconds()
        _warn_timeout_misconfiguration(timeout_seconds)

        async def _persist_memory_best_effort() -> None:
            try:
                loop = asyncio.get_running_loop()
                llm = self.runtime.resolve_default_llm()

                supports_event_loop = False
                try:
                    params = inspect.signature(self.memory.persist_turn).parameters
                    supports_event_loop = "event_loop" in params
                except (TypeError, ValueError):
                    supports_event_loop = False

                def _persist_turn() -> dict[str, Any]:
                    if supports_event_loop:
                        return self.memory.persist_turn(state, llm, event_loop=loop)
                    return self.memory.persist_turn(state, llm)

                await asyncio.to_thread(_persist_turn)
            except Exception as exc:
                logger.exception("Memory pipeline failed (non-fatal): %s", exc)

        try:
            result = await asyncio.wait_for(
                self.runtime.run_turn_async(
                    state,
                    requested_workflow_id=workflow_id,
                    step_callback=_capture_step,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            _log_event(
                "chat.turn.timeout",
                session_id=sid,
                user_id=user_id,
                timeout_seconds=timeout_seconds,
            )
            await _persist_memory_best_effort()
            raise asyncio.TimeoutError(f"Chat turn timed out after {int(timeout_seconds)} seconds")
        except Exception:
            await _persist_memory_best_effort()
            raise

        await _persist_memory_best_effort()

        self._remember_state(sid, state)
        _log_event(
            "chat.turn.complete",
            session_id=sid,
            success=bool(result.get("success")),
            result_type=result.get("type"),
            runtime_mode=state.get("runtime", {}).get("mode"),
            step_count=state.get("runtime", {}).get("step_count"),
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )

        if websocket_send:
            for step in captured_steps:
                await websocket_send(self._build_step_event(step))
            await websocket_send(
                {
                    "type": "completion",
                    "final_result": result,
                    "session_id": sid,
                    "timestamp": _ts(),
                }
            )

        return result

    def get_current_state(self, session_id: Optional[str] = None) -> Optional[RuntimeState]:
        with self._last_states_lock:
            if session_id:
                return self._last_states.get(session_id)
            if not self._last_states:
                return None
            _, latest = next(reversed(self._last_states.items()))
            return latest

    def health_check(self) -> Dict[str, Any]:
        try:
            runtime_probe = self.runtime.health_probe()
            memory_probe = _memory_probe()
            config_probe = {
                "config_version": _CONFIG_REGISTRY.config_version,
                "loaded_agents": len(_CONFIG_REGISTRY.agents),
                "loaded_subagents": len(getattr(_CONFIG_REGISTRY, "subagents", {})),
                "loaded_system_agents": len(getattr(_CONFIG_REGISTRY, "system_agents", {})),
                "loaded_workflows": len(_CONFIG_REGISTRY.workflows),
            }
            runtime_ok = bool(runtime_probe.get("ok", False))
            memory_ok = bool(memory_probe.get("ok", False))
            healthy = runtime_ok and memory_ok
            return {
                "status": "healthy" if healthy else "unhealthy",
                "message": "Runtime probes collected.",
                "probe": {
                    "runtime": runtime_probe,
                    "memory": memory_probe,
                    "config": config_probe,
                },
            }
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

    def _remember_state(self, session_id: str, state: RuntimeState) -> None:
        with self._last_states_lock:
            if session_id in self._last_states:
                self._last_states.pop(session_id)
            self._last_states[session_id] = state
            while len(self._last_states) > _MAX_LAST_STATES:
                self._last_states.popitem(last=False)

    @staticmethod
    def _build_state(
        user_message: str,
        user_id: str,
        session_id: str,
        workflow_id: Optional[str] = None,
        history_messages: Optional[list[BaseMessage]] = None,
        memory_summary: str = "",
    ) -> RuntimeState:
        initial_messages = list(history_messages or [])
        initial_messages.append(HumanMessage(content=user_message))
        return {
            "input": {
                "user_text": user_message,
                "user_id": user_id,
                "session_id": session_id,
            },
            "context": {
                "messages": initial_messages,
                "memory_summary": memory_summary,
            },
            "runtime": {
                "mode": "workflow" if workflow_id else "dynamic",
                "workflow_id": workflow_id,
                "current_node": None,
                "step_count": 0,
                "loop_count": 0,
                "status": "idle",
            },
            "io": {
                "last_model_output": None,
                "last_tool_outputs": [],
            },
            "artifacts": {
                "topic": None,
                "shared": {},
                "execution_trace": [],
            },
            "output": {
                "final_text": None,
                "final_structured": None,
            },
            "errors": {
                "last_error": None,
            },
        }

    @staticmethod
    def _build_step_event(step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "step",
            "step_number": step.get("step_number"),
            "node_name": step.get("node_name"),
            "agent_id": step.get("agent_id"),
            "next_node": step.get("next_node"),
            "supervisor_reason": step.get("supervisor_reason"),
            "tool_outputs": step.get("tool_outputs", []),
            "timestamp": _ts(),
        }


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def create_copilot(model_type: str = "ollama") -> AcademicCopilotApp:
    key = (model_type or "ollama").strip() or "ollama"
    with _APP_LOCK:
        existing = _COPILOT_BY_MODEL.get(key)
        if existing is not None:
            return existing
        app = AcademicCopilotApp(model_type=key)
        _COPILOT_BY_MODEL[key] = app
        return app


def _ts() -> str:
    return datetime.now().isoformat()


def _memory_probe() -> Dict[str, Any]:
    try:
        store = SQLiteStore()
        del store
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "timestamp": _ts(), **_sanitize_for_log(fields)}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def _sanitize_for_log(value: Any, *, key_name: str = "") -> Any:
    lower_key = key_name.lower()
    if any(marker in lower_key for marker in _LOG_SENSITIVE_KEY_MARKERS) or _LOG_SENSITIVE_SUFFIX_PATTERN.search(key_name):
        return "***REDACTED***"

    if isinstance(value, dict):
        return {str(k): _sanitize_for_log(v, key_name=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_log(item, key_name=key_name) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_log(item, key_name=key_name) for item in value)
    if isinstance(value, str):
        if value.lower().startswith("bearer "):
            return "Bearer ***REDACTED***"
        if value.startswith("sk-") and len(value) > 12:
            return "***REDACTED***"
        if len(value) > _LOG_MAX_STRING_LEN:
            return value[:_LOG_MAX_STRING_LEN] + "...<truncated>"
    return value


def _chat_turn_timeout_seconds() -> float:
    return read_env_float("CHAT_TURN_TIMEOUT_SECONDS", _DEFAULT_CHAT_TURN_TIMEOUT_SECONDS)


def _llm_request_timeout_seconds() -> float:
    return read_env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0)


def _has_unresolved_env_placeholder(value: Any) -> bool:
    return isinstance(value, str) and _ENV_PLACEHOLDER_PATTERN.search(value) is not None


def _warn_timeout_misconfiguration(chat_timeout: float) -> None:
    llm_timeout = _llm_request_timeout_seconds()
    supervisor_timeout = read_env_float("SUPERVISOR_MAX_WALL_TIME_SECONDS", 180.0)
    workflow_timeout = read_env_float("WORKFLOW_MAX_WALL_TIME_SECONDS", 300.0)
    key = (
        round(llm_timeout, 3),
        round(chat_timeout, 3),
        round(supervisor_timeout, 3),
        round(workflow_timeout, 3),
    )
    global _TIMEOUT_RELATION_WARNED_FOR
    with _TIMEOUT_RELATION_LOCK:
        if _TIMEOUT_RELATION_WARNED_FOR == key:
            return
        _TIMEOUT_RELATION_WARNED_FOR = key

    min_runtime_timeout = min(supervisor_timeout, workflow_timeout)
    if llm_timeout >= chat_timeout or chat_timeout >= min_runtime_timeout:
        logger.warning(
            "Timeout configuration may be suboptimal: require "
            "LLM_REQUEST_TIMEOUT_SECONDS(%.1f) < CHAT_TURN_TIMEOUT_SECONDS(%.1f) < "
            "min(SUPERVISOR_MAX_WALL_TIME_SECONDS=%.1f, WORKFLOW_MAX_WALL_TIME_SECONDS=%.1f).",
            llm_timeout,
            chat_timeout,
            supervisor_timeout,
            workflow_timeout,
        )


def warn_timeout_misconfiguration_once() -> None:
    _warn_timeout_misconfiguration(_chat_turn_timeout_seconds())


def validate_timeout_hierarchy_or_raise() -> None:
    llm_timeout = _llm_request_timeout_seconds()
    chat_timeout = _chat_turn_timeout_seconds()
    supervisor_timeout = read_env_float("SUPERVISOR_MAX_WALL_TIME_SECONDS", 180.0)
    workflow_timeout = read_env_float("WORKFLOW_MAX_WALL_TIME_SECONDS", 300.0)
    min_runtime_timeout = min(supervisor_timeout, workflow_timeout)
    if not (llm_timeout < chat_timeout < min_runtime_timeout):
        raise ValueError(
            "Invalid timeout hierarchy: expected "
            f"LLM_REQUEST_TIMEOUT_SECONDS({llm_timeout}) < "
            f"CHAT_TURN_TIMEOUT_SECONDS({chat_timeout}) < "
            f"min(SUPERVISOR_MAX_WALL_TIME_SECONDS={supervisor_timeout}, "
            f"WORKFLOW_MAX_WALL_TIME_SECONDS={workflow_timeout})"
        )
