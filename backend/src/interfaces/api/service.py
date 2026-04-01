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
import json
import logging
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
_COPILOT_APP: Optional["AcademicCopilotApp"] = None
_DEFAULT_CHAT_TURN_TIMEOUT_SECONDS = 120.0
_MAX_LAST_STATES = 128
_TIMEOUT_RELATION_WARNED = False


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
        recursion_limit: int = 25,
    ) -> Dict[str, Any]:
        del recursion_limit  # 简化 runtime 不依赖递归配置

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

        loop = asyncio.get_event_loop()
        timeout_seconds = _chat_turn_timeout_seconds()
        _warn_timeout_misconfiguration(timeout_seconds)
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.runtime.run_turn(
                        state,
                        requested_workflow_id=workflow_id,
                        step_callback=_capture_step,
                    ),
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
            raise asyncio.TimeoutError(
                f"Chat turn timed out after {int(timeout_seconds)} seconds"
            )

        try:
            llm = self.runtime.resolve_default_llm()
            await loop.run_in_executor(
                None,
                lambda: self.memory.persist_turn(state, llm),
            )
        except Exception as exc:
            logger.exception("Memory pipeline failed (non-fatal): %s", exc)

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
    global _COPILOT_APP
    with _APP_LOCK:
        if _COPILOT_APP is None or _COPILOT_APP.model_type != model_type:
            _COPILOT_APP = AcademicCopilotApp(model_type=model_type)
        return _COPILOT_APP


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
    payload = {"event": event, "timestamp": _ts(), **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def _chat_turn_timeout_seconds() -> float:
    return read_env_float("CHAT_TURN_TIMEOUT_SECONDS", _DEFAULT_CHAT_TURN_TIMEOUT_SECONDS)


def _warn_timeout_misconfiguration(chat_timeout: float) -> None:
    global _TIMEOUT_RELATION_WARNED
    if _TIMEOUT_RELATION_WARNED:
        return
    supervisor_timeout = read_env_float("SUPERVISOR_MAX_WALL_TIME_SECONDS", 180.0)
    workflow_timeout = read_env_float("WORKFLOW_MAX_WALL_TIME_SECONDS", 300.0)
    min_runtime_timeout = min(supervisor_timeout, workflow_timeout)
    if chat_timeout >= min_runtime_timeout:
        logger.warning(
            "Timeout configuration may be suboptimal: CHAT_TURN_TIMEOUT_SECONDS(%.1f) "
            ">= min(SUPERVISOR_MAX_WALL_TIME_SECONDS=%.1f, WORKFLOW_MAX_WALL_TIME_SECONDS=%.1f). "
            "Consider setting chat timeout smaller than runtime loop timeouts.",
            chat_timeout,
            supervisor_timeout,
            workflow_timeout,
        )
    _TIMEOUT_RELATION_WARNED = True


def warn_timeout_misconfiguration_once() -> None:
    _warn_timeout_misconfiguration(_chat_turn_timeout_seconds())
