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
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.infrastructure.memory import MemoryAdapter
from src.infrastructure.tools.tool_manager import get_tool_manager
from src.infrastructure.tools.loader import reload_tools

load_dotenv()

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"
_CONFIG_REGISTRY = ConfigRegistry(config_root=_CONFIG_ROOT)
logger = logging.getLogger(__name__)


def get_config_registry() -> ConfigRegistry:
    return _CONFIG_REGISTRY


def reload_runtime_config() -> Dict[str, Any]:
    report = _CONFIG_REGISTRY.reload()
    validation_errors = validate_runtime_bindings()
    failed = list(report["failed_objects"]) + validation_errors
    return {
        "config_version": report["config_version"],
        "loaded": {
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
    enabled_tool_ids = get_tool_manager().get_catalog_tool_ids(enabled_only=True)

    for agent_id, spec in _CONFIG_REGISTRY.agents.items():
        for tool_id in spec.tools:
            if tool_id in enabled_tool_ids:
                continue
            errors.append(
                {
                    "type": "binding",
                    "path": f"agent:{agent_id}",
                    "error": f"Unknown or disabled tool_id: {tool_id}",
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
        self._last_state: Optional[Dict[str, Any]] = None

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
        history_messages, memory_summary = self.memory.load_context(sid)
        state = self._build_state(
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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.runtime.run_turn(
                state,
                requested_workflow_id=workflow_id,
                step_callback=_capture_step,
            ),
        )

        try:
            llm = self.runtime.resolve_default_llm()
            await loop.run_in_executor(
                None,
                lambda: self.memory.persist_turn(state, llm),
            )
        except Exception as exc:
            logger.warning("Memory pipeline failed (non-fatal): %s", exc)

        self._last_state = state

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

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        return self._last_state

    def health_check(self) -> Dict[str, Any]:
        try:
            probe = self.runtime.health_probe()
            return {
                "status": "healthy",
                "message": "Runtime configuration loaded.",
                "probe": probe,
            }
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

    @staticmethod
    def _build_state(
        user_message: str,
        user_id: str,
        session_id: str,
        workflow_id: Optional[str] = None,
        history_messages: Optional[list[BaseMessage]] = None,
        memory_summary: str = "",
    ) -> Dict[str, Any]:
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
    return AcademicCopilotApp(model_type=model_type)


def _ts() -> str:
    return datetime.now().isoformat()
