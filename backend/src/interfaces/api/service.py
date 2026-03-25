"""
Academic Copilot 服务层。

职责：
  - 初始化 LLM（ollama / gemini / openai）
  - 调用 src/graph.py 的主图
  - 提供 chat_async() 统一对话接口
  - 提供 run_research_async() 向后兼容接口（WebSocket 流式）
  - health_check()

不包含任何 HTTP/WebSocket 逻辑，与框架无关。
"""
from __future__ import annotations
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain.chat_models import init_chat_model

from src.application.graph import build_main_graph
from src.application.runtime.config_registry import ConfigRegistry

load_dotenv()

_MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "ollama":  {"provider": "ollama",       "model": "llama3.1:8b"},
    "gemini":  {"provider": "google_genai", "model": "gemini-2.5-flash"},
    "openai":  {"provider": "openai",       "model": "gpt-4o"},
}

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"
_CONFIG_REGISTRY = ConfigRegistry(config_root=_CONFIG_ROOT)


def get_config_registry() -> ConfigRegistry:
    return _CONFIG_REGISTRY


def reload_runtime_config() -> Dict[str, Any]:
    report = _CONFIG_REGISTRY.reload()
    return {
        "config_version": report["config_version"],
        "loaded": {
            "agents": report["loaded_agents"],
            "workflows": report["loaded_workflows"],
        },
        "failed": report["failed_objects"],
    }


def _make_llm(model_type: str, temperature: float = 0):
    cfg = _MODEL_CONFIGS.get(model_type)
    if cfg is None:
        raise ValueError(f"Unsupported model_type: {model_type!r}. "
                         f"Choose from: {list(_MODEL_CONFIGS)}")
    if cfg["provider"] == "ollama":
        return ChatOllama(model=cfg["model"], temperature=temperature)
    return init_chat_model(model=cfg["model"], model_provider=cfg["provider"],
                           temperature=temperature)


class AcademicCopilotApp:
    """Academic Copilot 应用实例，封装 LLM + LangGraph 主图。"""

    def __init__(self, model_type: str = "ollama", temperature: float = 0) -> None:
        self.model_type = model_type
        self.llm = _make_llm(model_type, temperature)
        self.graph = build_main_graph(self.llm)
        self._last_state: Optional[Dict] = None

    # ── 对话接口 ──────────────────────────────────────────────────────────────

    async def chat_async(
        self,
        user_message: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        websocket_send: Optional[Callable] = None,
        recursion_limit: int = 25,
    ) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        inputs = self._build_state(user_message, user_id, sid, workflow_id)

        if websocket_send:
            await websocket_send({"type": "status",
                                  "message": f"Processing: {user_message[:80]}...",
                                  "session_id": sid, "timestamp": _ts()})

        final_step: Optional[Dict] = None
        step_count = 0

        async for step in self._stream(inputs, {"recursion_limit": recursion_limit}):
            step_count += 1
            node_name, output = next(iter(step.items()))
            final_step = step
            self._last_state = output

            if websocket_send:
                await websocket_send(self._build_step_event(node_name, output, step_count))
            await asyncio.sleep(0.05)

        result = self._extract_result(final_step)

        if websocket_send:
            await websocket_send({"type": "completion", "final_result": result,
                                  "session_id": sid, "timestamp": _ts()})
        return result

    async def run_research_async(
        self,
        initial_topic: str,
        websocket_send: Optional[Callable] = None,
        recursion_limit: int = 20,
    ) -> Dict[str, Any]:
        """向后兼容接口（WebSocket 流式研究）。"""
        return await self.chat_async(
            user_message=f"帮我写一个关于{initial_topic}的研究提案",
            websocket_send=websocket_send,
            recursion_limit=recursion_limit,
        )

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def get_current_state(self) -> Optional[Dict]:
        return self._last_state

    def health_check(self) -> Dict[str, Any]:
        try:
            self.llm.invoke([HumanMessage(content="ping")])
            return {"status": "healthy", "message": "LLM connection successful"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

    async def _stream(self, inputs: Dict, config: Dict):
        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(None, lambda: self.graph.stream(inputs, config))
        for step in stream:
            yield step

    @staticmethod
    def _build_state(
        user_message: str,
        user_id: str,
        session_id: str,
        workflow_id: Optional[str] = None,
    ) -> Dict:
        orchestration_mode = "workflow" if workflow_id else "dynamic"
        return {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "user_profile": None,
            "current_intent": None,
            "workflow_status": "idle",
            "collected_materials": [],
            "retrieved_resources": [],
            "current_draft_sections": None,
            "final_output": None,
            "initial_topic": None,
            "research_plan": None,
            "research_creation": None,
            "research_critic": None,
            "idea_validation_attempts": 0,
            "search_count": 0,
            "final_proposal": None,
            "session_id": session_id,
            "stm_token_count": 0,
            "stm_compressed": False,
            "ltm_extraction_done": False,
            "pending_workflow_confirmation": False,
            "suggested_workflow_id": workflow_id,
            "orchestration_mode": orchestration_mode,
            "selected_subagents": [],
            "confirmation_expires_at_turn": None,
            "last_selected_agent_id": None,
            "agent_retry_counters": {},
            "clarification_required": False,
        }

    @staticmethod
    def _build_step_event(node_name: str, output: Dict, step_count: int) -> Dict:
        event: Dict[str, Any] = {
            "type": "step", "step_number": step_count,
            "node_name": node_name, "timestamp": _ts(),
        }
        if output.get("current_intent"):
            i = output["current_intent"]
            event["intent"] = {"intent": i.intent, "confidence": i.confidence,
                               "workflow_topic": i.workflow_topic}
        if output.get("research_plan"):
            p = output["research_plan"]
            event["plan"] = {"step_type": p.step_type, "query": p.query}
            # 兼容旧前端字段
            event["details"] = {
                "action_type": p.step_type,
                "query": p.query,
                "has_enough_content": p.has_enough_content,
            }
        if "retrieved_resources" in output:
            event["resource_count"] = len(output["retrieved_resources"])
        if output.get("research_creation"):
            event["research_gap"] = output["research_creation"].research_gap
        if output.get("research_critic"):
            c = output["research_critic"]
            event["critic"] = {"is_valid": c.is_valid, "feedback": c.feedback}
            # 兼容旧前端字段
            event["critic_result"] = {"is_valid": c.is_valid, "feedback": c.feedback}
        return event

    @staticmethod
    def _extract_result(final_step: Optional[Dict]) -> Dict[str, Any]:
        if not final_step:
            return {"success": False, "message": "No output produced."}
        output = next(iter(final_step.values()))

        if fp := output.get("final_proposal"):
            return {"success": True, "type": "proposal", "proposal": {
                "title": fp.Title, "introduction": fp.Introduction,
                "research_problem": fp.ResearchProblem, "methodology": fp.Methodology,
                "expected_outcomes": fp.ExpectedOutcomes, "references": fp.References,
            }}

        if (fo := output.get("final_output")) and fo.get("type") == "survey":
            return {"success": True, **fo}

        msgs = output.get("messages", [])
        last_ai = next((m.content for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        if last_ai:
            return {"success": True, "type": "chat", "message": last_ai}

        return {"success": False, "message": "Process finished without recognizable output."}


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def create_copilot(model_type: str = "ollama") -> AcademicCopilotApp:
    return AcademicCopilotApp(model_type=model_type)


def create_research_agent(model_type: str = "ollama") -> AcademicCopilotApp:
    """向后兼容别名。"""
    return create_copilot(model_type)


# 向后兼容类型别名
ResearchAgentApp = AcademicCopilotApp


def _ts() -> str:
    return datetime.now().isoformat()
