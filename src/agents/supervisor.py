from __future__ import annotations
import os
from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from src.state import GlobalState, UserProfile, IntentClassification
from src.config.prompt import SUPERVISOR_PROMPT


def _load_user_profile(user_id: str) -> UserProfile:
    """从 data/users/{user_id}/memory.md 加载 UserProfile。"""
    profile_path = os.path.join("data", "users", user_id, "memory.md")
    if not os.path.exists(profile_path):
        return UserProfile(user_id=user_id)

    with open(profile_path, "r", encoding="utf-8") as f:
        raw_md = f.read()

    profile = UserProfile(user_id=user_id, raw_memory_md=raw_md)

    # 简单解析 markdown 节段
    import re
    sections = {
        "## Research Domains": "research_domains",
        "## Preferred Methodologies": "preferred_methodologies",
        "## Known Tools": "known_tools",
        "## Past Topics": "past_topics",
        "## Custom Facts": "custom_facts",
    }
    for header, field in sections.items():
        pattern = rf"{re.escape(header)}\n((?:- .+\n?)*)"
        m = re.search(pattern, raw_md)
        if m:
            items = [line.lstrip("- ").strip() for line in m.group(1).strip().splitlines()]
            setattr(profile, field, [i for i in items if i])

    return profile


def _build_profile_summary(profile: UserProfile) -> str:
    if not profile.raw_memory_md:
        return "No prior user profile available."
    lines = []
    if profile.research_domains:
        lines.append(f"Research Domains: {', '.join(profile.research_domains)}")
    if profile.preferred_methodologies:
        lines.append(f"Methodologies: {', '.join(profile.preferred_methodologies)}")
    if profile.past_topics:
        lines.append(f"Past Topics: {', '.join(profile.past_topics[-5:])}")
    return "\n".join(lines) if lines else "No specific profile data."


def _build_recent_conversation(state: GlobalState) -> str:
    messages = state.get("messages", [])[-10:]  # 最近 10 条（约 5 轮）
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"Assistant: {content[:300]}")
    return "\n".join(parts) if parts else "No prior conversation."


def supervisor_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """Supervisor 节点：加载 UserProfile + 分类意图 + 路由决策。"""
    user_id = state.get("user_id", "default")

    # 1. 首次加载 UserProfile
    user_profile = state.get("user_profile")
    if user_profile is None:
        user_profile = _load_user_profile(user_id)

    # 2. 取最新 HumanMessage
    messages = state.get("messages", [])
    latest_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        ""
    )

    # 3. LLM 结构化输出 IntentClassification（使用工厂 CHAIN 模式）
    from src.agents.factory import build_supervisor
    intent: IntentClassification = build_supervisor(llm).invoke({
        "user_profile_summary": _build_profile_summary(user_profile),
        "recent_conversation": _build_recent_conversation(state),
        "latest_message": latest_human,
    })

    return {
        "user_profile": user_profile,
        "current_intent": intent,
        "workflow_status": "running",
    }


def route_by_intent(state: GlobalState) -> str:
    """条件路由函数：根据 IntentClassification 决定下一节点。"""
    intent_obj = state.get("current_intent")
    if intent_obj is None:
        return "chitchat"
    mapping = {
        "CHITCHAT": "chitchat",
        "PROPOSAL_GEN": "proposal_workflow",
        "SURVEY_WRITE": "survey_workflow",
        "CLARIFY_NEEDED": "chitchat",
    }
    return mapping.get(intent_obj.intent, "chitchat")
