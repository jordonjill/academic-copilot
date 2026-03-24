from __future__ import annotations
from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, SystemMessage
from src.state import GlobalState


_CHITCHAT_SYSTEM = """You are Academic Copilot, an intelligent assistant specializing in academic research.
You help researchers write proposals, surveys, and navigate academic literature.
Be helpful, concise, and professional. Respond in the same language as the user.

If you received a CLARIFY_NEEDED intent, ask the clarification question clearly.
If you received a CHITCHAT intent, respond naturally and helpfully."""


def chitchat_node(state: GlobalState, llm: BaseLanguageModel) -> Dict:
    """直接对话节点：处理闲聊、系统介绍或澄清问题。"""
    intent_obj = state.get("current_intent")

    # CLARIFY_NEEDED 时直接返回澄清问题（无需 LLM）
    if intent_obj and intent_obj.intent == "CLARIFY_NEEDED" and intent_obj.clarification_question:
        reply = intent_obj.clarification_question
        return {
            "messages": [AIMessage(content=reply)],
            "workflow_status": "idle",
        }

    # 普通 CHITCHAT：注入系统 prompt 后调用 LLM
    messages = state.get("messages", [])
    chat_messages = [SystemMessage(content=_CHITCHAT_SYSTEM)] + list(messages)
    response = llm.invoke(chat_messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "workflow_status": "idle",
    }
