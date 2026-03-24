"""
Sub-Agent 双模式工厂。

两种模式：
  AgentMode.CHAIN  — prompt | llm[.with_structured_output(schema)]
                     适用于：Planner / Synthesizer / Reporter / Supervisor
                     返回：LangChain Runnable（直接 .invoke(input_dict) 使用）

  AgentMode.REACT  — create_react_agent(llm, tools, system_prompt)
                     适用于：Researcher / Critic（需要工具调用循环）
                     返回：CompiledGraph（可直接作为 LangGraph 节点）

统一入口：
  create_subagent(mode, llm, prompt=..., tools=..., output_schema=..., name=...)

角色工厂函数（预置系统 prompt + ToolGroup）：
  build_researcher(llm, tools?, user_profile_summary?) → CompiledGraph
  build_critic(llm, tools?)                            → CompiledGraph
  build_writer(llm, tools?)                            → CompiledGraph
  build_planner(llm)                                   → Runnable → ResearchPlan
  build_synthesizer(llm)                               → Runnable → ResearchCreation
  build_reporter(llm)                                  → Runnable → FinalProposal
  build_supervisor(llm)                                → Runnable → IntentClassification
"""
from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.infrastructure.config.prompt import (
    PLANNER_PROMPT,
    SYNTHESIZER_PROMPT,
    REPORTER_PROMPT,
    SUPERVISOR_PROMPT,
    CRITIC_QUERY_GENERATION_PROMPT,
    CRITIC_EVALUATION_PROMPT,
)
from src.domain.state import ResearchPlan, ResearchCreation, FinalProposal, IntentClassification


# ─────────────────────────────────────────────────────────────────────────────
# AgentMode 枚举
# ─────────────────────────────────────────────────────────────────────────────

class AgentMode(str, Enum):
    CHAIN = "chain"   # 单步推理：prompt → LLM → [结构化输出]
    REACT = "react"   # 工具调用循环：ReAct agent


# ─────────────────────────────────────────────────────────────────────────────
# 统一工厂入口
# ─────────────────────────────────────────────────────────────────────────────

def create_subagent(
    mode: AgentMode,
    llm: BaseLanguageModel,
    *,
    prompt: Union[str, BasePromptTemplate],
    tools: Optional[List[BaseTool]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    name: str = "agent",
) -> Union[Runnable, Any]:
    """
    统一 Sub-Agent 工厂。

    CHAIN 模式返回 LangChain Runnable，调用方式：
        chain.invoke({"key": value, ...})

    REACT 模式返回 CompiledGraph（create_react_agent 产物），调用方式：
        graph.invoke({"messages": [...], ...})
        或直接作为 LangGraph add_node 的节点。
    """
    if mode == AgentMode.CHAIN:
        if isinstance(prompt, str):
            prompt = PromptTemplate.from_template(prompt)
        chain_llm = llm.with_structured_output(output_schema) if output_schema else llm
        return prompt | chain_llm

    elif mode == AgentMode.REACT:
        system_content = prompt if isinstance(prompt, str) else prompt.template
        return create_react_agent(
            model=llm,
            tools=list(tools or []),
            prompt=SystemMessage(content=system_content),
            name=name,
        )

    else:
        raise ValueError(f"Unknown AgentMode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# 角色系统 Prompt（内置行为准则）
# ─────────────────────────────────────────────────────────────────────────────

_RESEARCHER_SYSTEM = """You are a rigorous academic researcher specializing in systematic literature review.

## Behavior Guidelines
1. **Decompose** every research topic into 3-5 targeted search queries before searching.
2. **Prioritize** peer-reviewed sources (journals, conferences, arXiv). Prefer papers from 2020+.
3. **Identify gaps**: After each search, explicitly note what is still missing or unexplored.
4. **Filter quality**: Skip results with no clear methodology or evaluation.
5. **User context**: {user_profile_summary}

When you have gathered sufficient coverage, summarize the research landscape concisely."""

_CRITIC_SYSTEM = """You are a skeptical academic reviewer. Your job is to challenge research ideas rigorously.

## Skeptic's Mindset
1. **Assume prior art exists** — search aggressively for overlapping work.
2. **Attack the weakest link** — novelty claim weak? Search for prior art. Feasibility doubtful? Find failure cases.
3. **Be constructive** — when invalidating, always suggest a concrete alternative direction.
4. Mark INVALID ONLY when: completely non-novel (identical prior work found) OR fundamental barrier proven.
5. **Default to VALID** for partially novel or moderately challenged ideas."""

_WRITER_SYSTEM = """You are an expert academic writer for research proposals and literature surveys.

## Writing Standards
1. Third-person academic voice. No first-person pronouns.
2. Outline all sections before writing.
3. Numbered citations [1], [2], … in order of first appearance.
4. Confirm each feedback point before revising; mark all changes.
5. Proposals: 800–1500 words. Survey sections: 500–800 words each."""


# ─────────────────────────────────────────────────────────────────────────────
# 角色工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_researcher(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
    user_profile_summary: str = "",
) -> Any:
    """
    REACT 模式：多轮搜索 + 研究空白识别。
    tools 默认从 ToolGroup.RESEARCH 获取。
    """
    if tools is None:
        from src.infrastructure.tools.registry import get_tools, ToolGroup
        tools = get_tools(ToolGroup.RESEARCH)
    prompt = _RESEARCHER_SYSTEM.format(
        user_profile_summary=user_profile_summary or "No prior user profile."
    )
    return create_subagent(AgentMode.REACT, llm, prompt=prompt, tools=tools, name="researcher")


def build_critic(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    REACT 模式：反例搜索 + 构建性反馈。
    tools 默认从 ToolGroup.CRITIQUE 获取。
    """
    if tools is None:
        from src.infrastructure.tools.registry import get_tools, ToolGroup
        tools = get_tools(ToolGroup.CRITIQUE)
    return create_subagent(AgentMode.REACT, llm, prompt=_CRITIC_SYSTEM, tools=tools, name="critic")


def build_writer(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    REACT 模式：学术写作 + 文件读写（可选 filesystem 工具）。
    tools 默认为空列表（纯写作时不需要搜索）。
    """
    if tools is None:
        from src.infrastructure.tools.registry import get_tools, ToolGroup
        tools = get_tools(ToolGroup.FILESYSTEM)  # 运行时注入的 MCP 文件工具
    return create_subagent(AgentMode.REACT, llm, prompt=_WRITER_SYSTEM, tools=tools, name="writer")


def build_planner(llm: BaseLanguageModel) -> Runnable:
    """CHAIN 模式：分析资源 → 输出 ResearchPlan（search / synthesize）。"""
    return create_subagent(
        AgentMode.CHAIN, llm,
        prompt=PLANNER_PROMPT,
        output_schema=ResearchPlan,
    )


def build_synthesizer(llm: BaseLanguageModel) -> Runnable:
    """CHAIN 模式：综合文献 → 识别 Gap → 输出 ResearchCreation。"""
    return create_subagent(
        AgentMode.CHAIN, llm,
        prompt=SYNTHESIZER_PROMPT,
        output_schema=ResearchCreation,
    )


def build_reporter(llm: BaseLanguageModel) -> Runnable:
    """CHAIN 模式：生成完整研究提案 → 输出 FinalProposal。"""
    return create_subagent(
        AgentMode.CHAIN, llm,
        prompt=REPORTER_PROMPT,
        output_schema=FinalProposal,
    )


def build_supervisor(llm: BaseLanguageModel) -> Runnable:
    """CHAIN 模式：意图识别 → 输出 IntentClassification。"""
    return create_subagent(
        AgentMode.CHAIN, llm,
        prompt=SUPERVISOR_PROMPT,
        output_schema=IntentClassification,
    )
