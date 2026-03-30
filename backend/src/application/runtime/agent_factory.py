from __future__ import annotations

from enum import Enum
from typing import Any, Callable, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.application.runtime.spec_models import AgentSpec

ToolResolver = Callable[[str], Optional[BaseTool]]


class AgentMode(str, Enum):
    CHAIN = "chain"
    REACT = "react"


def create_subagent(
    mode: AgentMode,
    llm: BaseLanguageModel,
    *,
    prompt: str | BasePromptTemplate,
    tools: Optional[List[BaseTool]] = None,
    name: str = "agent",
) -> Runnable | Any:
    if mode == AgentMode.CHAIN:
        if isinstance(prompt, str):
            prompt = PromptTemplate.from_template(prompt)
        return prompt | llm

    if mode == AgentMode.REACT:
        system_content = prompt if isinstance(prompt, str) else prompt.template
        return create_react_agent(
            model=llm,
            tools=list(tools or []),
            prompt=SystemMessage(content=system_content),
            name=name,
        )

    raise ValueError(f"Unknown AgentMode: {mode}")


def build_agent_from_spec(
    spec: AgentSpec,
    llm: BaseLanguageModel,
    tool_resolver: ToolResolver,
) -> Runnable | Any:
    tools: List[BaseTool] = []
    for tool_id in spec.tools:
        try:
            tool = tool_resolver(tool_id)
        except Exception as exc:
            raise ValueError(f"Failed to resolve tool: {tool_id}") from exc
        if tool is None:
            raise ValueError(f"Tool resolver returned None for: {tool_id}")
        tools.append(tool)

    if spec.mode == "chain":
        mode = AgentMode.CHAIN
    elif spec.mode == "react":
        mode = AgentMode.REACT
    else:
        raise ValueError(f"Unknown agent mode: {spec.mode}")

    return create_subagent(
        mode,
        llm,
        prompt=spec.system_prompt,
        tools=tools,
        name=spec.id,
    )
