from __future__ import annotations

from typing import Callable, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.application.agents import AgentMode, create_subagent
from src.application.runtime.spec_models import AgentSpec

ToolResolver = Callable[[str], BaseTool]


def build_agent_from_spec(
    spec: AgentSpec,
    llm: BaseLanguageModel,
    tool_resolver: ToolResolver,
):
    tools: List[BaseTool] = [tool_resolver(tool_id) for tool_id in spec.tools]

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
