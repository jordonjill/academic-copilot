from __future__ import annotations

from typing import Any, Callable, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from src.application.agents import AgentMode, create_subagent
from src.application.runtime.spec_models import AgentSpec

ToolResolver = Callable[[str], Optional[BaseTool]]


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
