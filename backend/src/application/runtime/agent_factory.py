from __future__ import annotations

from enum import Enum
from typing import Any, Callable, List, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from src.application.runtime.contracts.spec_models import AgentSpec

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
    output_schema: Any = None,
    name: str = "agent",
) -> Runnable | Any:
    if mode == AgentMode.CHAIN:
        if isinstance(prompt, str):
            prompt = PromptTemplate.from_template(prompt)
        if output_schema is not None and hasattr(llm, "with_structured_output"):
            llm = llm.with_structured_output(output_schema)
        return prompt | llm

    if mode == AgentMode.REACT:
        system_content = prompt if isinstance(prompt, str) else prompt.template
        return create_agent(
            model=llm,
            tools=list(tools or []),
            system_prompt=system_content,
            name=name,
        )

    raise ValueError(f"Unknown AgentMode: {mode}")


def build_agent_from_spec(
    spec: AgentSpec,
    llm: BaseLanguageModel,
    tool_resolver: ToolResolver,
) -> Runnable | Any:
    if spec.mode == "chain":
        mode = AgentMode.CHAIN
    elif spec.mode == "react":
        mode = AgentMode.REACT
    else:
        raise ValueError(f"Unknown agent mode: {spec.mode}")

    tools: List[BaseTool] = []
    if mode == AgentMode.REACT:
        for tool_id in spec.tools:
            try:
                tool = tool_resolver(tool_id)
            except Exception as exc:
                raise ValueError(f"Failed to resolve tool: {tool_id}") from exc
            if tool is None:
                # Runtime may hide budget-exhausted tools from the current execution.
                continue
            tools.append(tool)

    return create_subagent(
        mode,
        llm,
        prompt=spec.system_prompt,
        tools=tools,
        output_schema=None,
        name=spec.id,
    )
