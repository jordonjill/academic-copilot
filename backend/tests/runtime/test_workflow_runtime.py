import pytest
from langchain_core.tools import BaseTool

from src.application.agents import AgentMode
from src.application.runtime import agent_runtime
from src.application.runtime.spec_models import AgentSpec


class DummyTool(BaseTool):
    name: str
    description: str = "dummy tool"

    def _run(self, *args, **kwargs):
        return "ok"

    async def _arun(self, *args, **kwargs):
        return "ok"


def test_build_chain_agent_from_spec_resolves_tools(monkeypatch):
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["web_search", "arxiv"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    resolved = []

    def tool_resolver(tool_id):
        resolved.append(tool_id)
        return DummyTool(name=f"tool:{tool_id}")

    captured = {}

    def fake_create_subagent(mode, llm, *, prompt, tools=None, output_schema=None, name="agent"):
        captured["mode"] = mode
        captured["llm"] = llm
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["name"] = name
        captured["output_schema"] = output_schema
        return "agent_instance"

    monkeypatch.setattr(agent_runtime, "create_subagent", fake_create_subagent)

    llm = object()
    result = agent_runtime.build_agent_from_spec(spec, llm, tool_resolver)

    assert result == "agent_instance"
    assert resolved == ["web_search", "arxiv"]
    assert captured["mode"] == AgentMode.CHAIN
    assert captured["llm"] is llm
    assert captured["prompt"] == "You plan"
    assert [tool.name for tool in captured["tools"]] == ["tool:web_search", "tool:arxiv"]
    assert captured["name"] == "planner"
    assert captured["output_schema"] is None


def test_build_react_agent_from_spec_wires_mode(monkeypatch):
    spec = AgentSpec(
        id="researcher",
        name="Researcher Agent",
        mode="react",
        system_prompt="You research",
        tools=["web_search"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        return DummyTool(name=f"tool:{tool_id}")

    captured = {}

    def fake_create_subagent(mode, llm, *, prompt, tools=None, output_schema=None, name="agent"):
        captured["mode"] = mode
        captured["llm"] = llm
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["name"] = name
        captured["output_schema"] = output_schema
        return "react_agent"

    monkeypatch.setattr(agent_runtime, "create_subagent", fake_create_subagent)

    llm = object()
    result = agent_runtime.build_agent_from_spec(spec, llm, tool_resolver)

    assert result == "react_agent"
    assert captured["mode"] == AgentMode.REACT
    assert captured["llm"] is llm
    assert captured["prompt"] == "You research"
    assert [tool.name for tool in captured["tools"]] == ["tool:web_search"]
    assert captured["name"] == "researcher"
    assert captured["output_schema"] is None


def test_build_agent_from_spec_raises_on_resolver_error():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["web_search"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        raise RuntimeError("boom")

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "web_search" in str(exc.value)


def test_build_agent_from_spec_raises_on_missing_tool():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["arxiv"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        return None

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "arxiv" in str(exc.value)
