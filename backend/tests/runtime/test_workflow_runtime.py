import pytest

from src.application.runtime.spec_models import AgentSpec
from src.application.agents import AgentMode


def test_build_chain_agent_from_spec_resolves_tools(monkeypatch):
    try:
        from src.application.runtime import agent_runtime
    except ModuleNotFoundError:
        pytest.fail("agent_runtime module missing")

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
        return f"tool:{tool_id}"

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
    assert captured["tools"] == ["tool:web_search", "tool:arxiv"]
    assert captured["name"] == "planner"
    assert captured["output_schema"] is None
