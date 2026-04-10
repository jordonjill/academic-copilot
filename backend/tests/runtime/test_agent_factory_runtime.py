from __future__ import annotations

from src.application.runtime import agent_factory as agent_runtime
from src.application.runtime.agent_factory import AgentMode


def test_create_subagent_react_uses_langchain_create_agent(monkeypatch):
    captured = {}

    def fake_create_agent(model, tools=None, *, system_prompt=None, name=None, **kwargs):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["name"] = name
        captured["kwargs"] = kwargs
        return "react_agent"

    monkeypatch.setattr(agent_runtime, "create_agent", fake_create_agent)

    llm = object()
    tools = [object()]
    result = agent_runtime.create_subagent(
        AgentMode.REACT,
        llm,
        prompt="system prompt",
        tools=tools,
        name="search_scholar",
    )

    assert result == "react_agent"
    assert captured["model"] is llm
    assert captured["tools"] == tools
    assert captured["system_prompt"] == "system prompt"
    assert captured["name"] == "search_scholar"
    assert captured["kwargs"] == {}
