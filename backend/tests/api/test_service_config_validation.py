from __future__ import annotations

from types import SimpleNamespace

from src.application.runtime.spec_models import AgentSpec, WorkflowSpec
from src.interfaces.api import service


class _FakeToolManager:
    def __init__(self, tool_ids: set[str]) -> None:
        self._tools = {tool_id: object() for tool_id in tool_ids}

    def get_tool(self, tool_id: str):
        return self._tools.get(tool_id)


class _FakeRegistry:
    def __init__(self, agents, workflows, llms=None) -> None:
        self.llms = llms or {"openai_default": object()}
        self.agents = agents
        self.workflows = workflows
        self._version = 0

    def reload(self):
        self._version += 1
        return {
            "config_version": self._version,
            "loaded_llms": sorted(self.llms.keys()),
            "loaded_agents": sorted(self.agents.keys()),
            "loaded_workflows": sorted(self.workflows.keys()),
            "failed_objects": [],
        }


def test_validate_runtime_bindings_detects_unknown_tool(monkeypatch):
    agent = AgentSpec.model_validate(
        {
            "id": "researcher",
            "name": "Researcher",
            "mode": "react",
            "system_prompt": "x",
            "tools": ["missing_tool"],
            "llm": {"name": "openai_default"},
        }
    )
    fake_registry = SimpleNamespace(
        llms={"openai_default": object()},
        agents={"researcher": agent},
        workflows={},
    )

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager({"web_search"}))

    errors = service.validate_runtime_bindings()
    assert errors
    assert "Unresolvable tool_id: missing_tool" in errors[0]["error"]


def test_validate_runtime_bindings_detects_unknown_workflow_agent(monkeypatch):
    workflow = WorkflowSpec.model_validate(
        {
            "id": "wf1",
            "name": "wf1",
            "entry_node": "n1",
            "nodes": {
                "n1": {"type": "agent", "agent_id": "ghost"},
                "end": {"type": "terminal"},
            },
            "edges": [{"from": "n1", "to": "end"}],
            "limits": {"max_steps": 3},
        }
    )
    fake_registry = SimpleNamespace(llms={}, agents={}, workflows={"wf1": workflow})

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager(set()))

    errors = service.validate_runtime_bindings()
    assert errors
    assert errors[0]["path"] == "workflow:wf1.n1"
    assert "Unknown agent_id: ghost" in errors[0]["error"]


def test_reload_runtime_config_includes_binding_errors(monkeypatch):
    agent = AgentSpec.model_validate(
        {
            "id": "writer",
            "name": "Writer",
            "mode": "react",
            "system_prompt": "x",
            "tools": ["missing_tool"],
            "llm": {"name": "openai_default"},
        }
    )
    fake_registry = _FakeRegistry(agents={"writer": agent}, workflows={})

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager({"web_search"}))

    report = service.reload_runtime_config()
    assert report["config_version"] == 1
    assert report["failed"]
    assert "Unresolvable tool_id: missing_tool" in report["failed"][0]["error"]


def test_validate_runtime_bindings_rejects_chain_mode_with_tools(monkeypatch):
    agent = SimpleNamespace(
        mode="chain",
        tools=["missing_tool"],
        llm=SimpleNamespace(name="openai_default"),
    )
    fake_registry = SimpleNamespace(
        llms={"openai_default": object()},
        agents={"writer": agent},
        workflows={},
    )

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager(set()))

    errors = service.validate_runtime_bindings()
    assert errors
    assert errors[0]["path"] == "agent:writer"
    assert "chain mode does not support tools" in errors[0]["error"]


def test_validate_runtime_bindings_detects_unknown_llm_name(monkeypatch):
    agent = AgentSpec.model_validate(
        {
            "id": "writer",
            "name": "Writer",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "missing_llm"},
        }
    )
    fake_registry = SimpleNamespace(
        llms={"openai_default": object()},
        agents={"writer": agent},
        workflows={},
    )

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager(set()))

    errors = service.validate_runtime_bindings()
    assert errors
    assert errors[0]["path"] == "agent:writer"
    assert "Unknown llm.name: missing_llm" in errors[0]["error"]
