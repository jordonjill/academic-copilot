from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager({"scholar_search"}))

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
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager({"scholar_search"}))

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


def test_validate_runtime_bindings_detects_missing_llm_api_key_env(monkeypatch):
    agent = AgentSpec.model_validate(
        {
            "id": "writer",
            "name": "Writer",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "openai_default"},
        }
    )
    llm_spec = SimpleNamespace(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )
    fake_registry = SimpleNamespace(
        llms={"openai_default": llm_spec},
        agents={"writer": agent},
        workflows={},
    )

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager(set()))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    errors = service.validate_runtime_bindings()
    assert any("requires missing env var: OPENAI_API_KEY" in err["error"] for err in errors)


def test_validate_runtime_bindings_detects_unresolved_llm_placeholder(monkeypatch):
    agent = AgentSpec.model_validate(
        {
            "id": "writer",
            "name": "Writer",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "openai_default"},
        }
    )
    llm_spec = SimpleNamespace(
        model_name="gpt-4o-mini",
        base_url="${MISSING_BASE_URL}",
        api_key_env="OPENAI_API_KEY",
    )
    fake_registry = SimpleNamespace(
        llms={"openai_default": llm_spec},
        agents={"writer": agent},
        workflows={},
    )

    monkeypatch.setattr(service, "_CONFIG_REGISTRY", fake_registry)
    monkeypatch.setattr(service, "get_tool_manager", lambda: _FakeToolManager(set()))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    errors = service.validate_runtime_bindings()
    assert any("unresolved env placeholder" in err["error"] for err in errors)


def test_validate_timeout_hierarchy_or_raise_raises_for_invalid_order(monkeypatch):
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "80")
    monkeypatch.setenv("CHAT_TURN_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SUPERVISOR_MAX_WALL_TIME_SECONDS", "180")
    monkeypatch.setenv("WORKFLOW_MAX_WALL_TIME_SECONDS", "300")

    with pytest.raises(ValueError, match="Invalid timeout hierarchy"):
        service.validate_timeout_hierarchy_or_raise()


def test_sanitize_for_log_redacts_sensitive_values():
    payload = {
        "access_key": "abc",
        "Authorization": "Bearer secret-token",
        "nested": {"api_key": "sk-abcdef1234567890"},
        "normal": "ok",
    }
    sanitized = service._sanitize_for_log(payload)
    assert sanitized["access_key"] == "***REDACTED***"
    assert sanitized["Authorization"] == "***REDACTED***"
    assert sanitized["nested"]["api_key"] == "***REDACTED***"
    assert sanitized["normal"] == "ok"
