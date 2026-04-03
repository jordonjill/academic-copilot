from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.contracts.spec_models import AgentSpec


class _FakeRunnable:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, payload):
        return self._fn(payload)


def _state() -> dict:
    return {
        "input": {
            "user_text": "please research",
            "user_id": "u1",
            "session_id": "s1",
        },
        "context": {
            "messages": [HumanMessage(content="please research")],
            "memory_summary": "",
        },
        "runtime": {
            "mode": "dynamic",
            "workflow_id": None,
            "current_node": None,
            "step_count": 0,
            "loop_count": 0,
            "status": "idle",
        },
        "io": {
            "last_model_output": None,
            "last_execution_output": None,
            "last_tool_outputs": [],
        },
        "artifacts": {
            "topic": None,
            "shared": {},
        },
        "output": {
            "final_text": None,
            "final_structured": None,
        },
        "errors": {
            "last_error": None,
        },
    }


def test_supervisor_subagent_call_cap_per_agent(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.agents = {
        "supervisor": AgentSpec.model_validate(
            {
                "id": "supervisor",
                "name": "Supervisor",
                "mode": "chain",
                "system_prompt": "supervisor",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
        "researcher": AgentSpec.model_validate(
            {
                "id": "researcher",
                "name": "Researcher",
                "mode": "chain",
                "system_prompt": "researcher",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {}

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    decisions = iter(
        [
            {"action": "run_subagent", "target": "researcher", "done": False},
            {"action": "run_subagent", "target": "researcher", "done": False},
            {"action": "run_subagent", "target": "researcher", "done": False},
            {"action": "direct_reply", "message": "final answer", "done": True},
        ]
    )
    researcher_calls = {"count": 0}

    def _fake_build_agent_from_spec(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "researcher":
            def _invoke(payload):
                del payload
                researcher_calls["count"] += 1
                return "researcher output"

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected spec id: {spec.id}")

    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.build_agent_from_spec",
        _fake_build_agent_from_spec,
    )
    monkeypatch.setenv("SUPERVISOR_MAX_STEPS", "8")
    monkeypatch.setenv("SUPERVISOR_MAX_SUBAGENT_CALLS_PER_AGENT", "2")

    state = _state()
    result = engine.run_turn(state)

    assert result["success"] is True
    assert result["message"] == "final answer"
    assert researcher_calls["count"] == 2
    assert state["runtime"]["step_count"] == 2
    assert any(
        "call limit reached (2)" in str(message.content)
        for message in state["context"]["messages"]
    )


def test_supervisor_max_steps_env_invalid_falls_back(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.agents = {
        "supervisor": AgentSpec.model_validate(
            {
                "id": "supervisor",
                "name": "Supervisor",
                "mode": "chain",
                "system_prompt": "supervisor",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {}

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())
    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.build_agent_from_spec",
        lambda spec, llm, tool_resolver: _FakeRunnable(
            lambda payload: json.dumps({"action": "direct_reply", "message": "ok", "done": True})
        ),
    )
    monkeypatch.setenv("SUPERVISOR_MAX_STEPS", "8.0")

    result = engine.run_turn(_state())
    assert result["success"] is True
    assert result["message"] == "ok"
