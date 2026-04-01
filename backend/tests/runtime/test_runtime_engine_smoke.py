from __future__ import annotations

import asyncio
import json
import pytest

from langchain_core.messages import HumanMessage

from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


class _FakeRunnable:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, payload):
        return self._fn(payload)


def _state() -> dict:
    return {
        "input": {"user_text": "hello", "user_id": "u1", "session_id": "s1"},
        "context": {"messages": [HumanMessage(content="hello")], "memory_summary": ""},
        "runtime": {
            "mode": "dynamic",
            "workflow_id": None,
            "current_node": None,
            "step_count": 0,
            "loop_count": 0,
            "status": "idle",
        },
        "io": {"last_model_output": None, "last_tool_outputs": []},
        "artifacts": {"topic": None, "shared": {}},
        "output": {"final_text": None, "final_structured": None},
        "errors": {"last_error": None},
    }


def _registry(tmp_path) -> ConfigRegistry:
    reg = ConfigRegistry(config_root=tmp_path)
    reg.agents = {
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
        "reporter": AgentSpec.model_validate(
            {
                "id": "reporter",
                "name": "Reporter",
                "mode": "chain",
                "system_prompt": "reporter",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    reg.workflows = {
        "wf1": WorkflowSpec.model_validate(
            {
                "id": "wf1",
                "name": "wf1",
                "entry_node": "reporter_node",
                "nodes": {
                    "reporter_node": {"type": "agent", "agent_id": "reporter"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "reporter_node", "to": "end"}],
                "limits": {"max_steps": 5, "max_loops": 2},
            }
        )
    }
    return reg


def test_runtime_engine_supervisor_direct_reply(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(
                lambda payload: json.dumps({"action": "direct_reply", "message": "direct ok", "done": True})
            )
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    result = engine.run_turn(_state())
    assert result["success"] is True
    assert result["message"] == "direct ok"


def test_runtime_engine_logs_warning_when_supervisor_returns_empty(monkeypatch, tmp_path, caplog):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: "")
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    with caplog.at_level("WARNING"):
        result = engine.run_turn(_state())

    assert result["success"] is True
    assert result["message"] == "No output produced."
    assert any("supervisor.empty_decision_output" in record.message for record in caplog.records)


def test_runtime_engine_supervisor_runs_subagent_then_replies(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    decisions = iter(
        [
            {"action": "run_subagent", "target": "researcher", "done": False},
            {"action": "direct_reply", "message": "done after subagent", "done": True},
        ]
    )
    researcher_calls = {"count": 0}

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "researcher":
            return _FakeRunnable(lambda payload: researcher_calls.__setitem__("count", researcher_calls["count"] + 1) or "research output")
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)
    assert result["success"] is True
    assert result["message"] == "done after subagent"
    assert researcher_calls["count"] == 1
    assert state["runtime"]["step_count"] == 1


def test_runtime_engine_supervisor_starts_workflow(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    supervisor_calls = {"count": 0}
    reporter_calls = {"count": 0}

    supervisor_payloads: list[dict] = []

    def _supervisor_reply(payload):
        supervisor_payloads.append(payload)
        supervisor_calls["count"] += 1
        if supervisor_calls["count"] == 1:
            return json.dumps({"action": "start_workflow", "target": "wf1", "done": False})
        return json.dumps({"action": "direct_reply", "message": "workflow finished", "done": True})

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(_supervisor_reply)
        if spec.id == "reporter":
            def _invoke(payload):
                del payload
                reporter_calls["count"] += 1
                return json.dumps({"final_text": "reporter output"})

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)
    assert result["success"] is True
    assert result["message"] == "workflow finished"
    assert reporter_calls["count"] == 1
    assert state["runtime"]["mode"] == "workflow"
    assert state["runtime"]["workflow_id"] == "wf1"
    assert len(supervisor_payloads) == 2
    finalize_payload = supervisor_payloads[-1]
    assert finalize_payload["workflow_completed"] is True
    assert "available_agents" in finalize_payload
    assert "available_workflows" in finalize_payload
    assert "step_count" in finalize_payload
    assert "loop_count" in finalize_payload


def test_runtime_engine_workflow_loop_counts_entry_revisit(monkeypatch, tmp_path):
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
    registry.workflows = {
        "loop_wf": WorkflowSpec.model_validate(
            {
                "id": "loop_wf",
                "name": "loop_wf",
                "entry_node": "researcher_node",
                "nodes": {"researcher_node": {"type": "agent", "agent_id": "researcher"}},
                "edges": [{"from": "researcher_node", "to": "researcher_node"}],
                "limits": {"max_steps": 10, "max_loops": 1},
            }
        )
    }

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "researcher":
            return _FakeRunnable(lambda payload: "looping")
        raise AssertionError(f"Unexpected spec id: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    with pytest.raises(RuntimeError, match="max_loops exceeded"):
        engine.run_turn(state, requested_workflow_id="loop_wf")
    assert state["runtime"]["loop_count"] == 1


def test_apply_agent_output_recovers_when_shared_is_non_dict(tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    state = _state()
    state["artifacts"]["shared"] = None

    engine._apply_agent_output(
        state=state,
        node_name="researcher_node",
        agent_id="researcher",
        text="hello",
        parsed={"artifacts": {"shared": None}},
    )

    assert isinstance(state["artifacts"]["shared"], dict)
    assert state["artifacts"]["shared"]["researcher"]["output_text"] == "hello"


def test_requested_workflow_skips_finalize_when_supervisor_not_chain(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.agents = {
        "supervisor": AgentSpec.model_validate(
            {
                "id": "supervisor",
                "name": "Supervisor",
                "mode": "react",
                "system_prompt": "supervisor",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
        "reporter": AgentSpec.model_validate(
            {
                "id": "reporter",
                "name": "Reporter",
                "mode": "chain",
                "system_prompt": "reporter",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "wf1": WorkflowSpec.model_validate(
            {
                "id": "wf1",
                "name": "wf1",
                "entry_node": "reporter_node",
                "nodes": {
                    "reporter_node": {"type": "agent", "agent_id": "reporter"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "reporter_node", "to": "end"}],
                "limits": {"max_steps": 5, "max_loops": 2},
            }
        )
    }
    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "reporter":
            return _FakeRunnable(lambda payload: json.dumps({"final_text": "reporter output"}))
        raise AssertionError(f"Unexpected finalize execution for spec: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    result = engine.run_turn(_state(), requested_workflow_id="wf1")
    assert result["success"] is True
    assert result["message"] == "reporter output"


def test_run_turn_async_awaits_async_step_callback(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.agents = {
        "supervisor": AgentSpec.model_validate(
            {
                "id": "supervisor",
                "name": "Supervisor",
                "mode": "react",
                "system_prompt": "supervisor",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
        "reporter": AgentSpec.model_validate(
            {
                "id": "reporter",
                "name": "Reporter",
                "mode": "chain",
                "system_prompt": "reporter",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "wf1": WorkflowSpec.model_validate(
            {
                "id": "wf1",
                "name": "wf1",
                "entry_node": "reporter_node",
                "nodes": {
                    "reporter_node": {"type": "agent", "agent_id": "reporter"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "reporter_node", "to": "end"}],
                "limits": {"max_steps": 5, "max_loops": 2},
            }
        )
    }
    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())
    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.build_agent_from_spec",
        lambda spec, llm, tool_resolver: _FakeRunnable(
            lambda payload: json.dumps({"final_text": "reporter output"})
        ),
    )

    callback_calls = {"count": 0}

    async def _step_callback(payload):
        del payload
        callback_calls["count"] += 1

    result = asyncio.run(engine.run_turn_async(_state(), requested_workflow_id="wf1", step_callback=_step_callback))
    assert result["success"] is True
    assert callback_calls["count"] == 1
