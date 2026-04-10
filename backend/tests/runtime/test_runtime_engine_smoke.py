from __future__ import annotations

import asyncio
import json
import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.contracts.spec_models import AgentSpec, WorkflowSpec


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
        "io": {"last_model_output": None, "last_execution_output": None, "last_tool_outputs": []},
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


def test_runtime_engine_supervisor_direct_reply_done_false_continues(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    decisions = iter(
        [
            {"action": "direct_reply", "message": "interim draft", "done": False},
            {"action": "run_subagent", "target": "researcher", "instruction": "refine", "done": False},
            {"action": "direct_reply", "message": "final answer", "done": True},
        ]
    )
    researcher_calls = {"count": 0}

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "researcher":
            def _invoke(payload):
                del payload
                researcher_calls["count"] += 1
                return "research output"

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)
    assert result["success"] is True
    assert result["message"] == "final answer"
    assert researcher_calls["count"] == 1
    assert state["runtime"]["step_count"] == 1


def test_runtime_engine_invoke_async_fallbacks_to_sync_invoke_on_model_dump_attr_error(tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))

    class _Runnable:
        async def ainvoke(self, payload, config=None):
            del payload, config
            raise AttributeError("'str' object has no attribute 'model_dump'")

        def invoke(self, payload, config=None):
            del config
            return {"ok": True, "payload": payload}

    result = asyncio.run(engine._invoke_async(_Runnable(), {"x": 1}))
    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert result.get("payload") == {"x": 1}


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
            {
                "action": "run_subagent",
                "target": "researcher",
                "instruction": "collect three references",
                "done": False,
            },
            {"action": "direct_reply", "message": "done after subagent", "done": True},
        ]
    )
    researcher_calls = {"count": 0}
    researcher_payloads: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "researcher":
            def _invoke(payload):
                researcher_payloads.append(payload)
                researcher_calls["count"] += 1
                return "research output"

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)
    assert result["success"] is True
    assert result["message"] == "done after subagent"
    assert researcher_calls["count"] == 1
    assert state["runtime"]["step_count"] == 1
    assert len(researcher_payloads) == 1
    payload = researcher_payloads[0]
    assert payload["user_text"] == "collect three references"
    assert "[TASK_INPUT_V1]" in payload["messages"]
    assert '"protocol": "task_input_v1"' in payload["messages"]
    assert '"instruction": "collect three references"' in payload["messages"]
    assert "collect three references" in payload["messages"]
    assert "hello" not in payload["messages"]
    artifacts = json.loads(payload["artifacts"])
    assert artifacts["supervisor_instruction"] == "collect three references"
    assert any(
        isinstance(message, HumanMessage)
        and "Supervisor task for researcher: collect three references" in message.content
        for message in state["context"]["messages"]
    )


def test_runtime_engine_supervisor_run_agent_done_true_requires_followup_direct_reply(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    supervisor_calls = {"count": 0}
    researcher_calls = {"count": 0}

    def _supervisor_reply(payload):
        del payload
        supervisor_calls["count"] += 1
        if supervisor_calls["count"] == 1:
            return json.dumps(
                {
                    "action": "run_agent",
                    "target": "researcher",
                    "instruction": "write the document",
                    "done": True,
                }
            )
        return json.dumps({"action": "direct_reply", "message": "supervisor final reply", "done": True})

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(_supervisor_reply)
        if spec.id == "researcher":
            def _invoke(payload):
                del payload
                researcher_calls["count"] += 1
                return json.dumps({"final_text": "subagent draft"})

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)

    assert result["success"] is True
    assert result["message"] == "supervisor final reply"
    assert supervisor_calls["count"] == 2
    assert researcher_calls["count"] == 1
    assert state["output"]["final_text"] == "supervisor final reply"


def test_runtime_engine_supervisor_inline_input_artifacts_fill_empty_only(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    decisions = iter(
        [
            {
                "action": "run_agent",
                "target": "researcher",
                "instruction": "export current draft",
                "input_artifact_keys": ["draft"],
                "inline_input_artifacts": {
                    "draft": "INLINE_DRAFT_BODY",
                    "report_title": "Inline Title",
                },
                "done": False,
            },
            {"action": "direct_reply", "message": "ok", "done": True},
        ]
    )
    researcher_payloads: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "researcher":
            def _invoke(payload):
                researcher_payloads.append(payload)
                return "research output"

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected agent execution: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    state["artifacts"]["draft"] = "ORIGINAL_DRAFT_BODY"
    result = engine.run_turn(state)

    assert result["success"] is True
    assert result["message"] == "ok"
    assert len(researcher_payloads) == 1

    payload = researcher_payloads[0]
    artifacts = json.loads(payload["artifacts"])
    assert artifacts["draft"] == "ORIGINAL_DRAFT_BODY"
    assert artifacts["report_title"] == "Inline Title"


def test_runtime_engine_supervisor_starts_workflow(monkeypatch, tmp_path):
    engine = RuntimeEngine(registry=_registry(tmp_path))
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())
    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.load_ltm_profile_for_supervisor",
        lambda user_id: '{"profile":"ok"}',
    )

    supervisor_calls = {"count": 0}
    reporter_calls = {"count": 0}

    supervisor_payloads: list[dict] = []

    def _supervisor_reply(payload):
        supervisor_payloads.append(payload)
        supervisor_calls["count"] += 1
        if supervisor_calls["count"] == 1:
            return json.dumps({"action": "start_workflow", "target": "wf1", "done": True})
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
    assert state["runtime"]["mode"] == "dynamic"
    assert state["runtime"]["workflow_id"] is None
    assert state["runtime"]["step_count"] >= 1
    assert len(supervisor_payloads) == 2
    followup_payload = supervisor_payloads[-1]
    assert followup_payload["workflow_completed"] is False
    assert followup_payload["requested_workflow_id"] == ""
    assert "available_agents" in followup_payload
    assert "available_workflows" in followup_payload
    assert followup_payload["ltm_profile"] == '{"profile":"ok"}'
    assert "step_count" in followup_payload
    assert "loop_count" in followup_payload


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
    with pytest.raises(RuntimeError, match="max_steps exceeded"):
        engine.run_turn(state, requested_workflow_id="loop_wf")
    assert state["runtime"]["loop_count"] >= 1


def test_runtime_engine_workflow_node_cap_falls_forward(monkeypatch, tmp_path):
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
        "searcher": AgentSpec.model_validate(
            {
                "id": "searcher",
                "name": "Searcher",
                "mode": "chain",
                "system_prompt": "search",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
        "reader": AgentSpec.model_validate(
            {
                "id": "reader",
                "name": "Reader",
                "mode": "chain",
                "system_prompt": "read",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "cap_wf": WorkflowSpec.model_validate(
            {
                "id": "cap_wf",
                "name": "cap_wf",
                "entry_node": "search",
                "nodes": {
                    "search": {"type": "agent", "agent_id": "searcher"},
                    "read": {"type": "agent", "agent_id": "reader"},
                    "end": {"type": "terminal"},
                },
                "edges": [
                    {"from": "search", "to": "search"},
                    {"from": "search", "to": "read"},
                    {"from": "read", "to": "end"},
                ],
                "limits": {"max_steps": 10, "max_loops": 6, "max_search": 2},
            }
        )
    }
    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    call_counts = {"searcher": 0, "reader": 0}

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "searcher":
            def _invoke(payload):
                del payload
                call_counts["searcher"] += 1
                return "search loop"

            return _FakeRunnable(_invoke)
        if spec.id == "reader":
            def _invoke(payload):
                del payload
                call_counts["reader"] += 1
                return "reader done"

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected spec id: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    result = engine.run_turn(_state(), requested_workflow_id="cap_wf")
    assert result["success"] is True
    assert result["message"] == "reader done"
    assert call_counts["searcher"] == 2
    assert call_counts["reader"] == 1


def test_runtime_engine_workflow_react_payload_uses_task_input_protocol(monkeypatch, tmp_path):
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
        "searcher": AgentSpec.model_validate(
            {
                "id": "searcher",
                "name": "Searcher",
                "mode": "react",
                "system_prompt": "search",
                "tools": [],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "wf_react_protocol": WorkflowSpec.model_validate(
            {
                "id": "wf_react_protocol",
                "name": "wf_react_protocol",
                "entry_node": "search_node",
                "nodes": {
                    "search_node": {"type": "agent", "agent_id": "searcher"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "search_node", "to": "end"}],
                "limits": {"max_steps": 5, "max_loops": 2},
            }
        )
    }
    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    payloads: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm, tool_resolver
        if spec.id == "searcher":
            def _invoke(payload):
                payloads.append(payload)
                return {"messages": payload["messages"] + [AIMessage(content='{"final_text":"ok"}')]}

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected spec id: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    state["artifacts"]["paper_pool"] = [{"title": "Paper A"}]
    result = engine.run_turn(state, requested_workflow_id="wf_react_protocol")
    assert result["success"] is True
    assert payloads
    first_messages = payloads[0]["messages"]
    assert first_messages
    first_text = str(first_messages[0].content)
    assert "[TASK_INPUT_V1]" in first_text
    assert '"protocol": "task_input_v1"' in first_text
    assert '"node_name": "search_node"' in first_text
    assert '"paper_pool"' in first_text


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


def test_workflow_tool_budget_enforces_max_tool_limit(monkeypatch, tmp_path):
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
        "searcher": AgentSpec.model_validate(
            {
                "id": "searcher",
                "name": "Searcher",
                "mode": "react",
                "system_prompt": "search",
                "tools": ["scholar_search"],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "wf_budget": WorkflowSpec.model_validate(
            {
                "id": "wf_budget",
                "name": "wf_budget",
                "entry_node": "search",
                "nodes": {
                    "search": {"type": "agent", "agent_id": "searcher"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "search", "to": "end"}],
                "limits": {"max_steps": 5, "max_loops": 2, "max_tool_scholar_search": 1},
            }
        )
    }

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    @tool
    def scholar_search(query: str, max_results: int = 12, include_web: bool = True) -> dict:
        """Fake scholar search tool for runtime budget testing."""
        del query, max_results, include_web
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.get_tool",
        lambda tool_id: scholar_search if tool_id == "scholar_search" else None,
    )

    tool_results: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm
        if spec.id != "searcher":
            raise AssertionError(f"Unexpected spec id: {spec.id}")

        def _invoke(payload):
            tool_impl = tool_resolver("scholar_search")
            tool_results.append(tool_impl.invoke({"query": "q1", "max_results": 2, "include_web": False}))
            tool_results.append(tool_impl.invoke({"query": "q2", "max_results": 2, "include_web": False}))
            return {"messages": payload["messages"] + [AIMessage(content='{"final_text":"ok"}')]}

        return _FakeRunnable(_invoke)

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state, requested_workflow_id="wf_budget")
    assert result["success"] is True
    assert result["message"] == "ok"

    assert tool_results[0].get("ok") is True
    assert tool_results[1].get("ok") is False
    assert tool_results[1].get("error_code") == "TOOL_BUDGET_EXCEEDED"
    assert state["runtime"]["tool_budget"]["counts"]["scholar_search"] == 1


def test_direct_subagent_turn_tool_budget_enforces_limit(monkeypatch, tmp_path):
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
        "searcher": AgentSpec.model_validate(
            {
                "id": "searcher",
                "name": "Searcher",
                "mode": "react",
                "system_prompt": "search",
                "tools": ["scholar_search"],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {}

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    @tool
    def scholar_search(query: str, max_results: int = 12, include_web: bool = True) -> dict:
        """Fake scholar search tool for direct subagent turn budget testing."""
        del query, max_results, include_web
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.get_tool",
        lambda tool_id: scholar_search if tool_id == "scholar_search" else None,
    )

    decisions = iter(
        [
            {
                "action": "run_subagent",
                "target": "searcher",
                "instruction": "find evidence",
                "done": False,
            },
            {"action": "direct_reply", "message": "done", "done": True},
        ]
    )
    tool_results: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm
        if spec.id == "supervisor":
            return _FakeRunnable(lambda payload: json.dumps(next(decisions)))
        if spec.id == "searcher":
            def _invoke(payload):
                tool_impl = tool_resolver("scholar_search")
                tool_results.append(tool_impl.invoke({"query": "q1"}))
                tool_results.append(tool_impl.invoke({"query": "q2"}))
                tool_results.append(tool_impl.invoke({"query": "q3"}))
                return {"messages": payload["messages"] + [AIMessage(content='{"final_text":"ok"}')]}

            return _FakeRunnable(_invoke)
        raise AssertionError(f"Unexpected spec id: {spec.id}")

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state)
    assert result["success"] is True
    assert result["message"] == "done"

    assert tool_results[0].get("ok") is True
    assert tool_results[1].get("ok") is True
    assert tool_results[2].get("ok") is False
    assert tool_results[2].get("error_code") == "TOOL_BUDGET_EXCEEDED"
    assert state["runtime"]["tool_budget"]["scope"] == "turn"
    assert state["runtime"]["tool_budget"]["counts"]["scholar_search"] == 2


def test_workflow_tool_budget_enforces_per_visit_node_limit(monkeypatch, tmp_path):
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
        "searcher": AgentSpec.model_validate(
            {
                "id": "searcher",
                "name": "Searcher",
                "mode": "react",
                "system_prompt": "search",
                "tools": ["scholar_search"],
                "llm": {"name": "openai_default"},
            }
        ),
    }
    registry.workflows = {
        "wf_node_budget": WorkflowSpec.model_validate(
            {
                "id": "wf_node_budget",
                "name": "wf_node_budget",
                "entry_node": "search",
                "nodes": {
                    "search": {"type": "agent", "agent_id": "searcher"},
                    "end": {"type": "terminal"},
                },
                "edges": [{"from": "search", "to": "end"}],
                "limits": {
                    "max_steps": 5,
                    "max_loops": 2,
                    "max_node_tool_search__scholar_search": 1,
                },
            }
        )
    }

    engine = RuntimeEngine(registry=registry)
    monkeypatch.setattr(engine, "_resolve_llm", lambda spec: object())

    @tool
    def scholar_search(query: str, max_results: int = 12, include_web: bool = True) -> dict:
        """Fake scholar search tool for per-visit budget testing."""
        del query, max_results, include_web
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr(
        "src.application.runtime.runtime_engine.get_tool",
        lambda tool_id: scholar_search if tool_id == "scholar_search" else None,
    )

    tool_results: list[dict] = []

    def _fake_build(spec, llm, tool_resolver):
        del llm
        if spec.id != "searcher":
            raise AssertionError(f"Unexpected spec id: {spec.id}")

        def _invoke(payload):
            tool_impl = tool_resolver("scholar_search")
            tool_results.append(tool_impl.invoke({"query": "q1"}))
            tool_results.append(tool_impl.invoke({"query": "q2"}))
            return {"messages": payload["messages"] + [AIMessage(content='{"final_text":"ok"}')]}

        return _FakeRunnable(_invoke)

    monkeypatch.setattr("src.application.runtime.runtime_engine.build_agent_from_spec", _fake_build)

    state = _state()
    result = engine.run_turn(state, requested_workflow_id="wf_node_budget")
    assert result["success"] is True
    assert result["message"] == "ok"

    assert tool_results[0].get("ok") is True
    assert tool_results[1].get("ok") is False
    assert tool_results[1].get("error_code") == "NODE_TOOL_BUDGET_EXCEEDED"
    assert state["runtime"]["tool_budget"]["counts"]["scholar_search"] == 1
