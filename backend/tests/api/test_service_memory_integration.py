import asyncio
import contextlib

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from src.interfaces.api.service import AcademicCopilotApp


def test_chat_async_uses_memory_pipeline(monkeypatch):
    calls: dict = {"load_context_args": []}

    class _FakeMemory:
        def load_context(self, session_id: str):
            calls["load_context_args"].append(session_id)
            return (
                [
                    SystemMessage(content="[Compressed Context — 2026-03-30]\nprevious summary"),
                    AIMessage(content="previous reply"),
                ],
                "previous summary",
            )

        def persist_turn(self, state, llm):
            calls["persist_called"] = True
            calls["persist_llm"] = llm
            calls["persist_messages"] = list(state["context"]["messages"])
            return {"stm_compressed": False}

    fake_memory = _FakeMemory()
    monkeypatch.setattr("src.interfaces.api.service.MemoryAdapter", lambda: fake_memory)

    app = AcademicCopilotApp()

    async def _fake_run_turn_async(state, requested_workflow_id=None, step_callback=None):
        state["context"]["messages"].append(AIMessage(content="runtime answer"))
        state["output"]["final_text"] = "runtime answer"
        return {"success": True, "type": "chat", "message": "runtime answer"}

    fake_llm = object()
    monkeypatch.setattr(app.runtime, "run_turn_async", _fake_run_turn_async)
    monkeypatch.setattr(app.runtime, "resolve_default_llm", lambda: fake_llm)

    result = asyncio.run(
        app.chat_async(
            user_message="current question",
            user_id="u-test",
            session_id="s-test",
        )
    )

    assert result["success"] is True
    assert calls["load_context_args"] == ["s-test"]
    assert calls["persist_called"] is True
    assert calls["persist_llm"] is fake_llm

    persisted_messages = calls["persist_messages"]
    assert persisted_messages[0].content.startswith("[Compressed Context")
    assert persisted_messages[2].content == "current question"
    assert persisted_messages[-1].content == "runtime answer"
    assert app.get_current_state("s-test") is not None


def test_chat_async_keeps_last_states_per_session(monkeypatch):
    class _FakeMemory:
        def load_context(self, session_id: str):
            del session_id
            return ([], "")

        def persist_turn(self, state, llm):
            del state, llm
            return {"stm_compressed": False}

    monkeypatch.setattr("src.interfaces.api.service.MemoryAdapter", lambda: _FakeMemory())
    app = AcademicCopilotApp()

    async def _fake_run_turn_async(state, requested_workflow_id=None, step_callback=None):
        del requested_workflow_id, step_callback
        state["output"]["final_text"] = state["input"]["user_text"]
        return {"success": True, "type": "chat", "message": state["input"]["user_text"]}

    monkeypatch.setattr(app.runtime, "run_turn_async", _fake_run_turn_async)
    monkeypatch.setattr(app.runtime, "resolve_default_llm", lambda: object())

    asyncio.run(app.chat_async(user_message="first", session_id="s-1"))
    asyncio.run(app.chat_async(user_message="second", session_id="s-2"))

    state_1 = app.get_current_state("s-1")
    state_2 = app.get_current_state("s-2")
    assert state_1 is not None
    assert state_2 is not None
    assert state_1["input"]["user_text"] == "first"
    assert state_2["input"]["user_text"] == "second"


def test_chat_async_attaches_token_usage_to_returned_runtime(monkeypatch):
    token_usage = {
        "calls": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    captured: dict = {}

    class _FakeMemory:
        def load_context(self, session_id: str):
            del session_id
            return ([], "")

        def persist_turn(self, state, llm):
            del state, llm
            return {"stm_compressed": False}

    class _FakeObservation:
        def token_usage(self):
            return token_usage

        def update_output(self, output):
            captured["langfuse_output"] = output

        def update_error(self, error):
            captured["error"] = error

    @contextlib.contextmanager
    def _fake_observe_chat_turn(**kwargs):
        captured["observe_kwargs"] = kwargs
        yield _FakeObservation()

    monkeypatch.setattr("src.interfaces.api.service.MemoryAdapter", lambda: _FakeMemory())
    monkeypatch.setattr("src.interfaces.api.service.observe_chat_turn", _fake_observe_chat_turn)
    app = AcademicCopilotApp()

    async def _fake_run_turn_async(state, requested_workflow_id=None, step_callback=None):
        del requested_workflow_id, step_callback
        state["output"]["final_text"] = "ok"
        return {
            "success": True,
            "type": "chat",
            "message": "ok",
            "data": {
                "runtime": {
                    "mode": "dynamic",
                    "workflow_id": None,
                    "step_count": 0,
                    "loop_count": 0,
                    "status": "completed",
                },
                "outputs": {},
            },
        }

    monkeypatch.setattr(app.runtime, "run_turn_async", _fake_run_turn_async)
    monkeypatch.setattr(app.runtime, "resolve_default_llm", lambda: object())

    result = asyncio.run(app.chat_async(user_message="current question", session_id="s-token"))

    assert result["data"]["runtime"]["token_usage"] == token_usage
    assert captured["langfuse_output"]["data"]["runtime"]["token_usage"] == token_usage


def test_build_step_event_prefers_workflow_runtime_from_step_payload():
    state = {
        "runtime": {
            "mode": "dynamic",
            "workflow_id": None,
            "current_node": None,
            "max_steps": 8,
            "loop_count": 0,
            "status": "running",
        }
    }

    event = AcademicCopilotApp._build_step_event(
        {
            "step_number": 1,
            "node_name": "reporter_node",
            "agent_id": "reporter",
            "next_node": "end",
            "mode": "workflow",
            "workflow_id": "wf1",
            "current_node": "end",
            "max_steps": 5,
            "max_loops": 2,
            "loop_count": 0,
            "status": "running",
        },
        state,
    )

    assert event["workflow_id"] == "wf1"
    assert event["mode"] == "workflow"
    assert event["current_node"] == "end"
    assert event["max_steps"] == 5
    assert event["max_loops"] == 2


def test_chat_async_persists_memory_on_runtime_failure(monkeypatch):
    calls: dict = {"persist_called": False}

    class _FakeMemory:
        def load_context(self, session_id: str):
            del session_id
            return ([], "")

        def persist_turn(self, state, llm):
            del state, llm
            calls["persist_called"] = True
            return {"stm_compressed": False}

    monkeypatch.setattr("src.interfaces.api.service.MemoryAdapter", lambda: _FakeMemory())
    app = AcademicCopilotApp()

    async def _failing_run_turn_async(state, requested_workflow_id=None, step_callback=None):
        del state, requested_workflow_id, step_callback
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(app.runtime, "run_turn_async", _failing_run_turn_async)
    monkeypatch.setattr(app.runtime, "resolve_default_llm", lambda: object())

    with pytest.raises(RuntimeError, match="runtime failed"):
        asyncio.run(app.chat_async(user_message="failing turn", session_id="s-fail"))

    assert calls["persist_called"] is True
