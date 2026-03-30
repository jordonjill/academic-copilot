import asyncio

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

    def _fake_run_turn(state, requested_workflow_id=None, step_callback=None):
        state["context"]["messages"].append(AIMessage(content="runtime answer"))
        state["output"]["final_text"] = "runtime answer"
        return {"success": True, "type": "chat", "message": "runtime answer"}

    fake_llm = object()
    monkeypatch.setattr(app.runtime, "run_turn", _fake_run_turn)
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
