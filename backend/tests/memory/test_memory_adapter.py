import json
import sqlite3

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict

from src.infrastructure.memory.adapter import MemoryAdapter
from src.infrastructure.memory import sqlite_store
from src.infrastructure.memory import stm as stm_module
from src.infrastructure.memory.sqlite_store import SQLiteStore


def _prepare_db(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter.db"
    monkeypatch.setattr(sqlite_store, "CONVERSATION_DB", str(db_path))
    return db_path


def test_load_context_restores_latest_snapshot(monkeypatch, tmp_path):
    _prepare_db(tmp_path, monkeypatch)
    store = SQLiteStore()
    store.upsert_session("s-1", "u-1", "topic")

    snapshot_messages = [
        SystemMessage(content="[Compressed Context — 2026-03-30]\nprior summary"),
        AIMessage(content="previous answer"),
    ]
    store.save_working_context_snapshot(
        "s-1",
        json.dumps([message_to_dict(m) for m in snapshot_messages], ensure_ascii=False),
        token_count=12,
        is_compressed=True,
    )

    adapter = MemoryAdapter(store=store)
    restored, summary = adapter.load_context("s-1")

    assert len(restored) == 2
    assert restored[0].content.startswith("[Compressed Context")
    assert restored[1].content == "previous answer"
    assert summary == "prior summary"


def test_persist_turn_updates_state_after_compression(monkeypatch, tmp_path):
    _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 1)
    monkeypatch.setattr(stm_module, "STM_KEEP_RECENT", 1)

    adapter = MemoryAdapter(store=SQLiteStore())
    state = {
        "input": {
            "session_id": "s-2",
            "user_id": "u-2",
            "user_text": "topic",
        },
        "context": {
            "messages": [
                HumanMessage(content="old context"),
                AIMessage(content="assistant context"),
            ],
            "memory_summary": "",
        },
        "artifacts": {"topic": "topic"},
    }

    result = adapter.persist_turn(state, FakeListChatModel(responses=["fresh summary"]))

    assert result["stm_compressed"] is True
    assert state["context"]["messages"][0].content.startswith("[Compressed Context")
    assert state["context"]["memory_summary"] == "fresh summary"


def test_persist_turn_does_not_overwrite_with_empty_compressed_messages(monkeypatch, tmp_path):
    _prepare_db(tmp_path, monkeypatch)

    def _fake_stm(state, llm):
        del state, llm
        return {"stm_compressed": True, "messages": []}

    monkeypatch.setattr("src.infrastructure.memory.adapter.stm_compression_node", _fake_stm)

    adapter = MemoryAdapter(store=SQLiteStore())
    original_messages = [
        HumanMessage(content="old context"),
        AIMessage(content="assistant context"),
    ]
    state = {
        "input": {
            "session_id": "s-3",
            "user_id": "u-3",
            "user_text": "topic",
        },
        "context": {
            "messages": list(original_messages),
            "memory_summary": "",
        },
        "artifacts": {"topic": "topic"},
    }

    result = adapter.persist_turn(state, FakeListChatModel(responses=["unused"]))

    assert result["stm_compressed"] is True
    assert state["context"]["messages"] == original_messages


def test_persist_turn_fallback_persists_raw_snapshot_on_stm_failure(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)

    def _boom(state, llm):
        del state, llm
        raise RuntimeError("stm failed")

    monkeypatch.setattr("src.infrastructure.memory.adapter.stm_compression_node", _boom)

    adapter = MemoryAdapter(store=SQLiteStore())
    state = {
        "input": {
            "session_id": "s-4",
            "user_id": "u-4",
            "user_text": "topic",
        },
        "context": {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="world"),
            ],
            "memory_summary": "",
        },
        "artifacts": {"topic": "topic"},
    }

    result = adapter.persist_turn(state, FakeListChatModel(responses=["unused"]))
    assert result["memory_pipeline_degraded"] is True

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM working_context WHERE session_id='s-4'").fetchone()
        assert rows[0] >= 1
    finally:
        conn.close()
