from __future__ import annotations

import sqlite3

import pytest

from src.infrastructure.memory import sqlite_store
from src.infrastructure.memory.sqlite_store import SQLiteStore


def _prepare_db(tmp_path, monkeypatch):
    db_path = tmp_path / "sqlite_store.db"
    monkeypatch.setattr(sqlite_store, "CONVERSATION_DB", str(db_path))
    return db_path


def test_update_session_status_rejects_invalid_value(tmp_path, monkeypatch):
    _prepare_db(tmp_path, monkeypatch)
    store = SQLiteStore()
    store.upsert_session("s-1", "u-1", "topic")

    with pytest.raises(ValueError, match="Invalid session status"):
        store.update_session_status("s-1", "unknown")


def test_update_session_status_accepts_valid_values(tmp_path, monkeypatch):
    db_path = _prepare_db(tmp_path, monkeypatch)
    store = SQLiteStore()
    store.upsert_session("s-2", "u-2", "topic")
    store.update_session_status("s-2", "closed")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT status FROM sessions WHERE session_id='s-2'").fetchone()
        assert row is not None
        assert row[0] == "closed"
    finally:
        conn.close()


def test_delete_session_removes_related_rows(tmp_path, monkeypatch):
    db_path = _prepare_db(tmp_path, monkeypatch)
    store = SQLiteStore()
    store.upsert_session("s-delete", "u-1", "topic")
    store.save_messages("s-delete", [("human", "hello", True, 1)])
    store.save_raw_messages("s-delete", [("human", "hello", 1)])
    store.save_working_context_snapshot(
        session_id="s-delete",
        serialized_messages="[]",
        token_count=1,
        is_compressed=False,
    )
    store.save_compression_event(
        session_id="s-delete",
        trigger_reason="test",
        pre_tokens=10,
        post_tokens=5,
        summary_text="summary",
        summary_version="v1",
    )
    store.save_ltm_fact("u-1", "s-delete", "preference", "likes tests")

    deleted = store.delete_session("s-delete")

    assert deleted["sessions"] == 1
    assert deleted["messages"] == 1
    assert deleted["raw_messages"] == 1
    assert deleted["working_context"] == 1
    assert deleted["compression_events"] == 1
    assert deleted["ltm_facts"] == 1

    conn = sqlite3.connect(db_path)
    try:
        for table in (
            "sessions",
            "messages",
            "raw_messages",
            "working_context",
            "compression_events",
            "ltm_facts",
        ):
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id='s-delete'",
            ).fetchone()
            assert row is not None
            assert row[0] == 0
    finally:
        conn.close()
