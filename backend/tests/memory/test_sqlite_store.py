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
