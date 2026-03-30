"""
SQLite 持久化层。

表结构:
  sessions(session_id PK, user_id, created_at, topic, status)
  messages(id, session_id FK, role, content, timestamp, is_backbone, token_estimate)
  ltm_facts(id, user_id, session_id, fact_type, fact_content, extracted_at)
  raw_messages(id, session_id FK, role, content, timestamp, token_estimate)
  working_context(id, session_id FK, serialized_messages, token_count, is_compressed, created_at)
  compression_events(id, session_id FK, trigger_reason, pre_tokens, post_tokens, summary_text, summary_digest, summary_version, created_at)
"""
from __future__ import annotations
import hashlib
import sqlite3
import os
from datetime import UTC, datetime
from typing import List, Tuple
from src.infrastructure.config.config import CONVERSATION_DB


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CONVERSATION_DB), exist_ok=True)
    conn = sqlite3.connect(CONVERSATION_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等）。"""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                topic       TEXT,
                status      TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES sessions(session_id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                is_backbone     INTEGER DEFAULT 1,
                token_estimate  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS ltm_facts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                session_id    TEXT,
                fact_type     TEXT NOT NULL,
                fact_content  TEXT NOT NULL,
                extracted_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES sessions(session_id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                token_estimate  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS working_context (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL REFERENCES sessions(session_id),
                serialized_messages TEXT NOT NULL,
                token_count         INTEGER NOT NULL,
                is_compressed       INTEGER NOT NULL,
                created_at          TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compression_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES sessions(session_id),
                trigger_reason  TEXT NOT NULL,
                pre_tokens      INTEGER NOT NULL,
                post_tokens     INTEGER NOT NULL,
                summary_text    TEXT NOT NULL,
                summary_digest  TEXT NOT NULL,
                summary_version TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
        """)


class SQLiteStore:
    """会话持久化 CRUD 封装。"""

    def __init__(self):
        init_db()

    def upsert_session(self, session_id: str, user_id: str, topic: str = "") -> None:
        with _get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, topic)
                   VALUES (?, ?, ?, ?)""",
                (session_id, user_id, datetime.now(UTC).isoformat(), topic),
            )

    def save_messages(
        self,
        session_id: str,
        messages: List[Tuple[str, str, bool, int]],
        # (role, content, is_backbone, token_estimate)
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with _get_conn() as conn:
            conn.executemany(
                """INSERT INTO messages (session_id, role, content, timestamp, is_backbone, token_estimate)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(session_id, role, content, now, int(is_backbone), tokens)
                 for role, content, is_backbone, tokens in messages],
            )

    def save_raw_messages(self, session_id: str, messages: List[Tuple[str, str, int]]) -> None:
        if not messages:
            return
        now = datetime.now(UTC).isoformat()
        with _get_conn() as conn:
            conn.executemany(
                """INSERT INTO raw_messages (session_id, role, content, timestamp, token_estimate)
                   VALUES (?, ?, ?, ?, ?)""",
                [(session_id, role, content, now, tokens)
                 for role, content, tokens in messages],
            )

    def save_working_context_snapshot(
        self,
        session_id: str,
        serialized_messages: str,
        token_count: int,
        is_compressed: bool,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO working_context (session_id, serialized_messages, token_count, is_compressed, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, serialized_messages, token_count, int(is_compressed), now),
            )

    def save_compression_event(
        self,
        session_id: str,
        trigger_reason: str,
        pre_tokens: int,
        post_tokens: int,
        summary_text: str,
        summary_version: str,
    ) -> None:
        digest = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        now = datetime.now(UTC).isoformat()
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO compression_events
                   (session_id, trigger_reason, pre_tokens, post_tokens, summary_text, summary_digest, summary_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, trigger_reason, pre_tokens, post_tokens, summary_text, digest, summary_version, now),
            )

    def save_ltm_fact(self, user_id: str, session_id: str, fact_type: str, fact_content: str) -> None:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO ltm_facts (user_id, session_id, fact_type, fact_content, extracted_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, session_id, fact_type, fact_content, datetime.now(UTC).isoformat()),
            )

    def get_session_messages(self, session_id: str, backbone_only: bool = False) -> List[sqlite3.Row]:
        with _get_conn() as conn:
            if backbone_only:
                return conn.execute(
                    "SELECT * FROM messages WHERE session_id=? AND is_backbone=1 ORDER BY id",
                    (session_id,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()

    def get_latest_working_context_snapshot(self, session_id: str) -> sqlite3.Row | None:
        with _get_conn() as conn:
            return conn.execute(
                """SELECT * FROM working_context
                   WHERE session_id=?
                   ORDER BY id DESC
                   LIMIT 1""",
                (session_id,),
            ).fetchone()

    def update_session_status(self, session_id: str, status: str) -> None:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET status=? WHERE session_id=?",
                (status, session_id),
            )
