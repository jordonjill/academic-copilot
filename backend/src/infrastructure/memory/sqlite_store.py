"""
SQLite 持久化层。

表结构:
  sessions(session_id PK, user_id, created_at, topic, status)
  messages(id, session_id FK, role, content, timestamp, is_backbone, token_estimate)
  ltm_facts(id, user_id, session_id, fact_type, fact_content, extracted_at)
"""
from __future__ import annotations
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple
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
                (session_id, user_id, datetime.utcnow().isoformat(), topic),
            )

    def save_messages(
        self,
        session_id: str,
        messages: List[Tuple[str, str, bool, int]],
        # (role, content, is_backbone, token_estimate)
    ) -> None:
        now = datetime.utcnow().isoformat()
        with _get_conn() as conn:
            conn.executemany(
                """INSERT INTO messages (session_id, role, content, timestamp, is_backbone, token_estimate)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(session_id, role, content, now, int(is_backbone), tokens)
                 for role, content, is_backbone, tokens in messages],
            )

    def save_ltm_fact(self, user_id: str, session_id: str, fact_type: str, fact_content: str) -> None:
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO ltm_facts (user_id, session_id, fact_type, fact_content, extracted_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, session_id, fact_type, fact_content, datetime.utcnow().isoformat()),
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

    def update_session_status(self, session_id: str, status: str) -> None:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET status=? WHERE session_id=?",
                (status, session_id),
            )
