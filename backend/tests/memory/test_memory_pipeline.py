import json
import sqlite3
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.infrastructure.memory import stm as stm_module
from src.infrastructure.memory import sqlite_store
from src.infrastructure.memory.stm import stm_compression_node


def _prepare_db(tmp_path, monkeypatch):
    db_path = tmp_path / "pipeline.db"
    monkeypatch.setattr(sqlite_store, "CONVERSATION_DB", str(db_path))
    return str(db_path)


def _read_rows(db_path: str, table: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    conn.close()
    return rows


def _base_state(messages):
    return {
        "session_id": "session-abc",
        "user_id": "user-123",
        "initial_topic": "TestTopic",
        "messages": messages,
    }


def test_raw_messages_persist_each_turn(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 9999)
    ai_content = [{"type": "text", "text": "ai list"}]
    state = _base_state([
        HumanMessage(content="hello"),
        AIMessage(content=ai_content),
    ])
    result = stm_compression_node(state, FakeListChatModel(responses=["never-used"]))

    raw_rows = _read_rows(db_path, "raw_messages")
    assert len(raw_rows) == 2
    assert raw_rows[0]["role"] == "human"
    assert raw_rows[1]["role"] == "assistant"
    working_rows = _read_rows(db_path, "working_context")
    assert working_rows
    assert working_rows[-1]["is_compressed"] == 0
    assert working_rows[-1]["token_count"] == result["stm_token_count"]
    assert result["stm_compressed"] is False
    assert raw_rows[1]["content"] == json.dumps(ai_content, ensure_ascii=False)


def test_compression_rewrites_working_context_history_remains(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 2)
    monkeypatch.setattr(stm_module, "STM_KEEP_RECENT", 1)

    messages = [
        HumanMessage(content="old human data"),
        AIMessage(content="old ai data"),
        HumanMessage(content="recent human data"),
        AIMessage(content="recent ai data"),
    ]
    state = _base_state(messages)
    result = stm_compression_node(state, FakeListChatModel(responses=["compressed summary"]))

    assert result["stm_compressed"] is True
    compressed_messages = result["messages"]
    assert isinstance(compressed_messages[0], SystemMessage)
    assert "compressed" in compressed_messages[0].content.lower()
    assert compressed_messages[-1].content == messages[-1].content

    working_rows = _read_rows(db_path, "working_context")
    snapshot = json.loads(working_rows[-1]["serialized_messages"])
    assert snapshot[0]["type"] == "SystemMessage"
    assert snapshot[-1]["data"]["content"] == messages[-1].content

    raw_rows = _read_rows(db_path, "raw_messages")
    assert len(raw_rows) == 4


def test_compression_event_recorded(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 2)
    monkeypatch.setattr(stm_module, "STM_KEEP_RECENT", 1)
    state = _base_state([
        HumanMessage(content="preamble"),
        AIMessage(content="response"),
    ])
    stm_compression_node(state, FakeListChatModel(responses=["summary trace"]))

    events = _read_rows(db_path, "compression_events")
    assert len(events) == 1
    event = events[0]
    assert event["summary_text"] == "summary trace"
    assert event["summary_digest"]
    assert event["pre_tokens"] > 0
    assert event["post_tokens"] > 0


def test_no_compression_event_when_within_threshold(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 9999)
    state = _base_state([
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ])
    stm_compression_node(state, FakeListChatModel(responses=["unused summary"]))

    events = _read_rows(db_path, "compression_events")
    assert not events


def test_threshold_exceeded_without_old_messages_records_event(monkeypatch, tmp_path):
    db_path = _prepare_db(tmp_path, monkeypatch)
    monkeypatch.setattr(stm_module, "STM_TOKEN_THRESHOLD", 2)
    monkeypatch.setattr(stm_module, "STM_KEEP_RECENT", 5)
    long_text = "x" * 500
    state = _base_state([
        HumanMessage(content=long_text),
    ])
    result = stm_compression_node(state, FakeListChatModel(responses=["unused summary"]))

    events = _read_rows(db_path, "compression_events")
    assert len(events) == 1
    event = events[0]
    assert event["summary_text"] == "no-op compression"
    assert event["summary_digest"]
    assert event["pre_tokens"] == result["stm_token_count"]
    assert event["post_tokens"] == result["stm_token_count"]
    assert result["stm_compressed"] is False
