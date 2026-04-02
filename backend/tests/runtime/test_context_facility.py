from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.application.runtime.providers.context_facility import ContextFacility, ContextPolicy


def test_messages_to_text_uses_scope_windows():
    facility = ContextFacility(
        policy=ContextPolicy(
            default_messages_window=2,
            supervisor_messages_window=4,
            trace_recent_window=2,
            trace_max_items=6,
            shared_summary_items=3,
            text_preview_chars=120,
            trace_output_preview_chars=120,
            trace_reason_chars=120,
            trace_instruction_chars=120,
        )
    )
    messages = [
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        AIMessage(content="a2"),
    ]

    default_text = facility.messages_to_text(messages, scope="default")
    supervisor_text = facility.messages_to_text(messages, scope="supervisor")

    assert "u1" not in default_text
    assert "a1" not in default_text
    assert "u2" in default_text
    assert "a2" in default_text
    assert "u1" in supervisor_text
    assert "a1" in supervisor_text


def test_trace_is_capped_and_recent_window_applies():
    facility = ContextFacility(
        policy=ContextPolicy(
            default_messages_window=2,
            supervisor_messages_window=4,
            trace_recent_window=2,
            trace_max_items=3,
            shared_summary_items=3,
            text_preview_chars=120,
            trace_output_preview_chars=120,
            trace_reason_chars=120,
            trace_instruction_chars=120,
        )
    )
    artifacts: dict = {}
    for index in range(5):
        facility.append_trace(artifacts, entry={"n": index}, trace_key="execution_trace")

    trace = artifacts["execution_trace"]
    assert [item["n"] for item in trace] == [2, 3, 4]
    recent = facility.recent_trace(artifacts, trace_key="execution_trace")
    assert [item["n"] for item in recent] == [3, 4]


def test_compact_artifacts_excludes_internal_keys():
    facility = ContextFacility(
        policy=ContextPolicy(
            default_messages_window=2,
            supervisor_messages_window=4,
            trace_recent_window=2,
            trace_max_items=3,
            shared_summary_items=1,
            text_preview_chars=120,
            trace_output_preview_chars=120,
            trace_reason_chars=120,
            trace_instruction_chars=120,
        )
    )
    artifacts = {
        "shared": {
            "agent_a": {"node": "n1", "output_text": "abc", "parsed": {"k": 1}},
            "agent_b": {"node": "n2", "output_text": "def", "parsed": {"k": 2}},
        },
        "execution_trace": [{"step_count": 1}],
        "task_input": {"x": 1},
        "research_gap": "gap",
    }

    compact = facility.compact_artifacts(
        artifacts,
        excluded_keys={"shared", "execution_trace", "task_input"},
    )
    summary = compact["artifact_summary"]

    assert "research_gap" in summary
    assert "task_input" not in summary
    assert len(compact["shared_summary"]) == 1
