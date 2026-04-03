from __future__ import annotations

import logging

from src.application.runtime.execution.runtime_result_service import RuntimeResultService


def _state() -> dict:
    return {
        "artifacts": {"shared": {}},
        "io": {"last_tool_outputs": []},
        "output": {"final_text": None, "final_structured": None},
    }


def test_writer_academic_backfills_draft_from_final_text() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()

    service.apply_agent_output(
        state=state,
        node_name="write",
        agent_id="writer_academic",
        text="writer raw text",
        parsed={"final_text": "writer final text"},
    )

    assert state["output"]["final_text"] == "writer final text"
    assert state["artifacts"]["draft"] == "writer final text"


def test_writer_academic_backfills_draft_from_text_when_final_text_missing() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()

    service.apply_agent_output(
        state=state,
        node_name="write",
        agent_id="writer_academic",
        text="writer raw text",
        parsed={"artifacts": {"notes": "x"}},
    )

    assert state["artifacts"]["draft"] == "writer raw text"


def test_writer_academic_does_not_override_existing_draft() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()
    state["artifacts"]["draft"] = "existing draft"

    service.apply_agent_output(
        state=state,
        node_name="write",
        agent_id="writer_academic",
        text="writer raw text",
        parsed={"final_text": "writer final text"},
    )

    assert state["artifacts"]["draft"] == "existing draft"
