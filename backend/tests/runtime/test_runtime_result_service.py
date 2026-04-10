from __future__ import annotations

import logging

from src.application.runtime.execution.runtime_result_service import RuntimeResultService


def _state() -> dict:
    return {
        "artifacts": {"shared": {}},
        "io": {"last_tool_outputs": []},
        "output": {"final_text": None, "final_structured": None},
    }


def test_writer_academic_keeps_canonical_document_artifact() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()

    service.apply_agent_output(
        state=state,
        node_name="write",
        agent_id="writer_academic",
        text="writer raw text",
        parsed={
            "final_text": "draft ready",
            "artifacts": {
                "document": {
                    "title": "Doc title",
                    "summary": "Doc summary",
                    "body": "Doc body",
                    "references": [{"title": "Ref", "uri": "https://example.com"}],
                }
            },
        },
    )

    assert state["output"]["final_text"] == "draft ready"
    assert state["artifacts"]["document"]["title"] == "Doc title"
    assert state["artifacts"]["document"]["body"] == "Doc body"


def test_writer_academic_does_not_backfill_legacy_draft_field() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()

    service.apply_agent_output(
        state=state,
        node_name="write",
        agent_id="writer_academic",
        text="writer raw text",
        parsed={"final_text": "draft ready", "artifacts": {"notes": "x"}},
    )

    assert "draft" not in state["artifacts"]
