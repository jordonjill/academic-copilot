from __future__ import annotations

import logging

from src.application.runtime.execution.runtime_result_service import RuntimeResultService


def _state() -> dict:
    return {
        "artifacts": {"shared": {}},
        "runtime": {
            "mode": "dynamic",
            "workflow_id": None,
            "current_node": None,
            "step_count": 1,
            "max_steps": 8,
            "loop_count": 0,
            "status": "completed",
            "tool_budget": {
                "scope": "turn",
                "workflow_id": None,
                "limits": {"scholar_search": 2},
                "counts": {"scholar_search": 1},
                "node_visit_limits": {"internal": 1},
            },
        },
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


def test_build_result_exposes_only_public_outputs() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()
    state["output"]["final_text"] = "done"
    state["artifacts"].update(
        {
            "report_exports": {
                "docx_path": "data/exports/report.docx",
                "pdf_path": "data/exports/report.pdf",
            },
            "private_notes": "do not expose",
            "shared": {
                "researcher": {
                    "output_text": "large internal output",
                    "parsed": {"artifacts": {"files": [{"content": "secret-ish text"}]}},
                    "tool_outputs": [{"content": "tool internals"}],
                }
            },
        }
    )

    result = service.build_result(state)

    assert result["data"]["outputs"] == {
        "report_exports": {
            "docx_path": "data/exports/report.docx",
            "pdf_path": "data/exports/report.pdf",
        }
    }
    assert "artifacts" not in result["data"]
    assert "private_notes" not in result["data"]["outputs"]
    assert result["data"]["runtime"]["tool_budget"] == {
        "scope": "turn",
        "workflow_id": None,
        "limits": {"scholar_search": 2},
        "counts": {"scholar_search": 1},
    }


def test_build_result_extracts_nested_report_exports_only() -> None:
    service = RuntimeResultService(logging.getLogger(__name__))
    state = _state()
    state["output"]["final_text"] = "done"
    state["artifacts"]["shared"] = {
        "report_exporter": {
            "output_text": "internal",
            "parsed": {
                "artifacts": {
                    "report_exports": {"docx_path": "report.docx"},
                    "files": [{"content": "not public"}],
                }
            },
            "tool_outputs": [{"content": "not public"}],
        }
    }

    result = service.build_result(state)

    assert result["data"]["outputs"] == {
        "report_exports": {"docx_path": "report.docx"}
    }
