from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.application.runtime.execution.isolation_facility import IsolationFacility
from src.application.runtime.execution.runtime_result_service import RuntimeResultService
from src.application.runtime.execution.task_payload import render_chain_payload, render_react_messages
from src.application.runtime.providers.context_facility import ContextFacility, ContextPolicy


def _parent_state() -> dict:
    return {
        "input": {
            "user_text": "Write a short literature review.",
            "user_id": "u1",
            "session_id": "s1",
        },
        "context": {
            "messages": [HumanMessage(content="Write a short literature review.")],
            "memory_summary": "",
        },
        "runtime": {
            "mode": "dynamic",
            "workflow_id": None,
            "current_node": None,
            "step_count": 0,
            "loop_count": 0,
            "status": "running",
        },
        "io": {
            "last_model_output": None,
            "last_execution_output": None,
            "last_tool_outputs": [],
        },
        "artifacts": {
            "topic": "agent memory",
            "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}],
            "shared": {"legacy_agent": {"output_text": "legacy duplicate"}},
            "execution_trace": [{"action": "legacy"}],
            "supervisor_instruction": "legacy instruction",
            "task_input": {"legacy": True},
            "workflow_runner_input": {"legacy": True},
        },
        "task": {},
        "executions": [],
        "output": {
            "final_text": None,
            "final_structured": None,
        },
        "errors": {
            "last_error": None,
        },
    }


def _facility() -> IsolationFacility:
    result_service = RuntimeResultService(logging.getLogger(__name__))
    return IsolationFacility(
        logger=logging.getLogger(__name__),
        apply_agent_output=result_service.apply_agent_output,
    )


def test_child_state_gets_task_and_selected_artifacts_without_internal_artifacts() -> None:
    parent = _parent_state()
    facility = _facility()

    child = facility.build_isolated_subagent_state(
        parent,
        "reader_extractor",
        "Extract grounded claims from the evidence bank.",
        input_artifact_keys=[
            "evidence_bank",
            "shared",
            "supervisor_instruction",
            "task_input",
        ],
    )

    assert child["task"]["instruction"] == "Extract grounded claims from the evidence bank."
    assert child["task"]["input_artifacts"] == {
        "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}]
    }
    assert child["artifacts"]["evidence_bank"] == [
        {"title": "Memory in LLM agents", "uri": "paper://1"}
    ]
    assert "shared" not in child["artifacts"]
    assert "execution_trace" not in child["artifacts"]
    assert "supervisor_instruction" not in child["artifacts"]
    assert "task_input" not in child["artifacts"]
    assert "workflow_runner_input" not in child["artifacts"]
    assert child["executions"] == []

    assert child["context"]["messages"] == []


def test_chain_and_react_payloads_share_the_same_task_input_semantics() -> None:
    parent = _parent_state()
    facility = _facility()
    child = facility.build_isolated_subagent_state(
        parent,
        "reader_extractor",
        "Extract grounded claims from the evidence bank.",
        input_artifact_keys=["evidence_bank"],
    )

    chain_payload = render_chain_payload(child, ContextFacility())
    react_messages = render_react_messages(
        child,
        agent_id="reader_extractor",
        node_name="read",
        runtime_mode="subagent",
    )
    envelope = str(react_messages[0].content)
    react_payload = json.loads(envelope.split("\n", 2)[2])
    react_task_input = react_payload["task_input"]

    assert "messages" not in chain_payload
    assert chain_payload["supervisor_instruction"] == react_task_input["instruction"]
    assert json.loads(chain_payload["artifacts"]) == react_task_input["input_artifacts"]
    assert react_payload["protocol"] == "task_input_v1"
    assert react_payload["agent_id"] == "reader_extractor"
    assert react_payload["node_name"] == "read"


def test_render_react_messages_applies_context_facility_cap_without_task_input() -> None:
    class _FakeEncoder:
        @staticmethod
        def encode(text: str) -> list[str]:
            return text.split()

    facility = ContextFacility(
        policy=ContextPolicy(
            default_messages_window=4,
            supervisor_messages_window=4,
            trace_recent_window=2,
            trace_max_items=6,
            text_preview_chars=120,
            trace_output_preview_chars=120,
            trace_reason_chars=120,
            trace_instruction_chars=120,
            react_messages_token_cap=40,
        )
    )
    facility._token_encoder = _FakeEncoder()
    state = _parent_state()
    state["task"] = {}
    state["context"]["messages"] = [
        HumanMessage(content="alpha " * 20),
        AIMessage(content="beta " * 20),
        HumanMessage(content="gamma " * 20),
    ]

    react_messages = render_react_messages(
        state,
        agent_id="reader_extractor",
        node_name="read",
        runtime_mode="subagent",
        context_facility=facility,
    )

    assert len(react_messages) == 1
    assert str(react_messages[0].content) == state["context"]["messages"][-1].content


def test_subagent_result_merges_only_artifact_patch_and_execution_record_to_parent() -> None:
    parent = _parent_state()
    facility = _facility()
    original_messages = list(parent["context"]["messages"])

    facility.deliver_execution_result_to_supervisor(
        parent,
        {
            "source_kind": "subagent",
            "source_id": "reader_extractor",
            "output_text": "Extracted two grounded claims.",
            "tool_outputs": [{"tool": "paper_fetch", "ok": True}],
            "artifacts_patch": {
                "claim_map": [
                    {
                        "claim": "Agents need durable memory.",
                        "support_uri": "paper://1",
                    }
                ],
                "shared": {"should_not": "flow"},
                "supervisor_instruction": "should not flow",
            },
        },
    )

    assert parent["artifacts"] == {
        "topic": "agent memory",
        "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}],
        "claim_map": [
            {
                "claim": "Agents need durable memory.",
                "support_uri": "paper://1",
            }
        ],
    }
    assert parent["context"]["messages"] == original_messages
    assert parent["io"]["last_model_output"] == "Extracted two grounded claims."
    assert parent["io"]["last_execution_output"] == "Extracted two grounded claims."
    assert parent["output"]["final_text"] is None

    assert parent["executions"] == [
        {
            "source_kind": "subagent",
            "source_id": "reader_extractor",
            "node": "reader_extractor",
            "output_text": "Extracted two grounded claims.",
            "output_preview": "Extracted two grounded claims.",
            "artifact_keys": ["claim_map"],
            "tool_outputs": [{"tool": "paper_fetch", "ok": True}],
        }
    ]


def test_agent_final_text_is_execution_output_not_global_user_response() -> None:
    parent = _parent_state()
    service = RuntimeResultService(logging.getLogger(__name__))
    raw_model_output = '{"final_text": "Draft ready", "artifacts": {"document": {"title": "T"}}}'
    parent["io"]["last_model_output"] = raw_model_output

    service.apply_agent_output(
        state=parent,
        node_name="write",
        agent_id="writer_academic",
        text=raw_model_output,
        parsed={
            "status": "success",
            "final_text": "Draft ready",
            "artifacts": {"document": {"title": "T"}},
        },
    )

    assert parent["output"]["final_text"] is None
    assert parent["io"]["last_model_output"] == raw_model_output
    assert parent["io"]["last_execution_output"] == "Draft ready"
    assert parent["artifacts"] == {
        "topic": "agent memory",
        "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}],
        "document": {"title": "T"},
    }
    assert parent["executions"][-1]["output_text"] == "Draft ready"
    assert parent["executions"][-1]["artifact_keys"] == ["document"]


def test_io_last_outputs_are_recent_cache_not_information_flow() -> None:
    parent = _parent_state()
    parent["io"]["last_model_output"] = "stale raw output"
    parent["io"]["last_execution_output"] = "stale execution output"
    facility = _facility()

    child = facility.build_isolated_subagent_state(
        parent,
        "writer_academic",
        "Write only from selected evidence.",
        input_artifact_keys=["evidence_bank"],
    )

    assert child["task"]["input_artifacts"] == {
        "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}]
    }
    assert child["artifacts"] == {
        "topic": "agent memory",
        "evidence_bank": [{"title": "Memory in LLM agents", "uri": "paper://1"}],
    }
    assert child["io"] == {
        "last_model_output": None,
        "last_execution_output": None,
        "last_tool_outputs": [],
    }
