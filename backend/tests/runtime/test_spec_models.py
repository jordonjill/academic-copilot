import pytest
from pydantic import ValidationError

from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


def test_agent_spec_requires_core_fields():
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"id": "agent1"})


def test_workflow_spec_requires_entry_and_nodes():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"id": "workflow1", "name": "sample"})


def test_agent_spec_accepts_minimal_valid_payload():
    spec = AgentSpec.model_validate(
        {
            "id": "planner",
            "name": "Planner Agent",
            "mode": "chain",
            "system_prompt": "You plan",
            "llm": {"name": "openai_default"},
        }
    )
    assert spec.tools == []
    assert spec.llm.name == "openai_default"


def test_workflow_spec_accepts_minimal_valid_payload():
    spec = WorkflowSpec.model_validate(
        {
            "id": "proposal",
            "name": "Proposal v2",
            "entry_node": "start",
            "nodes": {"start": {"type": "agent"}},
            "edges": [{"from": "start", "to": "end"}],
        }
    )
    assert spec.entry_node == "start"
    assert len(spec.edges) == 1


def test_workflow_spec_requires_edges_field():
    with pytest.raises(ValidationError) as exc:
        WorkflowSpec.model_validate(
            {
                "id": "proposal",
                "name": "Proposal v2",
                "entry_node": "start",
                "nodes": {"start": {"type": "agent"}},
            }
        )
    errors = exc.value.errors()
    assert any(error["loc"] == ("edges",) for error in errors)


def test_agent_spec_rejects_extra_top_level_fields():
    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(
            {
                "id": "planner",
                "name": "Planner",
                "mode": "chain",
                "system_prompt": "plan",
                "llm": {"name": "openai_default"},
                "extra": "not allowed",
            }
        )
    errors = exc.value.errors()
    assert any(error["loc"] == ("extra",) for error in errors)


def test_agent_spec_rejects_extra_fields_in_llm():
    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(
            {
                "id": "planner",
                "name": "Planner",
                "mode": "react",
                "system_prompt": "respond",
                "llm": {
                    "name": "openai_default",
                    "extra_llm": "not allowed",
                },
            }
        )
    errors = exc.value.errors()
    assert any(error["loc"] == ("llm", "extra_llm") for error in errors)


def test_agent_spec_rejects_chain_mode_with_tools():
    with pytest.raises(ValidationError) as exc:
        AgentSpec.model_validate(
            {
                "id": "planner",
                "name": "Planner",
                "mode": "chain",
                "system_prompt": "plan",
                "tools": ["web_search"],
                "llm": {"name": "openai_default"},
            }
        )
    errors = exc.value.errors()
    assert any("chain mode does not support tools" in str(error.get("msg", "")) for error in errors)
