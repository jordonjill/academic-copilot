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
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        }
    )
    assert spec.tools == []
    assert spec.llm.model == "gpt-4o-mini"


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
