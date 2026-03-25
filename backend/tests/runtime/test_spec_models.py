import pytest
from pydantic import ValidationError

from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


def test_agent_spec_requires_core_fields():
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"id": "agent1"})


def test_workflow_spec_requires_entry_and_nodes():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"id": "workflow1", "name": "sample"})
