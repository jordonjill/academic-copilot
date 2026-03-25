import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


def test_agent_spec_requires_core_fields():
    with pytest.raises(ValidationError):
        AgentSpec.model_validate({"id": "agent1"})


def test_workflow_spec_requires_entry_and_nodes():
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate({"id": "workflow1", "name": "sample"})
