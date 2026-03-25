import pytest

from src.application.runtime.orchestrator import (
    MAX_RETRIES_PER_AGENT,
    SupervisorOrchestrator,
)


@pytest.fixture
def orchestrator() -> SupervisorOrchestrator:
    return SupervisorOrchestrator()


@pytest.fixture
def base_state() -> dict:
    return {
        "pending_workflow_confirmation": False,
        "suggested_workflow_id": None,
        "orchestration_mode": "workflow",
        "selected_subagents": [],
        "confirmation_expires_at_turn": None,
        "last_selected_agent_id": None,
        "agent_retry_counters": {},
    }


def test_pending_confirmation_interruption_switches_to_dynamic(orchestrator, base_state):
    state = base_state.copy()
    state.update(
        {
            "pending_workflow_confirmation": True,
            "orchestration_mode": "workflow",
            "suggested_workflow_id": "proposal",
            "confirmation_expires_at_turn": 10,
        }
    )

    orchestrator.handle_user_input(state, "顺便帮我解释一下这个术语")

    assert state["pending_workflow_confirmation"] is False
    assert state["suggested_workflow_id"] is None
    assert state["orchestration_mode"] == "dynamic"
    assert state["confirmation_expires_at_turn"] is None


@pytest.mark.parametrize(
    "text",
    ["yes", "use workflow", "使用", "no", "don't use", "不使用"],
)
def test_confirmation_responses_do_not_interrupt(orchestrator, base_state, text):
    state = base_state.copy()
    state["pending_workflow_confirmation"] = True
    state["orchestration_mode"] = "workflow"

    orchestrator.handle_user_input(state, text)

    assert state["pending_workflow_confirmation"] is True
    assert state["orchestration_mode"] == "workflow"


def test_retry_cap_blocks_same_agent_loop(orchestrator, base_state):
    state = base_state.copy()
    state.update(
        {
            "last_selected_agent_id": "researcher_proposal",
            "agent_retry_counters": {
                "researcher_proposal": MAX_RETRIES_PER_AGENT,
            },
            "selected_subagents": ["researcher_proposal"],
        }
    )

    selected = orchestrator.select_next_agent(state)

    assert selected is None


def test_retry_cap_chooses_alternative_agent(orchestrator, base_state):
    state = base_state.copy()
    state.update(
        {
            "last_selected_agent_id": "researcher_proposal",
            "agent_retry_counters": {
                "researcher_proposal": MAX_RETRIES_PER_AGENT,
            },
            "selected_subagents": [
                "researcher_proposal",
                "writer_proposal",
            ],
        }
    )

    selected = orchestrator.select_next_agent(state)

    assert selected == "writer_proposal"
    assert state["last_selected_agent_id"] == "writer_proposal"
    assert state["agent_retry_counters"]["writer_proposal"] == 1
    assert state["agent_retry_counters"]["researcher_proposal"] == MAX_RETRIES_PER_AGENT
