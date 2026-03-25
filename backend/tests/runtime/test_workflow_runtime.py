import pytest
from langchain_core.tools import BaseTool

from src.application.agents import AgentMode
from src.application.runtime import agent_runtime
from src.application.runtime.spec_models import AgentSpec, WorkflowSpec
from src.application.runtime.workflow_runtime import WorkflowRuntime
from src.domain.state.schema import ResearchCritic, ResearchPlan


class DummyTool(BaseTool):
    name: str
    description: str = "dummy tool"

    def _run(self, *args, **kwargs):
        return "ok"

    async def _arun(self, *args, **kwargs):
        return "ok"


def test_build_chain_agent_from_spec_resolves_tools(monkeypatch):
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["web_search", "arxiv"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    resolved = []

    def tool_resolver(tool_id):
        resolved.append(tool_id)
        return DummyTool(name=f"tool:{tool_id}")

    captured = {}

    def fake_create_subagent(mode, llm, *, prompt, tools=None, output_schema=None, name="agent"):
        captured["mode"] = mode
        captured["llm"] = llm
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["name"] = name
        captured["output_schema"] = output_schema
        return "agent_instance"

    monkeypatch.setattr(agent_runtime, "create_subagent", fake_create_subagent)

    llm = object()
    result = agent_runtime.build_agent_from_spec(spec, llm, tool_resolver)

    assert result == "agent_instance"
    assert resolved == ["web_search", "arxiv"]
    assert captured["mode"] == AgentMode.CHAIN
    assert captured["llm"] is llm
    assert captured["prompt"] == "You plan"
    assert [tool.name for tool in captured["tools"]] == ["tool:web_search", "tool:arxiv"]
    assert captured["name"] == "planner"
    assert captured["output_schema"] is None


def test_build_react_agent_from_spec_wires_mode(monkeypatch):
    spec = AgentSpec(
        id="researcher",
        name="Researcher Agent",
        mode="react",
        system_prompt="You research",
        tools=["web_search"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        return DummyTool(name=f"tool:{tool_id}")

    captured = {}

    def fake_create_subagent(mode, llm, *, prompt, tools=None, output_schema=None, name="agent"):
        captured["mode"] = mode
        captured["llm"] = llm
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["name"] = name
        captured["output_schema"] = output_schema
        return "react_agent"

    monkeypatch.setattr(agent_runtime, "create_subagent", fake_create_subagent)

    llm = object()
    result = agent_runtime.build_agent_from_spec(spec, llm, tool_resolver)

    assert result == "react_agent"
    assert captured["mode"] == AgentMode.REACT
    assert captured["llm"] is llm
    assert captured["prompt"] == "You research"
    assert [tool.name for tool in captured["tools"]] == ["tool:web_search"]
    assert captured["name"] == "researcher"
    assert captured["output_schema"] is None


def test_build_agent_from_spec_raises_on_resolver_error():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["web_search"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        raise RuntimeError("boom")

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "web_search" in str(exc.value)


def test_build_agent_from_spec_raises_on_missing_tool():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=["arxiv"],
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    def tool_resolver(tool_id):
        return None

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "arxiv" in str(exc.value)


@pytest.fixture
def proposal_workflow_spec():
    return WorkflowSpec.model_validate(
        {
            "id": "proposal_simple",
            "name": "Proposal Simple",
            "entry_node": "planner",
            "nodes": {
                "planner": {"type": "agent", "agent_id": "planner"},
                "researcher": {"type": "agent", "agent_id": "researcher"},
                "synthesizer": {"type": "agent", "agent_id": "synthesizer"},
                "critic": {"type": "agent", "agent_id": "critic"},
                "reporter": {"type": "agent", "agent_id": "reporter"},
                "end": {"type": "terminal"},
            },
            "edges": [
                {"from": "planner", "to": "researcher", "condition": "search"},
                {"from": "planner", "to": "synthesizer", "condition": "synthesize"},
                {"from": "researcher", "to": "planner"},
                {"from": "synthesizer", "to": "critic"},
                {
                    "from": "critic",
                    "to": "reporter",
                    "condition": {"field": "research_critic.is_valid", "equals": True},
                },
                {
                    "from": "critic",
                    "to": "synthesizer",
                    "condition": {"field": "research_critic.is_valid", "equals": False},
                },
                {"from": "reporter", "to": "end"},
            ],
            "limits": {"max_steps": 5, "max_loops": 3},
        }
    )


@pytest.fixture
def workflow_runtime(proposal_workflow_spec):
    return WorkflowRuntime(proposal_workflow_spec, agent_runner=None)


@pytest.fixture
def proposal_state():
    return {
        "route_key": "search",
        "research_plan": ResearchPlan(has_enough_content=True, step_type="search", query="q"),
        "research_critic": ResearchCritic(is_valid=True),
        "_step_count": 0,
        "_loop_count": 0,
    }


def test_workflow_routes_based_on_route_key(workflow_runtime, proposal_state):
    proposal_state["route_key"] = "search"
    next_node = workflow_runtime.next_node("planner", proposal_state)
    assert next_node == "researcher"


def test_workflow_routes_using_research_plan_when_route_key_missing(
    workflow_runtime, proposal_state
):
    proposal_state["route_key"] = ""
    proposal_state["research_plan"] = ResearchPlan(
        has_enough_content=True, step_type="synthesize", query="q"
    )
    next_node = workflow_runtime.next_node("planner", proposal_state)
    assert next_node == "synthesizer"


def test_default_edge_used_when_no_condition_matches(workflow_runtime, proposal_state):
    assert workflow_runtime.next_node("researcher", proposal_state) == "planner"


def test_dict_condition_routing(workflow_runtime, proposal_state):
    proposal_state["research_critic"] = ResearchCritic(is_valid=False)
    assert workflow_runtime.next_node("critic", proposal_state) == "synthesizer"
    proposal_state["research_critic"] = ResearchCritic(is_valid=True)
    assert workflow_runtime.next_node("critic", proposal_state) == "reporter"


def test_next_node_raises_without_valid_transition(workflow_runtime, proposal_state):
    with pytest.raises(RuntimeError):
        workflow_runtime.next_node("nonexistent", proposal_state)


def test_enforce_limits_raises_on_step_limit(workflow_runtime, proposal_state):
    proposal_state["_step_count"] = workflow_runtime.spec.limits.get("max_steps", 30)
    with pytest.raises(RuntimeError, match="max_steps"):
        workflow_runtime.enforce_limits(proposal_state)


def test_enforce_limits_raises_on_loop_limit(workflow_runtime, proposal_state):
    proposal_state["_loop_count"] = workflow_runtime.spec.limits.get("max_loops", 6)
    with pytest.raises(RuntimeError, match="max_loops"):
        workflow_runtime.enforce_limits(proposal_state)
