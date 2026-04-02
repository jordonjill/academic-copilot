import pytest
from langchain_core.tools import BaseTool

from src.application.runtime import agent_factory as agent_runtime
from src.application.runtime.agent_factory import AgentMode
from src.application.runtime.spec_models import AgentSpec, WorkflowSpec
from src.application.runtime.workflow_router import WorkflowRuntime


class DummyTool(BaseTool):
    name: str
    description: str = "dummy tool"

    def _run(self, *args, **kwargs):
        return "ok"

    async def _arun(self, *args, **kwargs):
        return "ok"


def test_build_chain_agent_from_spec_skips_tool_resolution(monkeypatch):
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="chain",
        system_prompt="You plan",
        tools=[],
        llm={"name": "openai_default"},
    )

    def tool_resolver(tool_id):
        raise RuntimeError(f"tool resolver should not be called for chain mode: {tool_id}")

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
    assert captured["mode"] == AgentMode.CHAIN
    assert captured["llm"] is llm
    assert captured["prompt"] == "You plan"
    assert captured["tools"] == []
    assert captured["name"] == "planner"
    assert captured["output_schema"] is None


def test_build_react_agent_from_spec_wires_mode(monkeypatch):
    spec = AgentSpec(
        id="researcher",
        name="Researcher Agent",
        mode="react",
        system_prompt="You research",
        tools=["scholar_search"],
        llm={"name": "openai_default"},
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
    assert [tool.name for tool in captured["tools"]] == ["tool:scholar_search"]
    assert captured["name"] == "researcher"
    assert captured["output_schema"] is None


def test_build_agent_from_spec_raises_on_resolver_error():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="react",
        system_prompt="You plan",
        tools=["scholar_search"],
        llm={"name": "openai_default"},
    )

    def tool_resolver(tool_id):
        raise RuntimeError("boom")

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "scholar_search" in str(exc.value)


def test_build_agent_from_spec_raises_on_missing_tool():
    spec = AgentSpec(
        id="planner",
        name="Planner Agent",
        mode="react",
        system_prompt="You plan",
        tools=["paper_fetch"],
        llm={"name": "openai_default"},
    )

    def tool_resolver(tool_id):
        return None

    with pytest.raises(ValueError) as exc:
        agent_runtime.build_agent_from_spec(spec, object(), tool_resolver)
    assert "paper_fetch" in str(exc.value)


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
                "critic_dict": {"type": "agent", "agent_id": "critic"},
                "dual": {"type": "agent", "agent_id": "dual"},
                "strict": {"type": "agent", "agent_id": "strict"},
                "reporter": {"type": "agent", "agent_id": "reporter"},
                "end": {"type": "terminal"},
            },
            "edges": [
                {"from": "planner", "to": "researcher", "condition": "search"},
                {"from": "planner", "to": "synthesizer", "condition": "synthesize"},
                {"from": "researcher", "to": "planner"},
                {"from": "synthesizer", "to": "critic"},
                {"from": "critic", "to": "reporter", "condition": "valid"},
                {"from": "critic", "to": "synthesizer", "condition": "revise"},
                {
                    "from": "critic_dict",
                    "to": "reporter",
                    "condition": {"field": "research_critic.is_valid", "equals": True},
                },
                {
                    "from": "critic_dict",
                    "to": "synthesizer",
                    "condition": {"field": "research_critic.is_valid", "equals": False},
                },
                {"from": "dual", "to": "reporter", "condition": "use_cond"},
                {"from": "dual", "to": "end"},
                {"from": "strict", "to": "synthesizer", "condition": "only"},
                {"from": "reporter", "to": "end"},
            ],
            "limits": {"max_steps": 8, "max_loops": 3},
        }
    )


@pytest.fixture
def workflow_runtime(proposal_workflow_spec):
    return WorkflowRuntime(proposal_workflow_spec, agent_runner=None)


@pytest.fixture
def proposal_state():
    return {
        "_step_count": 0,
        "_loop_count": 0,
    }


def test_allowed_next_nodes_returns_all_candidates(workflow_runtime):
    assert workflow_runtime.allowed_next_nodes("planner") == ["researcher", "synthesizer"]
    assert workflow_runtime.allowed_next_nodes("researcher") == ["planner"]


def test_next_node_returns_first_allowed_edge(workflow_runtime, proposal_state):
    assert workflow_runtime.next_node("planner", proposal_state) == "researcher"
    assert workflow_runtime.next_node("researcher", proposal_state) == "planner"


def test_assert_transition_allowed_accepts_configured_edge(workflow_runtime):
    workflow_runtime.assert_transition_allowed("planner", "researcher")
    workflow_runtime.assert_transition_allowed("planner", "synthesizer")


def test_assert_transition_allowed_rejects_out_of_graph_transition(workflow_runtime):
    with pytest.raises(RuntimeError, match="Invalid transition"):
        workflow_runtime.assert_transition_allowed("planner", "reporter")


def test_next_node_raises_without_valid_transition(workflow_runtime, proposal_state):
    with pytest.raises(RuntimeError):
        workflow_runtime.next_node("nonexistent", proposal_state)


def test_enforce_limits_raises_on_step_limit(workflow_runtime, proposal_state):
    proposal_state["_step_count"] = workflow_runtime.spec.limits.get("max_steps", 30)
    with pytest.raises(RuntimeError, match="max_steps"):
        workflow_runtime.enforce_limits(proposal_state)


def test_enforce_limits_does_not_raise_on_loop_limit(workflow_runtime, proposal_state):
    proposal_state["_loop_count"] = workflow_runtime.spec.limits.get("max_loops", 6)
    workflow_runtime.enforce_limits(proposal_state)


def test_next_node_supports_expression_condition():
    spec = WorkflowSpec.model_validate(
        {
            "id": "expr_workflow",
            "name": "Expression Workflow",
            "entry_node": "start",
            "nodes": {
                "start": {"type": "agent", "agent_id": "planner"},
                "good": {"type": "terminal"},
                "bad": {"type": "terminal"},
            },
            "edges": [
                {"from": "start", "to": "good", "condition": "artifacts.score >= 0.8"},
                {"from": "start", "to": "bad"},
            ],
            "limits": {},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)
    next_node = runtime.next_node("start", {"artifacts": {"score": 0.9}})
    assert next_node == "good"


def test_next_node_loop_saturated_falls_back_to_first_edge():
    spec = WorkflowSpec.model_validate(
        {
            "id": "loop_saturated_fallback",
            "name": "Loop Saturated Fallback",
            "entry_node": "critic",
            "nodes": {
                "critic": {"type": "agent", "agent_id": "critic"},
                "export": {"type": "terminal"},
                "revise": {"type": "terminal"},
            },
            "edges": [
                {"from": "critic", "to": "export", "condition": {"field": "artifacts.quality_report.decision", "equals": "approve"}},
                {"from": "critic", "to": "revise", "condition": {"field": "artifacts.quality_report.decision", "equals": "revise"}},
            ],
            "limits": {"max_loops": 2},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)
    saturated_state = {
        "runtime": {"loop_count": 2},
        "artifacts": {"quality_report": {"decision": "revise"}},
    }
    next_node = runtime.next_node("critic", saturated_state)
    assert next_node == "export"


def test_is_node_visit_saturated_uses_max_search_alias():
    spec = WorkflowSpec.model_validate(
        {
            "id": "node_limit_alias",
            "name": "Node Limit Alias",
            "entry_node": "search",
            "nodes": {
                "search": {"type": "agent", "agent_id": "searcher"},
                "read": {"type": "agent", "agent_id": "reader"},
                "end": {"type": "terminal"},
            },
            "edges": [
                {"from": "search", "to": "read"},
                {"from": "read", "to": "end"},
            ],
            "limits": {"max_steps": 6, "max_loops": 1, "max_search": 2},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)

    assert runtime.is_node_visit_saturated("search", {"search": 2}) is False
    assert runtime.is_node_visit_saturated("search", {"search": 3}) is True


def test_next_node_for_saturated_node_avoids_self_loop():
    spec = WorkflowSpec.model_validate(
        {
            "id": "node_limit_self_loop",
            "name": "Node Limit Self Loop",
            "entry_node": "search",
            "nodes": {
                "search": {"type": "agent", "agent_id": "searcher"},
                "read": {"type": "agent", "agent_id": "reader"},
                "end": {"type": "terminal"},
            },
            "edges": [
                {"from": "search", "to": "search", "condition": {"field": "x", "equals": 1}},
                {"from": "search", "to": "read"},
                {"from": "read", "to": "end"},
            ],
            "limits": {"max_steps": 6, "max_loops": 1, "max_search": 1},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)
    assert runtime.next_node_for_saturated_node("search") == "read"


def test_workflow_spec_rejects_sum_of_node_caps_exceeding_max_steps():
    with pytest.raises(ValueError, match="sum of node visit limits must be <= limits.max_steps"):
        WorkflowSpec.model_validate(
            {
                "id": "bad_caps",
                "name": "Bad Caps",
                "entry_node": "planner",
                "nodes": {
                    "planner": {"type": "agent", "agent_id": "planner"},
                    "search": {"type": "agent", "agent_id": "search"},
                    "end": {"type": "terminal"},
                },
                "edges": [
                    {"from": "planner", "to": "search"},
                    {"from": "search", "to": "end"},
                ],
                "limits": {"max_steps": 3, "max_loops": 1, "max_planner": 2, "max_search": 2},
            }
        )


def test_workflow_spec_rejects_path_plus_loops_exceeding_max_steps():
    with pytest.raises(ValueError, match="max_steps must cover baseline path plus max_loops"):
        WorkflowSpec.model_validate(
            {
                "id": "bad_loop_budget",
                "name": "Bad Loop Budget",
                "entry_node": "planner",
                "nodes": {
                    "planner": {"type": "agent", "agent_id": "planner"},
                    "search": {"type": "agent", "agent_id": "search"},
                    "end": {"type": "terminal"},
                },
                "edges": [
                    {"from": "planner", "to": "search"},
                    {"from": "search", "to": "end"},
                ],
                "limits": {"max_steps": 3, "max_loops": 2},
            }
        )


def test_expression_condition_with_missing_field_falls_back():
    spec = WorkflowSpec.model_validate(
        {
            "id": "expr_missing_field",
            "name": "Expression Missing Field",
            "entry_node": "start",
            "nodes": {
                "start": {"type": "agent", "agent_id": "planner"},
                "good": {"type": "terminal"},
                "fallback": {"type": "terminal"},
            },
            "edges": [
                {"from": "start", "to": "good", "condition": "artifacts.score >= 0.8"},
                {"from": "start", "to": "fallback"},
            ],
            "limits": {},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)
    next_node = runtime.next_node("start", {"artifacts": {}})
    assert next_node == "fallback"


def test_dict_condition_missing_expected_value_falls_back():
    spec = WorkflowSpec.model_validate(
        {
            "id": "dict_missing_expected",
            "name": "Dict Missing Expected",
            "entry_node": "start",
            "nodes": {
                "start": {"type": "agent", "agent_id": "planner"},
                "good": {"type": "terminal"},
                "fallback": {"type": "terminal"},
            },
            "edges": [
                {
                    "from": "start",
                    "to": "good",
                    "condition": {"field": "artifacts.score", "op": "gte"},
                },
                {"from": "start", "to": "fallback"},
            ],
            "limits": {},
        }
    )
    runtime = WorkflowRuntime(spec, agent_runner=None)
    next_node = runtime.next_node("start", {"artifacts": {"score": 0.9}})
    assert next_node == "fallback"
