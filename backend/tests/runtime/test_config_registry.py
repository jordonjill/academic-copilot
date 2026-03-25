import pytest

from src.application.runtime import hook_registry
from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.hook_registry import HOOK_REGISTRY, register_hook, resolve_hook


def test_unknown_hook_rejected():
    with pytest.raises(KeyError):
        resolve_hook("not_registered_hook")


@pytest.fixture(autouse=True)
def clear_hooks():
    hook_registry._HOOK_REGISTRY.clear()
    yield
    hook_registry._HOOK_REGISTRY.clear()


def test_register_then_resolve():
    HOOK_REGISTRY  # ensure the read-only view is accessible
    def sample_hook():
        return "ok"

    register_hook("test-hook", sample_hook)
    assert resolve_hook("test-hook") is sample_hook


def test_duplicate_registration_rejected():
    def noop():
        return None

    register_hook("dup-hook", noop)
    with pytest.raises(KeyError):
        register_hook("dup-hook", noop)


def test_registry_loads_workflow_and_agents(tmp_path):
    config_root = tmp_path / "config"
    agents_dir = config_root / "agents"
    workflows_dir = config_root / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)

    (agents_dir / "planner.yaml").write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "name: Proposal Planner",
                "mode: chain",
                "system_prompt: |",
                "  Decide whether to search for sources or synthesize findings.",
                "tools: []",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
                "  temperature: 0.0",
            ]
        )
    )
    (workflows_dir / "proposal_v2.yaml").write_text(
        "\n".join(
            [
                "id: proposal_v2",
                "name: Proposal v2",
                "entry_node: planner",
                "nodes:",
                "  planner:",
                "    type: agent",
                "    agent_id: planner_proposal",
                "  end:",
                "    type: terminal",
                "edges:",
                "  - from: planner",
                "    to: end",
            ]
        )
    )

    registry = ConfigRegistry(config_root=config_root)
    report = registry.reload()

    assert report["config_version"] == 1
    assert "planner_proposal" in report["loaded_agents"]
    assert "proposal_v2" in report["loaded_workflows"]


def test_registry_partial_failure_isolated(tmp_path):
    config_root = tmp_path / "config"
    agents_dir = config_root / "agents"
    workflows_dir = config_root / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)

    (agents_dir / "planner.yaml").write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "name: Proposal Planner",
                "mode: chain",
                "system_prompt: |",
                "  Decide whether to search for sources or synthesize findings.",
                "tools: []",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
                "  temperature: 0.0",
            ]
        )
    )
    (agents_dir / "bad.yaml").write_text(
        "\n".join(
            [
                "id: bad_agent",
                "mode: chain",
                "system_prompt: oops",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
            ]
        )
    )
    (workflows_dir / "proposal_v2.yaml").write_text(
        "\n".join(
            [
                "id: proposal_v2",
                "name: Proposal v2",
                "entry_node: planner",
                "nodes:",
                "  planner:",
                "    type: agent",
                "    agent_id: planner_proposal",
                "  end:",
                "    type: terminal",
                "edges:",
                "  - from: planner",
                "    to: end",
            ]
        )
    )

    registry = ConfigRegistry(config_root=config_root)
    report = registry.reload()

    assert "planner_proposal" in report["loaded_agents"]
    assert "proposal_v2" in report["loaded_workflows"]
    assert "bad_agent" not in report["loaded_agents"]
    assert report["failed_objects"]
    assert any(
        failure.get("type") == "agent"
        and "bad.yaml" in failure.get("path", "")
        for failure in report["failed_objects"]
    )


def test_registry_reload_increments_and_replaces_state(tmp_path):
    config_root = tmp_path / "config"
    agents_dir = config_root / "agents"
    workflows_dir = config_root / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)

    (agents_dir / "planner.yaml").write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "name: Proposal Planner",
                "mode: chain",
                "system_prompt: |",
                "  Decide whether to search for sources or synthesize findings.",
                "tools: []",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
                "  temperature: 0.0",
            ]
        )
    )
    (workflows_dir / "proposal_v2.yaml").write_text(
        "\n".join(
            [
                "id: proposal_v2",
                "name: Proposal v2",
                "entry_node: planner",
                "nodes:",
                "  planner:",
                "    type: agent",
                "    agent_id: planner_proposal",
                "  end:",
                "    type: terminal",
                "edges:",
                "  - from: planner",
                "    to: end",
            ]
        )
    )

    registry = ConfigRegistry(config_root=config_root)
    first = registry.reload()
    assert first["config_version"] == 1
    assert "planner_proposal" in first["loaded_agents"]

    (agents_dir / "planner.yaml").write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "name: Updated Planner",
                "mode: chain",
                "system_prompt: |",
                "  Decide with updated context.",
                "tools: []",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
                "  temperature: 0.0",
            ]
        )
    )
    (agents_dir / "extra.yaml").write_text(
        "\n".join(
            [
                "id: extra_agent",
                "name: Extra Agent",
                "mode: chain",
                "system_prompt: |",
                "  Extra behavior for reload test.",
                "tools: []",
                "llm:",
                "  provider: openai",
                "  model: gpt-4o-mini",
                "  temperature: 0.0",
            ]
        )
    )
    (workflows_dir / "proposal_v2.yaml").write_text(
        "\n".join(
            [
                "id: proposal_v2",
                "name: Proposal v2",
                "entry_node: planner",
                "nodes:",
                "  planner:",
                "    type: agent",
                "    agent_id: planner_proposal",
                "  end:",
                "    type: terminal",
                "edges:",
                "  - from: planner",
                "    to: end",
            ]
        )
    )

    second = registry.reload()
    assert second["config_version"] == 2
    assert registry.agents["planner_proposal"].name == "Updated Planner"
    assert "extra_agent" in registry.agents
