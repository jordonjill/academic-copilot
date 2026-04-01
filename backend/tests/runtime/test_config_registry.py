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
                "  name: openai_default",
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
                "  name: openai_default",
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
                "  name: openai_default",
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
                "  name: openai_default",
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
                "  name: openai_default",
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
                "  name: openai_default",
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


def test_registry_survives_oserror(tmp_path, monkeypatch):
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
                "  name: openai_default",
            ]
        )
    )
    (agents_dir / "bad_io.yaml").write_text("id: bad_io_agent")
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

    original_load = registry._load_yaml

    def fail_on_bad_io(path):
        if path.name == "bad_io.yaml":
            raise OSError("read failed")
        return original_load(path)

    monkeypatch.setattr(registry, "_load_yaml", fail_on_bad_io)

    report = registry.reload()
    assert "planner_proposal" in report["loaded_agents"]
    assert any(
        failure.get("type") == "agent"
        and "bad_io.yaml" in failure.get("path", "")
        for failure in report["failed_objects"]
    )


def test_registry_duplicate_id_rejected(tmp_path):
    config_root = tmp_path / "config"
    agents_dir = config_root / "agents"
    workflows_dir = config_root / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)

    (agents_dir / "dup_a.yaml").write_text(
        "\n".join(
            [
                "id: duplicate_agent",
                "name: First Agent",
                "mode: chain",
                "system_prompt: |",
                "  First version.",
                "tools: []",
                "llm:",
                "  name: openai_default",
            ]
        )
    )
    (agents_dir / "dup_b.yaml").write_text(
        "\n".join(
            [
                "id: duplicate_agent",
                "name: Second Agent",
                "mode: chain",
                "system_prompt: |",
                "  Second version.",
                "tools: []",
                "llm:",
                "  name: openai_default",
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
                "    agent_id: duplicate_agent",
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
    assert "duplicate_agent" not in report["loaded_agents"]
    assert any(
        failure.get("type") == "agent"
        and "dup_a.yaml" in failure.get("path", "")
        for failure in report["failed_objects"]
    )
    assert any(
        failure.get("type") == "agent"
        and "dup_b.yaml" in failure.get("path", "")
        for failure in report["failed_objects"]
    )


def test_registry_preserves_last_known_good(tmp_path):
    config_root = tmp_path / "config"
    agents_dir = config_root / "agents"
    workflows_dir = config_root / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)

    agent_path = agents_dir / "planner.yaml"
    agent_path.write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "name: Proposal Planner",
                "mode: chain",
                "system_prompt: |",
                "  Decide whether to search for sources or synthesize findings.",
                "tools: []",
                "llm:",
                "  name: openai_default",
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
    assert "planner_proposal" in first["loaded_agents"]
    assert registry.agents["planner_proposal"].name == "Proposal Planner"

    agent_path.write_text(
        "\n".join(
            [
                "id: planner_proposal",
                "mode: chain",
                "system_prompt: invalid now",
                "llm:",
                "  name: openai_default",
            ]
        )
    )

    second = registry.reload()
    assert "planner_proposal" in second["loaded_agents"]
    assert registry.agents["planner_proposal"].name == "Proposal Planner"
    assert any(
        failure.get("type") == "agent"
        and "planner.yaml" in failure.get("path", "")
        for failure in second["failed_objects"]
    )


def test_registry_llm_env_expansion_allows_missing_variables_for_unused_profiles(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    config_root.mkdir(parents=True)
    monkeypatch.delenv("MISSING_LLM_BASE_URL", raising=False)

    (config_root / "llms.yaml").write_text(
        "\n".join(
            [
                "llms:",
                "  test_llm:",
                "    model_name: gpt-4o-mini",
                "    base_url: ${MISSING_LLM_BASE_URL}",
                "    api_key_env: OPENAI_API_KEY",
                "    temperature: 0.0",
            ]
        )
    )

    registry = ConfigRegistry(config_root=config_root)
    report = registry.reload()

    assert "test_llm" in report["loaded_llms"]
    assert registry.llms["test_llm"].base_url == "${MISSING_LLM_BASE_URL}"
    assert report["failed_objects"] == []


def test_registry_llm_validation_failure_reports_unresolved_placeholders(tmp_path, monkeypatch):
    config_root = tmp_path / "config"
    config_root.mkdir(parents=True)
    monkeypatch.delenv("MISSING_LLM_BASE_URL", raising=False)

    (config_root / "llms.yaml").write_text(
        "\n".join(
            [
                "llms:",
                "  bad_llm:",
                "    base_url: ${MISSING_LLM_BASE_URL}",
                "    api_key_env: OPENAI_API_KEY",
                "    temperature: 0.0",
            ]
        )
    )

    registry = ConfigRegistry(config_root=config_root)
    report = registry.reload()

    assert "bad_llm" not in report["loaded_llms"]
    assert any(
        failure.get("type") == "llm"
        and "unresolved env placeholders: MISSING_LLM_BASE_URL" in failure.get("error", "")
        for failure in report["failed_objects"]
    )
