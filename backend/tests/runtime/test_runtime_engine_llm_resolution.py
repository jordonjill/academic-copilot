from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.contracts.spec_models import AgentSpec, LLMProfileSpec


def test_resolve_llm_rejects_unresolved_base_url_placeholder(tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.llms = {
        "broken_profile": LLMProfileSpec.model_validate(
            {
                "name": "broken_profile",
                "model_name": "gpt-4o-mini",
                "base_url": "${MISSING_BASE_URL}",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0.0,
            }
        )
    }
    spec = AgentSpec.model_validate(
        {
            "id": "supervisor",
            "name": "Supervisor",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "broken_profile"},
        }
    )

    engine = RuntimeEngine(registry=registry)
    with pytest.raises(RuntimeError, match="Unresolved env placeholder in base_url"):
        engine._resolve_llm(spec)


def test_resolve_llm_sets_custom_user_agent_for_non_openai_host(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.llms = {
        "openai_default": LLMProfileSpec.model_validate(
            {
                "name": "openai_default",
                "model_name": "gpt-5",
                "base_url": "https://proxy.example.com/v1",
                "temperature": 0.0,
            }
        )
    }
    spec = AgentSpec.model_validate(
        {
            "id": "supervisor",
            "name": "Supervisor",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "openai_default"},
        }
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/Test")
    monkeypatch.setattr("src.application.runtime.runtime_engine.ChatOpenAI", _FakeChatOpenAI)

    engine = RuntimeEngine(registry=registry)
    engine._resolve_llm(spec)

    assert captured_kwargs["default_headers"] == {"User-Agent": "AcademicCopilot/Test"}


def test_resolve_llm_keeps_openai_host_default_user_agent(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.llms = {
        "openai_default": LLMProfileSpec.model_validate(
            {
                "name": "openai_default",
                "model_name": "gpt-5",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0.0,
            }
        )
    }
    spec = AgentSpec.model_validate(
        {
            "id": "supervisor",
            "name": "Supervisor",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "openai_default"},
        }
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/Test")
    monkeypatch.setattr("src.application.runtime.runtime_engine.ChatOpenAI", _FakeChatOpenAI)

    engine = RuntimeEngine(registry=registry)
    engine._resolve_llm(spec)

    assert "default_headers" not in captured_kwargs


def test_resolve_llm_reuses_single_instance_under_concurrency(monkeypatch, tmp_path):
    registry = ConfigRegistry(config_root=tmp_path)
    registry.llms = {
        "openai_default": LLMProfileSpec.model_validate(
            {
                "name": "openai_default",
                "model_name": "gpt-4o-mini",
                "base_url": "https://example.com/v1",
                "temperature": 0.0,
            }
        )
    }
    spec = AgentSpec.model_validate(
        {
            "id": "supervisor",
            "name": "Supervisor",
            "mode": "chain",
            "system_prompt": "x",
            "tools": [],
            "llm": {"name": "openai_default"},
        }
    )

    create_count = {"value": 0}
    create_lock = threading.Lock()

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            with create_lock:
                create_count["value"] += 1
            # Expand race window to verify in-flight coordination.
            time.sleep(0.05)

    monkeypatch.setattr("src.application.runtime.runtime_engine.ChatOpenAI", _FakeChatOpenAI)

    engine = RuntimeEngine(registry=registry)

    def _worker():
        return engine._resolve_llm(spec)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _worker(), range(8)))

    assert create_count["value"] == 1
    assert all(result is results[0] for result in results)
