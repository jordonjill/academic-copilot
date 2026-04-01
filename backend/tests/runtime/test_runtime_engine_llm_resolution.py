from __future__ import annotations

import pytest

from src.application.runtime.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.application.runtime.spec_models import AgentSpec, LLMProfileSpec


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
