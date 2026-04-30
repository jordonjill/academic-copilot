from __future__ import annotations

import asyncio
import types
import uuid

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine
from src.infrastructure.observability.langfuse_observability import (
    TokenUsageCollector,
    build_langchain_config,
    langchain_observation_context,
)


def test_build_langchain_config_merges_context_callbacks_metadata_and_tags() -> None:
    sentinel = object()
    existing_callback = object()

    with langchain_observation_context(
        callbacks=[sentinel],
        metadata={"langfuse_session_id": "s1", "context_key": "context"},
        tags=["academic-copilot", "agent:writer"],
    ):
        config = build_langchain_config(
            {
                "callbacks": [existing_callback],
                "metadata": {"local_key": "local"},
                "tags": ["local-tag"],
            },
            metadata={"node_name": "writer_node"},
            tags=["workflow:proposal"],
            run_name="agent.writer",
        )

    assert config is not None
    assert config["callbacks"] == [existing_callback, sentinel]
    assert config["metadata"] == {
        "local_key": "local",
        "langfuse_session_id": "s1",
        "context_key": "context",
        "node_name": "writer_node",
    }
    assert config["tags"] == ["local-tag", "academic-copilot", "agent:writer", "workflow:proposal"]
    assert config["run_name"] == "agent.writer"


def test_runtime_engine_invoke_async_passes_observation_config(tmp_path) -> None:
    engine = RuntimeEngine(registry=ConfigRegistry(config_root=tmp_path))
    sentinel = object()
    captured: dict = {}

    class _Runnable:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {"ok": True}

    with langchain_observation_context(callbacks=[sentinel], metadata={"session_id": "s1"}, tags=["trace"]):
        result = asyncio.run(
            engine._invoke_async(
                _Runnable(),
                {"x": 1},
                config={"metadata": {"local": True}, "tags": ["local"]},
            )
        )

    assert result == {"ok": True}
    assert captured["payload"] == {"x": 1}
    assert captured["config"]["callbacks"] == [sentinel]
    assert captured["config"]["metadata"] == {"local": True, "session_id": "s1"}
    assert captured["config"]["tags"] == ["local", "trace"]


def test_token_usage_collector_reads_provider_usage() -> None:
    collector = TokenUsageCollector()
    run_id = uuid.uuid4()

    collector.on_llm_start(
        {"kwargs": {"model": "gpt-test"}},
        ["hello"],
        run_id=run_id,
    )
    collector.on_llm_end(
        types.SimpleNamespace(
            llm_output={
                "model_name": "gpt-test",
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
            generations=[],
        ),
        run_id=run_id,
    )

    summary = collector.summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 5
    assert summary["total_tokens"] == 15
    assert summary["cached_tokens"] == 3
    assert summary["reasoning_tokens"] == 2
    assert summary["estimated_calls"] == 0
    assert summary["by_model"]["gpt-test"]["total_tokens"] == 15


def test_token_usage_collector_estimates_when_provider_usage_missing() -> None:
    collector = TokenUsageCollector()
    run_id = uuid.uuid4()

    collector.on_llm_start(
        {"kwargs": {"model": "local-model"}},
        ["input prompt"],
        run_id=run_id,
    )
    collector.on_llm_end(
        types.SimpleNamespace(
            llm_output={},
            generations=[[types.SimpleNamespace(text="estimated output")]],
        ),
        run_id=run_id,
    )

    summary = collector.summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] > 0
    assert summary["output_tokens"] > 0
    assert summary["total_tokens"] == summary["input_tokens"] + summary["output_tokens"]
    assert summary["estimated_calls"] == 1
    assert summary["by_model"]["local-model"]["estimated_calls"] == 1
