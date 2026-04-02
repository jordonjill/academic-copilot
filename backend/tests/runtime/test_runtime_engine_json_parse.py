from __future__ import annotations

from src.application.runtime.config.config_registry import ConfigRegistry
from src.application.runtime.runtime_engine import RuntimeEngine


def test_try_parse_json_extracts_first_valid_object_without_cross_block(tmp_path):
    engine = RuntimeEngine(registry=ConfigRegistry(config_root=tmp_path))
    text = (
        "prefix\n"
        "{\"action\": \"direct_reply\", \"message\": \"ok\"}\n"
        "middle\n"
        "{\"noise\": true}\n"
    )
    parsed = engine._try_parse_json(text)
    assert parsed is not None
    assert parsed.get("action") == "direct_reply"


def test_try_parse_json_returns_none_when_no_valid_object(tmp_path):
    engine = RuntimeEngine(registry=ConfigRegistry(config_root=tmp_path))
    text = "no json here {not-valid"
    assert engine._try_parse_json(text) is None
