from __future__ import annotations

import yaml

from src.infrastructure.tools.tool_manager import ToolManager


def test_tool_manager_loads_internal_tools(tmp_path):
    catalog_path = tmp_path / "tools.yaml"
    payload = {
        "version": "1.0",
        "servers": {},
        "tools": {
            "arxiv": {
                "transport": "internal",
                "module": "src.infrastructure.tools.arxiv_search",
                "attribute": "search_arxiv",
                "enabled": True,
            },
            "web_search": {
                "transport": "internal",
                "module": "src.infrastructure.tools.crawl_search",
                "attribute": "crawl_search",
                "enabled": True,
            },
            "disabled_tool": {
                "transport": "internal",
                "module": "src.infrastructure.tools.arxiv_search",
                "attribute": "search_arxiv",
                "enabled": False,
            },
        },
    }
    catalog_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manager = ToolManager(catalog_path=catalog_path)
    report = manager.load_internal_only()

    assert "arxiv" in report["loaded_tools"]
    assert "web_search" in report["loaded_tools"]
    assert "disabled_tool" not in report["loaded_tools"]
    assert manager.get_tool("arxiv") is not None
    assert manager.get_tool("disabled_tool") is None
    assert manager.get_catalog_tool_ids(enabled_only=True) == {"arxiv", "web_search"}
    assert manager.get_catalog_tool_ids(enabled_only=False) == {"arxiv", "web_search", "disabled_tool"}


def test_tool_manager_handles_missing_catalog_gracefully(tmp_path):
    catalog_path = tmp_path / "missing.yaml"
    manager = ToolManager(catalog_path=catalog_path)
    report = manager.load_internal_only()

    assert report["loaded_tools"] == []
    assert manager.get_tool("web_search") is None
