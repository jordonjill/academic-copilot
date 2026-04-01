from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

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
                "settings": {
                    "max_results": 7,
                    "include_domains": ["https://example.org"],
                },
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
    assert manager.get_tool_settings("web_search") == {
        "max_results": 7,
        "include_domains": ["https://example.org"],
    }
    assert manager.get_tool_settings("unknown_tool") == {}
    assert manager.get_catalog_tool_ids(enabled_only=True) == {"arxiv", "web_search"}
    assert manager.get_catalog_tool_ids(enabled_only=False) == {"arxiv", "web_search", "disabled_tool"}


def test_tool_manager_handles_missing_catalog_gracefully(tmp_path):
    catalog_path = tmp_path / "missing.yaml"
    manager = ToolManager(catalog_path=catalog_path)
    report = manager.load_internal_only()

    assert report["loaded_tools"] == []
    assert manager.get_tool("web_search") is None


def test_tool_manager_loads_mcp_tools_with_async_client(monkeypatch, tmp_path):
    catalog_path = tmp_path / "tools.yaml"
    payload = {
        "version": "1.0",
        "servers": {
            "filesystem": {
                "transport": "stdio",
                "enabled": True,
                "command": "python",
                "args": ["fake_server.py"],
            }
        },
        "tools": {
            "filesystem": {
                "transport": "mcp",
                "server": "filesystem",
                "tool_name": "read_file",
                "enabled": True,
            }
        },
    }
    catalog_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    class _FakeMCPTool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = "fake mcp tool"

    class _FakeMultiServerMCPClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self, *, server_name=None):
            if server_name == "filesystem":
                return [_FakeMCPTool("read_file")]
            return []

        async def __aenter__(self):
            raise AssertionError("ToolManager must not use MultiServerMCPClient as async context manager")

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setitem(
        sys.modules,
        "langchain_mcp_adapters.client",
        SimpleNamespace(MultiServerMCPClient=_FakeMultiServerMCPClient),
    )

    manager = ToolManager(catalog_path=catalog_path)
    report = asyncio.run(manager.reload())

    assert "filesystem" in report["loaded_servers"]
    assert "filesystem" in report["loaded_tools"]
    assert manager.get_tool("filesystem") is not None


def test_tool_manager_reports_mcp_server_failures(monkeypatch, tmp_path):
    catalog_path = tmp_path / "tools.yaml"
    payload = {
        "version": "1.0",
        "servers": {
            "broken_server": {
                "transport": "stdio",
                "enabled": True,
                "command": "python",
                "args": ["broken_server.py"],
            }
        },
        "tools": {
            "broken_tool": {
                "transport": "mcp",
                "server": "broken_server",
                "tool_name": "read_file",
                "enabled": True,
            }
        },
    }
    catalog_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    class _BrokenClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self, *, server_name=None):
            del server_name
            raise RuntimeError("server unavailable")

    monkeypatch.setitem(
        sys.modules,
        "langchain_mcp_adapters.client",
        SimpleNamespace(MultiServerMCPClient=_BrokenClient),
    )

    manager = ToolManager(catalog_path=catalog_path)
    report = asyncio.run(manager.reload())

    assert report["failed_servers"]
    assert "broken_server" in report["failed_servers"]
    assert report["failed_tools"]
    assert "broken_tool" in report["failed_tools"]
