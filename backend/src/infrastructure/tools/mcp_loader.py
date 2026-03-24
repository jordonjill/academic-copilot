"""
MCP 工具动态加载器。

设计：
  - 在 FastAPI lifespan 中初始化所有 enabled MCP server
  - 提供 get_tools_for_role(role) 按角色分发工具集
  - 内置工具（transport=internal）直接从 Python 模块导入，不走 MCP 协议

依赖：pip install langchain-mcp-adapters pyyaml
"""
from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from langchain_core.tools import BaseTool

# 内置工具注册表（transport=internal 的工具直接导入）
_INTERNAL_TOOLS: Dict[str, List[BaseTool]] = {}

# MCP 客户端（lifespan 初始化后填充）
_mcp_tools_cache: Dict[str, List[BaseTool]] = {}

# 角色→server 名称映射（从 yaml 加载）
_role_server_map: Dict[str, List[str]] = {}

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "mcp_servers.yaml"


def _load_yaml_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {"servers": {}, "role_tools": {}}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_env(value: str) -> str:
    """展开 ${VAR} 形式的环境变量引用。"""
    import re
    return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ""), value)


def _load_internal_tools(server_name: str, module_path: str) -> List[BaseTool]:
    """从内部 Python 模块动态加载工具函数。"""
    try:
        mod = importlib.import_module(module_path)
        tools = []
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if callable(attr) and hasattr(attr, "name") and hasattr(attr, "description"):
                tools.append(attr)
        return tools
    except Exception as e:
        print(f"[MCP] Failed to load internal tool '{server_name}' from '{module_path}': {e}")
        return []


async def initialize_mcp_tools() -> None:
    """
    FastAPI lifespan 启动时调用。
    加载所有 enabled server 的工具并缓存。
    """
    global _mcp_tools_cache, _role_server_map, _INTERNAL_TOOLS

    config = _load_yaml_config()
    servers: Dict[str, Any] = config.get("servers", {})
    _role_server_map = config.get("role_tools", {})

    mcp_server_params: Dict[str, Any] = {}

    for name, server_cfg in servers.items():
        if not server_cfg.get("enabled", False):
            continue

        transport = server_cfg.get("transport", "stdio")

        if transport == "internal":
            module_path = server_cfg.get("module", "")
            tools = _load_internal_tools(name, module_path)
            _INTERNAL_TOOLS[name] = tools
            _mcp_tools_cache[name] = tools
            print(f"[MCP] Loaded internal tools for '{name}': {[t.name for t in tools]}")
            continue

        # stdio / sse MCP servers → langchain-mcp-adapters
        command = server_cfg.get("command", "")
        args = server_cfg.get("args", [])
        env = {k: _expand_env(v) for k, v in (server_cfg.get("env") or {}).items()}

        if transport == "stdio":
            mcp_server_params[name] = {
                "command": command,
                "args": args,
                "env": env or None,
                "transport": "stdio",
            }

    # 批量初始化 MCP servers（仅在有 langchain-mcp-adapters 时执行）
    if mcp_server_params:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            async with MultiServerMCPClient(mcp_server_params) as client:
                for server_name in mcp_server_params:
                    try:
                        tools = client.get_tools()
                        _mcp_tools_cache[server_name] = tools
                        print(f"[MCP] Loaded {len(tools)} tools from '{server_name}'")
                    except Exception as e:
                        print(f"[MCP] Failed to get tools from '{server_name}': {e}")
                        _mcp_tools_cache[server_name] = []
        except ImportError:
            print("[MCP] langchain-mcp-adapters not installed. MCP servers skipped.")
        except Exception as e:
            print(f"[MCP] Initialization error: {e}")


def get_tools_for_role(role: str) -> List[BaseTool]:
    """
    按角色返回工具列表。
    优先从缓存获取，若 MCP 未初始化则直接加载内置工具作为降级。
    """
    server_names = _role_server_map.get(role, [])
    tools: List[BaseTool] = []
    seen_names: set = set()

    for server_name in server_names:
        for t in _mcp_tools_cache.get(server_name, []):
            if t.name not in seen_names:
                tools.append(t)
                seen_names.add(t.name)

    # 降级：如果缓存为空，直接加载 arxiv 和 crawl_search 内置工具
    if not tools:
        tools = _get_fallback_tools(role)

    return tools


def _get_fallback_tools(role: str) -> List[BaseTool]:
    """降级工具集（MCP 未初始化时使用）。"""
    from src.infrastructure.tools.crawl_search import crawl_search
    from src.infrastructure.tools.arxiv_search import search_arxiv

    if role in ("researcher",):
        return [crawl_search, search_arxiv]
    if role in ("critic",):
        return [crawl_search, search_arxiv]
    return []
