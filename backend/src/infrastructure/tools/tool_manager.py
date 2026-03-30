from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_TOOL_CATALOG_PATH_ENV = "TOOL_CATALOG_PATH"


@dataclass
class ServerSpec:
    name: str
    transport: str
    enabled: bool
    command: str = ""
    args: list[str] | None = None
    env: dict[str, str] | None = None


@dataclass
class ToolSpec:
    tool_id: str
    transport: str
    enabled: bool
    module: str = ""
    attribute: str = ""
    server: str = ""
    tool_name: str = ""


def _default_catalog_path() -> Path:
    env_path = os.getenv(_TOOL_CATALOG_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser()
    # backend/src/infrastructure/tools/tool_manager.py -> backend/config/tools.yaml
    return Path(__file__).resolve().parents[3] / "config" / "tools.yaml"


def _expand_env(value: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


class ToolManager:
    """Unified tool manager for internal and MCP-backed tools."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        self.catalog_path = catalog_path or _default_catalog_path()
        self.version = 0
        self._loaded = False
        self._tool_specs: dict[str, ToolSpec] = {}
        self._server_specs: dict[str, ServerSpec] = {}
        self._tools: dict[str, BaseTool] = {}
        self._server_tools: dict[str, list[BaseTool]] = {}

    def _load_catalog_payload(self) -> dict[str, Any]:
        if not self.catalog_path.exists():
            logger.warning("Tool catalog not found: %s", self.catalog_path)
            return {"servers": {}, "tools": {}}
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            logger.warning("Tool catalog root must be mapping: %s", self.catalog_path)
            return {"servers": {}, "tools": {}}
        return payload

    def _parse_specs(self, payload: dict[str, Any]) -> None:
        self._server_specs = {}
        self._tool_specs = {}

        servers = payload.get("servers", {})
        if isinstance(servers, dict):
            for name, raw in servers.items():
                if not isinstance(raw, dict):
                    continue
                self._server_specs[name] = ServerSpec(
                    name=name,
                    transport=str(raw.get("transport", "stdio")),
                    enabled=bool(raw.get("enabled", False)),
                    command=str(raw.get("command", "")),
                    args=list(raw.get("args", []) or []),
                    env={k: _expand_env(str(v)) for k, v in (raw.get("env") or {}).items()},
                )

        tools = payload.get("tools", {})
        if isinstance(tools, dict):
            for tool_id, raw in tools.items():
                if not isinstance(raw, dict):
                    continue
                self._tool_specs[tool_id] = ToolSpec(
                    tool_id=tool_id,
                    transport=str(raw.get("transport", "internal")),
                    enabled=bool(raw.get("enabled", False)),
                    module=str(raw.get("module", "")),
                    attribute=str(raw.get("attribute", "")),
                    server=str(raw.get("server", "")),
                    tool_name=str(raw.get("tool_name", "")),
                )

    def _load_internal_tool(self, spec: ToolSpec) -> BaseTool | None:
        if not spec.module or not spec.attribute:
            logger.warning(
                "Internal tool '%s' missing module/attribute in catalog",
                spec.tool_id,
            )
            return None
        try:
            mod = importlib.import_module(spec.module)
            tool_obj = getattr(mod, spec.attribute)
        except Exception:
            logger.exception(
                "Failed to import internal tool '%s' from %s.%s",
                spec.tool_id,
                spec.module,
                spec.attribute,
            )
            return None

        if not hasattr(tool_obj, "name") or not hasattr(tool_obj, "description"):
            logger.warning(
                "Internal tool '%s' is missing tool interface (name/description)",
                spec.tool_id,
            )
            return None
        return tool_obj

    def _pick_server_tool(self, server_name: str, spec: ToolSpec) -> BaseTool | None:
        server_tools = self._server_tools.get(server_name, [])
        if not server_tools:
            logger.warning(
                "No MCP tools loaded for server '%s' (tool_id=%s)",
                server_name,
                spec.tool_id,
            )
            return None

        if spec.tool_name:
            for tool in server_tools:
                if getattr(tool, "name", "") == spec.tool_name:
                    return tool
            logger.warning(
                "MCP tool '%s' not found on server '%s' for tool_id '%s'",
                spec.tool_name,
                server_name,
                spec.tool_id,
            )
            return None

        return server_tools[0]

    async def _load_mcp_server_tools(self) -> None:
        self._server_tools = {}

        mcp_server_params: dict[str, dict[str, Any]] = {}
        for name, spec in self._server_specs.items():
            if not spec.enabled:
                continue
            if spec.transport != "stdio":
                continue
            mcp_server_params[name] = {
                "command": spec.command,
                "args": spec.args or [],
                "env": spec.env or None,
                "transport": "stdio",
            }

        if not mcp_server_params:
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.warning("langchain-mcp-adapters not installed, MCP tools skipped.")
            return

        try:
            async with MultiServerMCPClient(mcp_server_params) as client:
                combined_tools: list[BaseTool] | None = None
                for server_name in mcp_server_params:
                    tools: list[BaseTool] = []
                    try:
                        try:
                            tools = client.get_tools(server_name=server_name)
                        except TypeError:
                            try:
                                tools = client.get_tools(server_name)
                            except TypeError:
                                pass

                        if not tools:
                            if combined_tools is None:
                                combined_tools = list(client.get_tools())
                            if len(mcp_server_params) == 1:
                                tools = list(combined_tools)
                            else:
                                prefix = f"{server_name}."
                                tools = [
                                    t
                                    for t in combined_tools
                                    if getattr(t, "name", "").startswith(prefix)
                                ]
                        self._server_tools[server_name] = tools
                        logger.info(
                            "Loaded %d MCP tools from server '%s'",
                            len(tools),
                            server_name,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to fetch MCP tools for server '%s'",
                            server_name,
                        )
                        self._server_tools[server_name] = []
        except Exception:
            logger.exception("MCP client initialization failed")

    def _collect_enabled_internal_tools(self) -> dict[str, BaseTool]:
        tools: dict[str, BaseTool] = {}
        for tool_id, spec in self._tool_specs.items():
            if not spec.enabled or spec.transport != "internal":
                continue
            loaded = self._load_internal_tool(spec)
            if loaded is not None:
                tools[tool_id] = loaded
        return tools

    def _collect_enabled_mcp_tools(self) -> dict[str, BaseTool]:
        tools: dict[str, BaseTool] = {}
        for tool_id, spec in self._tool_specs.items():
            if not spec.enabled or spec.transport != "mcp":
                continue
            if not spec.server:
                logger.warning("MCP tool '%s' missing server in catalog", tool_id)
                continue
            picked = self._pick_server_tool(spec.server, spec)
            if picked is not None:
                tools[tool_id] = picked
        return tools

    def load_internal_only(self) -> dict[str, Any]:
        payload = self._load_catalog_payload()
        self._parse_specs(payload)
        self._server_tools = {}
        tools = self._collect_enabled_internal_tools()
        self._tools = tools
        self.version += 1
        self._loaded = True
        return self.report()

    async def reload(self) -> dict[str, Any]:
        payload = self._load_catalog_payload()
        self._parse_specs(payload)
        tools = self._collect_enabled_internal_tools()
        await self._load_mcp_server_tools()
        tools.update(self._collect_enabled_mcp_tools())
        self._tools = tools
        self.version += 1
        self._loaded = True
        return self.report()

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_internal_only()

    def get_tool(self, tool_id: str) -> BaseTool | None:
        self.ensure_loaded()
        return self._tools.get(tool_id)

    def get_catalog_tool_ids(self, *, enabled_only: bool = True) -> set[str]:
        payload = self._load_catalog_payload()
        tools = payload.get("tools", {})
        if not isinstance(tools, dict):
            return set()

        tool_ids: set[str] = set()
        for tool_id, raw in tools.items():
            if not isinstance(raw, dict):
                continue
            if enabled_only and not bool(raw.get("enabled", False)):
                continue
            tool_ids.add(str(tool_id))
        return tool_ids

    def report(self) -> dict[str, Any]:
        return {
            "tool_catalog_path": str(self.catalog_path),
            "version": self.version,
            "loaded_tools": sorted(self._tools.keys()),
            "loaded_servers": sorted(self._server_tools.keys()),
        }


_TOOL_MANAGER = ToolManager()


def get_tool_manager() -> ToolManager:
    return _TOOL_MANAGER
