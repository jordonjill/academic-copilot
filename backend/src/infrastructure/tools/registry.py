from __future__ import annotations

from langchain_core.tools import BaseTool

from src.infrastructure.tools.tool_manager import get_tool_manager


def get_tool(tool_id: str) -> BaseTool | None:
    return get_tool_manager().get_tool(tool_id)
