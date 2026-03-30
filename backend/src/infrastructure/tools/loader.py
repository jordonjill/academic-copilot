from __future__ import annotations

from typing import Any

from src.infrastructure.tools.tool_manager import get_tool_manager


async def initialize_tools() -> dict[str, Any]:
    return await get_tool_manager().reload()


async def reload_tools() -> dict[str, Any]:
    return await get_tool_manager().reload()
