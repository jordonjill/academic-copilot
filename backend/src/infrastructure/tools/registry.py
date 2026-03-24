"""
工具注册表（Tool Registry）。

设计原则：
  - ToolGroup 枚举定义原子工具集和组合工具集
  - get_tools(*groups) 按需获取去重工具列表，惰性导入避免循环依赖
  - MCP 动态工具通过 mcp_loader 在运行时注入，inject_mcp_tools() 完成注册
  - 对外暴露预定义角色工具包常量（RESEARCHER_TOOLS / CRITIC_TOOLS / WRITER_TOOLS）

用法示例：
  from src.infrastructure.tools.registry import get_tools, ToolGroup
  tools = get_tools(ToolGroup.ARXIV, ToolGroup.WEB_SEARCH)

  from src.infrastructure.tools.registry import RESEARCHER_TOOLS
"""
from __future__ import annotations
from enum import Enum
from typing import List

from langchain_core.tools import BaseTool


class ToolGroup(str, Enum):
    # ── 原子工具 ──────────────────────────────────────────────────────────────
    WEB_SEARCH = "web_search"      # Tavily + Jina 网页爬取
    ARXIV = "arxiv"                # ArXiv 学术论文搜索
    FILESYSTEM = "filesystem"      # MCP filesystem（运行时注入）

    # ── 组合工具包（角色预设）────────────────────────────────────────────────
    RESEARCH = "research"          # WEB_SEARCH + ARXIV
    CRITIQUE = "critique"          # WEB_SEARCH + ARXIV（与 RESEARCH 相同，语义分离）


# 运行时 MCP 工具存储（由 inject_mcp_tools() 填充）
_mcp_registry: dict[str, List[BaseTool]] = {
    ToolGroup.FILESYSTEM: [],
}


def inject_mcp_tools(group: ToolGroup, tools: List[BaseTool]) -> None:
    """在 FastAPI lifespan 中将 MCP 工具注入指定 ToolGroup。"""
    _mcp_registry[group] = tools


def get_tools(*groups: ToolGroup) -> List[BaseTool]:
    """
    按 ToolGroup 获取工具列表，自动去重（保持插入顺序）。

    惰性导入：避免在模块顶层引入 crawl_search / search_arxiv 触发循环导入。
    """
    from src.infrastructure.tools.crawl_search import crawl_search
    from src.infrastructure.tools.arxiv_search import search_arxiv

    _atom_map: dict[ToolGroup, List[BaseTool]] = {
        ToolGroup.WEB_SEARCH: [crawl_search],
        ToolGroup.ARXIV: [search_arxiv],
        ToolGroup.FILESYSTEM: list(_mcp_registry.get(ToolGroup.FILESYSTEM, [])),
        ToolGroup.RESEARCH: [crawl_search, search_arxiv],
        ToolGroup.CRITIQUE: [crawl_search, search_arxiv],
    }

    seen: set[str] = set()
    result: List[BaseTool] = []
    for group in groups:
        for tool in _atom_map.get(group, []):
            if tool.name not in seen:
                seen.add(tool.name)
                result.append(tool)
    return result


# ── 预定义角色工具包（惰性属性风格，避免模块加载时副作用）────────────────────

def _researcher_tools() -> List[BaseTool]:
    return get_tools(ToolGroup.RESEARCH)

def _critic_tools() -> List[BaseTool]:
    return get_tools(ToolGroup.CRITIQUE)

def _writer_tools() -> List[BaseTool]:
    return get_tools(ToolGroup.FILESYSTEM)


# 延迟绑定属性，在首次 import 后调用时才实际加载
RESEARCHER_TOOLS: List[BaseTool] = []
CRITIC_TOOLS: List[BaseTool] = []
WRITER_TOOLS: List[BaseTool] = []


def _init_role_tools() -> None:
    """在应用启动（lifespan）后调用，确保工具完整初始化。"""
    global RESEARCHER_TOOLS, CRITIC_TOOLS, WRITER_TOOLS
    RESEARCHER_TOOLS = _researcher_tools()
    CRITIC_TOOLS = _critic_tools()
    WRITER_TOOLS = _writer_tools()
