"""
LTM（长期记忆）事实提取管道。

触发时机：STM 压缩节点中通过 event-loop task 异步调用，不阻塞主流程。

流程：
  1. 将对话主干拼接为文本
  2. LLM with_structured_output → 6 类事实 JSON
  3. Union 去重追加到现有用户画像字段
  4. 序列化为 memory.md 写入 data/users/{user_id}/
  5. 写入 SQLite ltm_facts 表
"""
from __future__ import annotations
import atexit
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import List, Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from src.infrastructure.config.config import MEMORY_PIPELINE_ENABLED, USERS_DIR
from src.infrastructure.config.prompt import LTM_EXTRACTION_PROMPT


_MAX_PAST_TOPICS = 20  # past_topics FIFO 滚动上限
logger = logging.getLogger(__name__)


def _ltm_max_workers() -> int:
    raw = os.getenv("LTM_MAX_WORKERS", "2").strip()
    try:
        workers = int(raw)
    except ValueError:
        workers = 2
    return max(1, workers)


_LTM_EXECUTOR = ThreadPoolExecutor(
    max_workers=_ltm_max_workers(),
    thread_name_prefix="ltm-worker",
)


def _shutdown_ltm_executor() -> None:
    _LTM_EXECUTOR.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_ltm_executor)


def _build_backbone_text(backbone: List[BaseMessage]) -> str:
    parts = []
    for m in backbone:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"{role}: {content[:500]}")
    return "\n".join(parts)


def _load_existing_profile(user_id: str) -> Dict[str, List[str]]:
    """从 memory.md 解析现有 profile 字段（或返回空默认值）。"""
    profile_path = os.path.join(USERS_DIR, user_id, "memory.md")
    default: Dict[str, List[str]] = {
        "research_domains": [],
        "methodologies": [],
        "tools_and_frameworks": [],
        "past_topics": [],
        "writing_preferences": [],
        "custom_facts": [],
    }
    if not os.path.exists(profile_path):
        return default

    import re
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = f.read()

    header_to_key = {
        "## Research Domains": "research_domains",
        "## Preferred Methodologies": "methodologies",
        "## Known Tools & Frameworks": "tools_and_frameworks",
        "## Past Topics": "past_topics",
        "## Writing Preferences": "writing_preferences",
        "## Custom Facts": "custom_facts",
    }
    for header, key in header_to_key.items():
        pattern = rf"{re.escape(header)}\n((?:- .+\n?)*)"
        m = re.search(pattern, raw)
        if m:
            items = [line.lstrip("- ").strip() for line in m.group(1).strip().splitlines()]
            default[key] = [i for i in items if i]

    return default


def _merge_profiles(existing: Dict[str, List[str]], new_facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Union 去重追加，past_topics 保留最近 _MAX_PAST_TOPICS 条。"""
    merged: Dict[str, List[str]] = {}
    for key in existing:
        combined = list(dict.fromkeys(existing.get(key, []) + new_facts.get(key, [])))
        if key == "past_topics" and len(combined) > _MAX_PAST_TOPICS:
            combined = combined[-_MAX_PAST_TOPICS:]
        merged[key] = combined
    return merged


def _write_memory_md(user_id: str, profile: Dict[str, List[str]]) -> str:
    """将 profile 序列化为 memory.md 格式并写入磁盘。"""
    user_dir = os.path.join(USERS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    profile_path = os.path.join(user_dir, "memory.md")

    now = datetime.now(UTC).isoformat()
    lines = [
        "# User Research Profile",
        f"*Last Updated: {now}*",
        "",
        "## Research Domains",
        *[f"- {item}" for item in profile.get("research_domains", [])],
        "",
        "## Preferred Methodologies",
        *[f"- {item}" for item in profile.get("methodologies", [])],
        "",
        "## Known Tools & Frameworks",
        *[f"- {item}" for item in profile.get("tools_and_frameworks", [])],
        "",
        "## Past Topics",
        *[f"- {item}" for item in profile.get("past_topics", [])],
        "",
        "## Writing Preferences",
        *[f"- {item}" for item in profile.get("writing_preferences", [])],
        "",
        "## Custom Facts",
        *[f"- {item}" for item in profile.get("custom_facts", [])],
        "",
    ]
    content = "\n".join(lines)
    with open(profile_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


async def extract_and_update_ltm(
    user_id: str,
    session_id: str,
    backbone: List[BaseMessage],
    llm: BaseLanguageModel,
) -> None:
    """
    异步 LTM 提取入口。
    在 asyncio 事件循环中通过 run_in_executor 调用同步 LLM。
    """
    if not MEMORY_PIPELINE_ENABLED:
        return

    try:
        backbone_text = _build_backbone_text(backbone)
        if not backbone_text.strip():
            return

        loop = asyncio.get_running_loop()

        # 在线程池中运行同步 LLM 调用
        def _call_llm():
            chain = LTM_EXTRACTION_PROMPT | llm
            response = chain.invoke({"conversation_backbone": backbone_text})
            return response.content if hasattr(response, "content") else str(response)

        raw_json = await loop.run_in_executor(_LTM_EXECUTOR, _call_llm)

        # 解析 JSON
        try:
            new_facts: Dict[str, List[str]] = json.loads(raw_json)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', raw_json, re.DOTALL)
            if not m:
                logger.warning("[LTM] JSON parse failed: %s", raw_json[:200])
                return
            new_facts = json.loads(m.group())

        # 合并 + 写入 memory.md
        existing = _load_existing_profile(user_id)
        merged = _merge_profiles(existing, new_facts)
        _write_memory_md(user_id, merged)

        # 写入 SQLite
        from src.infrastructure.memory.sqlite_store import SQLiteStore
        store = SQLiteStore()
        for fact_type, items in new_facts.items():
            for item in items:
                if item.strip():
                    store.save_ltm_fact(user_id, session_id, fact_type, item)

        logger.info("[LTM] Updated memory.md for user %s", user_id)

    except Exception as e:
        logger.exception("[LTM] Extraction failed for user %s: %s", user_id, e)
