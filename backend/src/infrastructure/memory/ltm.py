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
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List, Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from src.infrastructure.config.config import (
    LTM_FACT_MAX_CHARS,
    LTM_MAX_ITEMS_PER_CATEGORY,
    LTM_MEMORY_MD_MAX_CHARS,
    LTM_PAST_TOPICS_MAX_ITEMS,
    LTM_SUPERVISOR_PROFILE_MAX_CHARS,
    MEMORY_PIPELINE_ENABLED,
    USERS_DIR,
)
from src.infrastructure.config.prompt import LTM_EXTRACTION_PROMPT
from src.infrastructure.observability.langfuse_observability import (
    OP_MEMORY_LTM_EXTRACTION,
    build_langchain_config,
    operation_metadata,
)


logger = logging.getLogger(__name__)
_USERS_ROOT = Path(USERS_DIR).expanduser().resolve()
_USER_ID_PATTERN = re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9_@.-]{1,128}$")
_PROFILE_KEYS = (
    "research_domains",
    "methodologies",
    "tools_and_frameworks",
    "past_topics",
    "writing_preferences",
    "custom_facts",
)
_TRIM_PRIORITY = (
    "custom_facts",
    "past_topics",
    "tools_and_frameworks",
    "methodologies",
    "research_domains",
    "writing_preferences",
)
_SIMILARITY_JACCARD_THRESHOLD = 0.82
_SIMILARITY_MIN_TOKEN_COUNT = 4
_SIMILARITY_MIN_SUBSTRING_CHARS = 18


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
    # Best effort to preserve final LTM writes during normal process exit.
    _LTM_EXECUTOR.shutdown(wait=True, cancel_futures=False)


atexit.register(_shutdown_ltm_executor)


def _build_backbone_text(backbone: List[BaseMessage]) -> str:
    parts = []
    for m in backbone:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"{role}: {content[:500]}")
    return "\n".join(parts)


def _safe_user_memory_path(user_id: str) -> Path:
    normalized_user_id = (user_id or "").strip()
    if not _USER_ID_PATTERN.fullmatch(normalized_user_id):
        raise ValueError(f"Invalid user_id path component: {user_id}")
    try:
        candidate = (_USERS_ROOT / normalized_user_id / "memory.md").resolve()
        candidate.relative_to(_USERS_ROOT)
    except ValueError as exc:
        raise ValueError(f"user_id '{user_id}' resolves outside user root") from exc
    return candidate


def _load_existing_profile(user_id: str) -> Dict[str, List[str]]:
    """从 memory.md 解析现有 profile 字段（或返回空默认值）。"""
    profile_path = _safe_user_memory_path(user_id)
    default: Dict[str, List[str]] = _empty_profile()
    if not profile_path.exists():
        return default

    import re
    with profile_path.open("r", encoding="utf-8") as f:
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
            default[key] = _normalize_items(items, max_items=_category_cap(key))

    return default


def _merge_profiles(existing: Dict[str, List[str]], new_facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge new facts into existing profile with semantic de-duplication and caps."""
    merged, _ = _merge_profiles_with_delta(existing, new_facts)
    return merged


def _merge_profiles_with_delta(
    existing: Dict[str, List[str]],
    new_facts: Dict[str, List[str]],
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Merge with semantic de-duplication and return delta facts that were newly added/updated.

    Delta is used to avoid repeatedly writing near-duplicate facts into SQLite.
    """
    merged: Dict[str, List[str]] = _empty_profile()
    delta: Dict[str, List[str]] = _empty_profile()
    for key in _PROFILE_KEYS:
        existing_items = existing.get(key, [])
        new_items = new_facts.get(key, [])
        if not isinstance(existing_items, list):
            existing_items = []
        if not isinstance(new_items, list):
            new_items = []
        merged_items, delta_items = _merge_item_lists(
            existing_items,
            new_items,
            max_items=_category_cap(key),
        )
        merged[key] = merged_items
        delta[key] = delta_items
    return merged, delta


def _write_memory_md(user_id: str, profile: Dict[str, List[str]]) -> str:
    """将 profile 序列化为 memory.md 格式并写入磁盘。"""
    profile_path = _safe_user_memory_path(user_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = _normalize_profile(profile)
    now = datetime.now(UTC).isoformat()
    content = _render_memory_md(cleaned, timestamp_iso=now)
    while len(content) > LTM_MEMORY_MD_MAX_CHARS and _profile_has_any_item(cleaned):
        if not _drop_oldest_by_priority(cleaned):
            break
        content = _render_memory_md(cleaned, timestamp_iso=now)
    with profile_path.open("w", encoding="utf-8") as f:
        f.write(content)
    return content


def load_ltm_profile_for_supervisor(user_id: str) -> str:
    """Load compact LTM profile text for supervisor prompt injection."""
    if not MEMORY_PIPELINE_ENABLED:
        return ""
    try:
        profile = _load_existing_profile(user_id)
    except Exception:
        logger.exception("[LTM] Failed to load memory profile for user %s", user_id)
        return ""

    compact = {key: value for key, value in profile.items() if isinstance(value, list) and value}
    if not compact:
        return ""

    trimmed = {key: list(value) for key, value in compact.items()}
    text = json.dumps(trimmed, ensure_ascii=False)
    while len(text) > LTM_SUPERVISOR_PROFILE_MAX_CHARS and _profile_has_any_item(trimmed):
        if not _drop_oldest_by_priority(trimmed):
            break
        text = json.dumps({k: v for k, v in trimmed.items() if v}, ensure_ascii=False)

    if len(text) > LTM_SUPERVISOR_PROFILE_MAX_CHARS:
        text = text[:LTM_SUPERVISOR_PROFILE_MAX_CHARS]
    return text


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
            config = build_langchain_config(
                {
                    "run_name": OP_MEMORY_LTM_EXTRACTION,
                    "metadata": operation_metadata(
                        OP_MEMORY_LTM_EXTRACTION,
                        operation_type="memory",
                        memory_stage="ltm_extraction",
                        session_id=session_id,
                        user_id=user_id,
                    ),
                    "tags": ["memory", "ltm"],
                }
            )
            try:
                response = chain.invoke({"conversation_backbone": backbone_text}, config=config)
            except TypeError:
                response = chain.invoke({"conversation_backbone": backbone_text})
            return response.content if hasattr(response, "content") else str(response)

        raw_json = await loop.run_in_executor(_LTM_EXECUTOR, _call_llm)

        # 解析 JSON
        try:
            new_facts: Dict[str, List[str]] = json.loads(raw_json)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw_json, re.DOTALL)
            if not m:
                logger.warning("[LTM] JSON parse failed: %s", raw_json[:200])
                return
            try:
                new_facts = json.loads(m.group())
            except json.JSONDecodeError:
                logger.warning("[LTM] JSON parse failed after regex extraction: %s", raw_json[:200])
                return

        # 合并 + 写入 memory.md
        existing = _load_existing_profile(user_id)
        merged, delta = _merge_profiles_with_delta(existing, new_facts)
        _write_memory_md(user_id, merged)

        # 写入 SQLite
        from src.infrastructure.memory.sqlite_store import SQLiteStore
        store = SQLiteStore()
        for fact_type, items in delta.items():
            for item in items:
                if item.strip():
                    store.save_ltm_fact(user_id, session_id, fact_type, item)

        logger.info("[LTM] Updated memory.md for user %s", user_id)

    except Exception as e:
        logger.exception("[LTM] Extraction failed for user %s: %s", user_id, e)


def _empty_profile() -> Dict[str, List[str]]:
    return {key: [] for key in _PROFILE_KEYS}


def _category_cap(key: str) -> int:
    if key == "past_topics":
        return LTM_PAST_TOPICS_MAX_ITEMS
    return LTM_MAX_ITEMS_PER_CATEGORY


def _normalize_fact_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > LTM_FACT_MAX_CHARS:
        text = text[:LTM_FACT_MAX_CHARS].rstrip()
    return text


def _normalize_items(items: List[Any], *, max_items: int) -> List[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _normalize_fact_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    if len(normalized) > max_items:
        normalized = normalized[-max_items:]
    return normalized


def _semantic_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _tokenize_for_similarity(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) >= 2}


def _is_semantic_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a.casefold() == b.casefold():
        return True

    key_a = _semantic_key(a)
    key_b = _semantic_key(b)
    if key_a and key_a == key_b:
        return True

    lower_a = a.casefold()
    lower_b = b.casefold()
    if len(lower_a) <= len(lower_b):
        shorter, longer = lower_a, lower_b
    else:
        shorter, longer = lower_b, lower_a
    if len(shorter) >= _SIMILARITY_MIN_SUBSTRING_CHARS and shorter in longer:
        return True

    tokens_a = _tokenize_for_similarity(a)
    tokens_b = _tokenize_for_similarity(b)
    if min(len(tokens_a), len(tokens_b)) < _SIMILARITY_MIN_TOKEN_COUNT:
        return False
    union = tokens_a | tokens_b
    if not union:
        return False
    jaccard = len(tokens_a & tokens_b) / float(len(union))
    return jaccard >= _SIMILARITY_JACCARD_THRESHOLD


def _find_semantic_match(items: list[str], candidate: str) -> int | None:
    for idx, existing in enumerate(items):
        if _is_semantic_duplicate(existing, candidate):
            return idx
    return None


def _prefer_richer_text(old_text: str, new_text: str) -> str:
    if old_text.casefold() == new_text.casefold():
        return old_text
    old_tokens = _tokenize_for_similarity(old_text)
    new_tokens = _tokenize_for_similarity(new_text)
    old_score = len(old_text) + (6 * len(old_tokens))
    new_score = len(new_text) + (6 * len(new_tokens))
    if new_score > old_score:
        return new_text
    if new_score == old_score and len(new_text) >= len(old_text):
        return new_text
    return old_text


def _semantic_dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if _find_semantic_match(deduped, item) is None:
            deduped.append(item)
            continue
        match_idx = _find_semantic_match(deduped, item)
        if match_idx is not None:
            deduped[match_idx] = _prefer_richer_text(deduped[match_idx], item)
    return deduped


def _merge_item_lists(
    existing_items: list[Any],
    new_items: list[Any],
    *,
    max_items: int,
) -> tuple[list[str], list[str]]:
    baseline = _normalize_items(existing_items, max_items=max(1, max_items * 4))
    merged = list(_semantic_dedupe_preserve_order(baseline))
    delta: list[str] = []

    for raw in new_items:
        candidate = _normalize_fact_text(raw)
        if not candidate:
            continue
        match_idx = _find_semantic_match(merged, candidate)
        if match_idx is None:
            merged.append(candidate)
            delta.append(candidate)
            continue
        improved = _prefer_richer_text(merged[match_idx], candidate)
        if improved != merged[match_idx]:
            merged[match_idx] = improved
            delta.append(improved)

    merged = _semantic_dedupe_preserve_order(merged)
    if len(merged) > max_items:
        merged = merged[-max_items:]

    # Keep delta compact and only facts represented in final merged profile.
    compact_delta: list[str] = []
    for item in _semantic_dedupe_preserve_order(delta):
        if _find_semantic_match(merged, item) is not None:
            compact_delta.append(item)
    return merged, compact_delta


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized = _empty_profile()
    for key in _PROFILE_KEYS:
        raw_items = profile.get(key, [])
        if not isinstance(raw_items, list):
            raw_items = []
        normalized[key] = _normalize_items(raw_items, max_items=_category_cap(key))
    return normalized


def _render_memory_md(profile: Dict[str, List[str]], *, timestamp_iso: str) -> str:
    lines = [
        "# User Research Profile",
        f"*Last Updated: {timestamp_iso}*",
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
    return "\n".join(lines)


def _profile_has_any_item(profile: Dict[str, List[str]]) -> bool:
    for key in _PROFILE_KEYS:
        items = profile.get(key, [])
        if isinstance(items, list) and items:
            return True
    return False


def _drop_oldest_by_priority(profile: Dict[str, List[str]]) -> bool:
    for key in _TRIM_PRIORITY:
        items = profile.get(key, [])
        if isinstance(items, list) and items:
            items.pop(0)
            return True
    return False
