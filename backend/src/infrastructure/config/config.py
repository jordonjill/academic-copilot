from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default_relative: str) -> str:
    backend_root = Path(__file__).resolve().parents[3]
    raw = os.getenv(name)
    value = raw.strip() if raw is not None else ""
    base = backend_root
    if not value:
        return str((base / default_relative).resolve())

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


# ===== STM/LTM =====
STM_TOKEN_THRESHOLD = _env_int("STM_TOKEN_THRESHOLD", 6000, minimum=1)
STM_KEEP_RECENT = _env_int("STM_KEEP_RECENT", 6, minimum=0)
LTM_MAX_ITEMS_PER_CATEGORY = _env_int("LTM_MAX_ITEMS_PER_CATEGORY", 24, minimum=1)
LTM_PAST_TOPICS_MAX_ITEMS = _env_int("LTM_PAST_TOPICS_MAX_ITEMS", 20, minimum=1)
LTM_FACT_MAX_CHARS = _env_int("LTM_FACT_MAX_CHARS", 240, minimum=16)
LTM_MEMORY_MD_MAX_CHARS = _env_int("LTM_MEMORY_MD_MAX_CHARS", 8000, minimum=256)
LTM_SUPERVISOR_PROFILE_MAX_CHARS = _env_int("LTM_SUPERVISOR_PROFILE_MAX_CHARS", 2000, minimum=64)

# Memory pipeline switch:
# When enabled, chat requests load working context from SQLite and persist
# each completed turn through STM/LTM pipeline.
MEMORY_PIPELINE_ENABLED = _env_bool("MEMORY_PIPELINE_ENABLED", True)

# ===== Persistence paths =====
DATA_DIR = _env_path("DATA_DIR", "data")
USERS_DIR = _env_path("USERS_DIR", "data/users")
CONVERSATION_DB = _env_path("CONVERSATION_DB", "data/conversations.db")
