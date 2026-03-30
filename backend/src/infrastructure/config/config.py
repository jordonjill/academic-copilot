from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ===== Runtime limits =====
MAX_SEARCHES = _env_int("MAX_SEARCHES", 5)
MAX_VALIDATION_ATTEMPTS = _env_int("MAX_VALIDATION_ATTEMPTS", 3)
MAX_TAVILY_SEARCHES = _env_int("MAX_TAVILY_SEARCHES", 10)

# ===== STM/LTM =====
STM_TOKEN_THRESHOLD = _env_int("STM_TOKEN_THRESHOLD", 6000)
STM_KEEP_RECENT = _env_int("STM_KEEP_RECENT", 6)

# Memory pipeline switch:
# When enabled, chat requests load working context from SQLite and persist
# each completed turn through STM/LTM pipeline.
MEMORY_PIPELINE_ENABLED = _env_bool("MEMORY_PIPELINE_ENABLED", True)

# ===== Persistence paths =====
DATA_DIR = os.getenv("DATA_DIR", "data")
USERS_DIR = os.getenv("USERS_DIR", "data/users")
CONVERSATION_DB = os.getenv("CONVERSATION_DB", "data/conversations.db")
