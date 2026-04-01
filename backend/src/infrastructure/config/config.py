from __future__ import annotations

import os


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


# ===== STM/LTM =====
STM_TOKEN_THRESHOLD = _env_int("STM_TOKEN_THRESHOLD", 6000, minimum=1)
STM_KEEP_RECENT = _env_int("STM_KEEP_RECENT", 6, minimum=0)

# Memory pipeline switch:
# When enabled, chat requests load working context from SQLite and persist
# each completed turn through STM/LTM pipeline.
MEMORY_PIPELINE_ENABLED = _env_bool("MEMORY_PIPELINE_ENABLED", True)

# ===== Persistence paths =====
DATA_DIR = os.getenv("DATA_DIR", "data")
USERS_DIR = os.getenv("USERS_DIR", "data/users")
CONVERSATION_DB = os.getenv("CONVERSATION_DB", "data/conversations.db")
