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

# Memory pipeline switch:
# When enabled, chat requests load working context from SQLite and persist
# each completed turn through STM/LTM pipeline.
MEMORY_PIPELINE_ENABLED = _env_bool("MEMORY_PIPELINE_ENABLED", True)

# ===== Persistence paths =====
DATA_DIR = _env_path("DATA_DIR", "data")
USERS_DIR = _env_path("USERS_DIR", "data/users")
CONVERSATION_DB = _env_path("CONVERSATION_DB", "data/conversations.db")
