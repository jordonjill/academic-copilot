from __future__ import annotations

import hashlib
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque

from fastapi import HTTPException, Request


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _rate_limit_enabled() -> bool:
    return os.getenv("CHAT_RATE_LIMIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _rate_limit_requests() -> int:
    return _env_int("CHAT_RATE_LIMIT_REQUESTS", 60, minimum=1)


def _rate_limit_window_seconds() -> int:
    return _env_int("CHAT_RATE_LIMIT_WINDOW_SECONDS", 60, minimum=1)


@dataclass
class _SlidingWindowLimiter:
    max_requests: int
    window_seconds: int
    _events: dict[str, Deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def check(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.max_requests:
                retry_after = self.window_seconds - (now - events[0])
                return False, max(0.1, retry_after)
            events.append(now)
            return True, 0.0


_LIMITER_CONFIG: tuple[int, int] | None = None
_LIMITER: _SlidingWindowLimiter | None = None
_LIMITER_LOCK = Lock()


def _get_limiter() -> _SlidingWindowLimiter:
    global _LIMITER_CONFIG, _LIMITER
    config = (_rate_limit_requests(), _rate_limit_window_seconds())
    with _LIMITER_LOCK:
        if _LIMITER is None or _LIMITER_CONFIG != config:
            _LIMITER = _SlidingWindowLimiter(max_requests=config[0], window_seconds=config[1])
            _LIMITER_CONFIG = config
        return _LIMITER


def _client_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    auth = request.headers.get("authorization", "").strip()
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else auth
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16] if token else "anon"
    return f"{host}:{token_hash}"


async def enforce_chat_rate_limit(request: Request) -> None:
    if not _rate_limit_enabled():
        return
    limiter = _get_limiter()
    allowed, retry_after = limiter.check(_client_key(request))
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded for /chat",
        headers={"Retry-After": str(int(math.ceil(retry_after)))},
    )
