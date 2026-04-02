from __future__ import annotations

import os


def read_env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = 0.0,
    inclusive_minimum: bool = False,
) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None:
        if inclusive_minimum:
            if value < minimum:
                return default
        else:
            if value <= minimum:
                return default
    return value


def read_env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value
