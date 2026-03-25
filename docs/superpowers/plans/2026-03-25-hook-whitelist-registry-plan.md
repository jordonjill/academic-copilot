# Hook Whitelist Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the hook whitelist registry module plus regression test so unknown hooks fail fast.

**Architecture:** A single whitelist module exposes a global dict and small helper functions; the test imports `resolve_hook` and asserts it raises for unregistered names. No other runtime state is required yet.

**Tech Stack:** Python 3, pytest, uv (task runner).

---

### Task 1: Hook whitelist registry

**Files:**
- Create: `backend/src/application/runtime/hook_registry.py`
- Create: `backend/tests/runtime/test_config_registry.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from src.application.runtime.hook_registry import resolve_hook


def test_unknown_hook_rejected():
    with pytest.raises(KeyError):
        resolve_hook("not_registered_hook")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py::test_unknown_hook_rejected -v`
Expected: FAIL because `hook_registry` module does not yet exist.

- [ ] **Step 3: Write minimal registry implementation**

```python
from typing import Callable, Dict

HOOK_REGISTRY: Dict[str, Callable] = {}


def register_hook(name: str, fn: Callable) -> None:
    if name in HOOK_REGISTRY:
        raise KeyError(f"Hook already registered: {name}")
    HOOK_REGISTRY[name] = fn


def resolve_hook(name: str) -> Callable:
    if name not in HOOK_REGISTRY:
        raise KeyError(f"Hook not registered: {name}")
    return HOOK_REGISTRY[name]
```

- [ ] **Step 4: Run test to verify it now passes**

Run: `cd backend && uv run pytest tests/runtime/test_config_registry.py::test_unknown_hook_rejected -v`
Expected: PASS.

- [ ] **Step 5: Commit the change**

```bash
git add backend/src/application/runtime/hook_registry.py backend/tests/runtime/test_config_registry.py docs/superpowers/specs/2026-03-25-hook-whitelist-registry-design.md docs/superpowers/plans/2026-03-25-hook-whitelist-registry-plan.md
git commit -m "feat(runtime): add whitelist hook registry"
```
