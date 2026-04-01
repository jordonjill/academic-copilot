from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.memory import ltm as ltm_module


def test_safe_user_memory_path_rejects_path_traversal(tmp_path, monkeypatch):
    users_root = (tmp_path / "users").resolve()
    monkeypatch.setattr(ltm_module, "_USERS_ROOT", users_root)
    with pytest.raises(ValueError):
        ltm_module._safe_user_memory_path("../../etc/passwd")


def test_safe_user_memory_path_stays_within_users_root(tmp_path, monkeypatch):
    users_root = (tmp_path / "users").resolve()
    monkeypatch.setattr(ltm_module, "_USERS_ROOT", users_root)
    safe_path = ltm_module._safe_user_memory_path("alice")
    assert str(safe_path).startswith(str(users_root))
    assert safe_path == users_root / "alice" / "memory.md"
