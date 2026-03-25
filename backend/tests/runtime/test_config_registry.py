import pytest

from src.application.runtime.hook_registry import resolve_hook


def test_unknown_hook_rejected():
    with pytest.raises(KeyError):
        resolve_hook("not_registered_hook")
