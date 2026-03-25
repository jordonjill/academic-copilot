import pytest

from src.application.runtime import hook_registry
from src.application.runtime.hook_registry import HOOK_REGISTRY, register_hook, resolve_hook


def test_unknown_hook_rejected():
    with pytest.raises(KeyError):
        resolve_hook("not_registered_hook")


@pytest.fixture(autouse=True)
def clear_hooks():
    hook_registry._HOOK_REGISTRY.clear()
    yield
    hook_registry._HOOK_REGISTRY.clear()


def test_register_then_resolve():
    HOOK_REGISTRY  # ensure the read-only view is accessible
    def sample_hook():
        return "ok"

    register_hook("test-hook", sample_hook)
    assert resolve_hook("test-hook") is sample_hook


def test_duplicate_registration_rejected():
    def noop():
        return None

    register_hook("dup-hook", noop)
    with pytest.raises(KeyError):
        register_hook("dup-hook", noop)
