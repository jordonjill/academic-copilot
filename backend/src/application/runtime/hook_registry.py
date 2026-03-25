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
