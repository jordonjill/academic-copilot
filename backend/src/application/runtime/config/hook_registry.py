from types import MappingProxyType
from typing import Callable, Mapping


_HOOK_REGISTRY: dict[str, Callable[..., object]] = {}
HOOK_REGISTRY: Mapping[str, Callable[..., object]] = MappingProxyType(_HOOK_REGISTRY)


def register_hook(name: str, fn: Callable[..., object]) -> None:
    if not callable(fn):
        raise TypeError(f"Hook must be callable: {name}")
    if name in _HOOK_REGISTRY:
        raise KeyError(f"Hook already registered: {name}")
    _HOOK_REGISTRY[name] = fn


def resolve_hook(name: str) -> Callable[..., object]:
    if name not in _HOOK_REGISTRY:
        raise KeyError(f"Hook not registered: {name}")
    return _HOOK_REGISTRY[name]
