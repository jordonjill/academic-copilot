# Hook Whitelist Registry Design

- Date: 2026-03-25
- Related Task: MVP-1 Task 2 (Hook whitelist registry)

## Objective
Provide a minimal, controlled registry for runtime hooks so that configuration files can reference hooks by name without allowing dynamic module imports. The registry must explicitly reject unknown hook names and keep the surface simple for now.

## Architecture
- Single module at `backend/src/application/runtime/hook_registry.py` exposes `HOOK_REGISTRY`, `register_hook`, and `resolve_hook`.
- `HOOK_REGISTRY` is a module-level `Dict[str, Callable]` that stores the approved hooks.
- Consumers (e.g., agent/workflow runtimes) call `resolve_hook` when they encounter a hook name in YAML. The registry is populated by calling `register_hook` at import time for each supported hook.

## API Design
1. `HOOK_REGISTRY: Dict[str, Callable]`: central map of hook names to callables.
2. `register_hook(name: str, fn: Callable) -> None`: writes a callable into `HOOK_REGISTRY`. Duplicate registrations raise `KeyError` to keep the registry deterministic.
3. `resolve_hook(name: str) -> Callable`: looks up a name and raises `KeyError` with a clear message if the entry is missing.

## Error Handling
- The only failure mode is missing hook names, which raises `KeyError` from `resolve_hook`. This error can bubble up to config reload or runtime execution, ensuring consumers do not execute unknown hooks.
- Duplicate registration attempts also raise `KeyError` so that we do not silently override existing whitelist entries.

## Testing
- Add `backend/tests/runtime/test_config_registry.py` containing `test_unknown_hook_rejected`. The test imports `resolve_hook` and asserts it raises `KeyError` when asked for an unregistered name.
- Tests run via `uv run pytest tests/runtime/test_config_registry.py::test_unknown_hook_rejected -v`.
