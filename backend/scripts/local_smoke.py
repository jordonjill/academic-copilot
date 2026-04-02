from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure `src` package is importable when running as:
# `uv run python scripts/local_smoke.py`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.interfaces.api.service import (
    create_copilot,
    get_config_registry,
    reload_runtime_config,
    reload_tools_config,
)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_json_like(data: Any) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"- {key}: {value}")
        return
    print(data)


def check_runtime_config() -> bool:
    _print_header("Runtime Config")
    report = reload_runtime_config()
    failed = report.get("failed", [])
    loaded = report.get("loaded", {})
    print(f"config_version={report.get('config_version')}")
    print(
        "loaded: llms=%s agents=%s workflows=%s"
        % (
            len(loaded.get("llms", [])),
            len(loaded.get("agents", [])),
            len(loaded.get("workflows", [])),
        )
    )
    if failed:
        print(f"failed_count={len(failed)}")
        for item in failed:
            print(f"- {item}")
        return False
    print("failed_count=0")
    return True


async def check_tools_reload() -> bool:
    _print_header("Tools Reload")
    report = await reload_tools_config()
    failed = report.get("failed", [])
    print(f"version={report.get('version')}")
    print(
        "loaded_tools=%s loaded_servers=%s"
        % (
            len(report.get("loaded_tools", [])),
            len(report.get("loaded_servers", [])),
        )
    )
    if failed:
        print(f"failed_count={len(failed)}")
        for item in failed:
            print(f"- {item}")
        return False
    print("failed_count=0")
    return True


def check_health() -> bool:
    _print_header("Service Health")
    health = create_copilot().health_check()
    _print_json_like(health)
    return health.get("status") == "healthy"


async def check_chat_once(
    *,
    message: str,
    user_id: str,
    session_id: str,
    workflow_id: str | None = None,
) -> bool:
    _print_header("Chat")
    started = time.perf_counter()
    try:
        result = await create_copilot().chat_async(
            user_message=message,
            user_id=user_id,
            session_id=session_id,
            workflow_id=workflow_id,
        )
    except Exception as exc:  # pragma: no cover - smoke script
        print(f"chat_error={type(exc).__name__}: {exc}")
        return False
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    print(f"duration_ms={duration_ms}")
    print(f"success={result.get('success')}")
    print(f"type={result.get('type')}")
    text = result.get("message")
    if isinstance(text, str):
        print(f"message={text[:400]}")
    return bool(result.get("success"))


async def check_chat_two_turns(*, user_id: str, session_id: str) -> bool:
    _print_header("Chat Two Turns (same session)")
    first = await check_chat_once(
        message="hello",
        user_id=user_id,
        session_id=session_id,
    )
    second = await check_chat_once(
        message="summarize what I just said in one sentence",
        user_id=user_id,
        session_id=session_id,
    )
    return first and second


async def check_workflow(*, user_id: str, session_id: str) -> bool:
    _print_header("Workflow Chat")
    registry = get_config_registry()
    workflow_ids = list(registry.workflows.keys())
    if not workflow_ids:
        print("no workflows configured, skipped")
        return True
    workflow_id = workflow_ids[0]
    print(f"workflow_id={workflow_id}")
    return await check_chat_once(
        message="please start workflow and produce a brief output",
        user_id=user_id,
        session_id=session_id,
        workflow_id=workflow_id,
    )


async def run(args: argparse.Namespace) -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    all_ok = True
    if args.mode in {"config", "all"}:
        all_ok = check_runtime_config() and all_ok

    if args.reload_tools:
        all_ok = await check_tools_reload() and all_ok

    if args.mode in {"health", "all"}:
        all_ok = check_health() and all_ok

    if args.mode in {"chat", "all"}:
        all_ok = await check_chat_once(
            message=args.message,
            user_id=args.user_id,
            session_id=args.session_id,
        ) and all_ok

    if args.two_turns:
        all_ok = await check_chat_two_turns(
            user_id=args.user_id,
            session_id=f"{args.session_id}-2turns",
        ) and all_ok

    if args.workflow:
        all_ok = await check_workflow(
            user_id=args.user_id,
            session_id=f"{args.session_id}-workflow",
        ) and all_ok

    _print_header("Result")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local smoke checks without starting FastAPI server.",
    )
    parser.add_argument(
        "--mode",
        choices=("config", "health", "chat", "all"),
        default="all",
        help="Which base check set to run.",
    )
    parser.add_argument(
        "--message",
        default="hello",
        help="Message for chat smoke test.",
    )
    parser.add_argument(
        "--user-id",
        default="local-smoke-user",
        help="User id used for chat tests.",
    )
    parser.add_argument(
        "--session-id",
        default=f"local-smoke-{int(time.time())}",
        help="Session id used for chat tests.",
    )
    parser.add_argument(
        "--reload-tools",
        action="store_true",
        help="Also run tools reload check.",
    )
    parser.add_argument(
        "--two-turns",
        action="store_true",
        help="Run two-turn chat in the same session.",
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Run one workflow-id chat test using the first configured workflow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/1.0")
    raise SystemExit(asyncio.run(run(parse_args())))
