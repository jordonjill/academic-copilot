from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Ensure `src` package is importable when running as:
# `uv run python scripts/local_cli.py`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.interfaces.api.service import create_copilot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local CLI chat runner (no FastAPI server, no frontend).",
    )
    parser.add_argument(
        "--user-id",
        default="local-cli-user",
        help="Runtime user_id for memory/session scoping.",
    )
    parser.add_argument(
        "--session-id",
        default=f"local-cli-{uuid.uuid4().hex[:8]}",
        help="Session id used across turns in this CLI run.",
    )
    parser.add_argument(
        "--workflow-id",
        default=None,
        help="Optional fixed workflow_id for all turns.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Run one-shot message then exit.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print runtime metadata for each turn.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print step/agent/tool trace for each turn.",
    )
    return parser


def _print_banner(user_id: str, session_id: str, workflow_id: Optional[str]) -> None:
    print("Academic Copilot Local CLI")
    print(f"user_id={user_id}")
    print(f"session_id={session_id}")
    print(f"workflow_id={workflow_id or '<dynamic>'}")
    print("Commands: :exit | :workflow <id> | :workflow off | :trace on|off")


def _extract_text(result: dict) -> str:
    message = result.get("message")
    if isinstance(message, str) and message.strip():
        return message
    data = result.get("data")
    if isinstance(data, dict):
        output = data.get("output")
        if isinstance(output, dict):
            final_text = output.get("final_text")
            if isinstance(final_text, str) and final_text.strip():
                return final_text
    return "<empty reply>"


def _short(value: Any, limit: int = 220) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _print_trace(result: dict[str, Any], events: list[dict[str, Any]]) -> None:
    data = result.get("data")
    runtime: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    if isinstance(data, dict):
        runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
        artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}

    print("[trace] runtime mode=%s workflow_id=%s step_count=%s loop_count=%s"
          % (
              runtime.get("mode"),
              runtime.get("workflow_id"),
              runtime.get("step_count"),
              runtime.get("loop_count"),
          ))

    step_events = [evt for evt in events if isinstance(evt, dict) and evt.get("type") == "step"]
    if not step_events:
        print("[trace] no step events (likely direct supervisor reply)")
    else:
        print(f"[trace] step_events={len(step_events)}")
        for evt in step_events:
            print(
                "[step %s] node=%s agent=%s next=%s reason=%s"
                % (
                    evt.get("step_number"),
                    evt.get("node_name"),
                    evt.get("agent_id"),
                    evt.get("next_node"),
                    evt.get("supervisor_reason"),
                )
            )
            tool_outputs = evt.get("tool_outputs")
            if isinstance(tool_outputs, list) and tool_outputs:
                print(f"  tools_used={len(tool_outputs)}")
                for idx, item in enumerate(tool_outputs, start=1):
                    print(f"  tool_output[{idx}]={_short(item)}")

    shared = artifacts.get("shared")
    if not isinstance(shared, dict) or not shared:
        print("[trace] artifacts.shared is empty")
        return
    print(f"[trace] artifacts.shared agents={len(shared)}")
    for agent_id, item in shared.items():
        if not isinstance(item, dict):
            print(f"[artifact] agent={agent_id} value={_short(item)}")
            continue
        node = item.get("node")
        output_text = item.get("output_text")
        parsed = item.get("parsed")
        tool_outputs = item.get("tool_outputs")
        tool_count = len(tool_outputs) if isinstance(tool_outputs, list) else 0
        parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else []
        print(f"[artifact] agent={agent_id} node={node} tools={tool_count} parsed_keys={parsed_keys}")
        if output_text:
            print(f"  output={_short(output_text)}")


async def _run_turn(
    *,
    copilot,
    user_id: str,
    session_id: str,
    message: str,
    workflow_id: Optional[str],
    show_metadata: bool,
    trace: bool,
) -> bool:
    started = time.perf_counter()
    events: list[dict[str, Any]] = []

    async def _capture_event(payload: dict[str, Any]) -> None:
        if isinstance(payload, dict):
            events.append(payload)

    try:
        result = await copilot.chat_async(
            user_message=message,
            user_id=user_id,
            session_id=session_id,
            workflow_id=workflow_id,
            websocket_send=_capture_event if trace else None,
        )
    except Exception as exc:  # pragma: no cover - local helper
        print(f"[error] {type(exc).__name__}: {exc}")
        if trace:
            print("[trace] partial trace before failure:")
            _print_trace({}, events)
        return False
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    text = _extract_text(result)
    print(f"\nassistant> {text}\n")
    if show_metadata:
        print(
            "[meta] success=%s type=%s latency_ms=%s"
            % (result.get("success"), result.get("type"), latency_ms)
        )
    if trace:
        _print_trace(result, events)
    return bool(result.get("success"))


async def _run_repl(args: argparse.Namespace) -> int:
    load_dotenv(BACKEND_ROOT / ".env")
    os.environ.setdefault("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/1.0")

    copilot = create_copilot()
    workflow_id = args.workflow_id
    trace_enabled = args.trace
    _print_banner(args.user_id, args.session_id, workflow_id)
    print(f"trace={'on' if trace_enabled else 'off'}")

    if args.message:
        ok = await _run_turn(
            copilot=copilot,
            user_id=args.user_id,
            session_id=args.session_id,
            message=args.message,
            workflow_id=workflow_id,
            show_metadata=args.show_metadata,
            trace=trace_enabled,
        )
        return 0 if ok else 1

    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not raw:
            continue
        if raw in {":exit", ":quit"}:
            print("bye")
            return 0
        if raw.startswith(":workflow"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 1 or parts[1].strip().lower() in {"off", "none"}:
                workflow_id = None
                print("workflow_id=<dynamic>")
            else:
                workflow_id = parts[1].strip()
                print(f"workflow_id={workflow_id}")
            continue
        if raw.startswith(":trace"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                print(f"trace={'on' if trace_enabled else 'off'}")
            else:
                value = parts[1].strip().lower()
                trace_enabled = value in {"1", "true", "yes", "on"}
                print(f"trace={'on' if trace_enabled else 'off'}")
            continue

        await _run_turn(
            copilot=copilot,
            user_id=args.user_id,
            session_id=args.session_id,
            message=raw,
            workflow_id=workflow_id,
            show_metadata=args.show_metadata,
            trace=trace_enabled,
        )


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run_repl(args))


if __name__ == "__main__":
    raise SystemExit(main())
