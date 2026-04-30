from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|access[_-]?key|authorization|token|secret|password|cookie)",
    re.I,
)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d .-]{7,}\d)\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

_CURRENT_CALLBACKS: ContextVar[tuple[Any, ...]] = ContextVar(
    "academic_copilot_langchain_callbacks",
    default=(),
)
_CURRENT_METADATA: ContextVar[dict[str, Any]] = ContextVar(
    "academic_copilot_langchain_metadata",
    default={},
)
_CURRENT_TAGS: ContextVar[tuple[str, ...]] = ContextVar(
    "academic_copilot_langchain_tags",
    default=(),
)

_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_CLIENT_LOCK = threading.Lock()
_WARNED_MESSAGES: set[str] = set()
OP_CHAT_TURN = "chat.turn"
OP_SUPERVISOR_DECIDE = "supervisor.decide"
OP_SUPERVISOR_FINALIZE = "supervisor.finalize"
OP_AGENT_CHAIN = "agent.chain"
OP_AGENT_REACT = "agent.react"
OP_MEMORY_STM_COMPRESSION = "memory.stm_compression"
OP_MEMORY_LTM_EXTRACTION = "memory.ltm_extraction"
_LANGFUSE_METADATA_VALUE_MAX_CHARS = 200


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    logger.warning(message, *args)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        return max(minimum, default)


def langfuse_enabled() -> bool:
    return _env_flag("LANGFUSE_ENABLED", False)


def token_observability_enabled() -> bool:
    return _env_flag("TOKEN_USAGE_OBSERVABILITY_ENABLED", True)


def _langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()) and bool(
        os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    )


def _max_masked_string_chars() -> int:
    return _env_int("LANGFUSE_MAX_STRING_CHARS", 4000, minimum=128)


def _mask_string(value: str) -> str:
    masked = _BEARER_PATTERN.sub("Bearer ***REDACTED***", value)
    masked = _OPENAI_KEY_PATTERN.sub("***REDACTED_KEY***", masked)
    masked = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", masked)
    masked = _CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", masked)
    if _env_flag("LANGFUSE_MASK_PHONE_NUMBERS", True):
        masked = _PHONE_PATTERN.sub("[REDACTED_PHONE]", masked)
    max_chars = _max_masked_string_chars()
    if len(masked) > max_chars:
        return masked[:max_chars] + "...<truncated>"
    return masked


def mask_data(data: Any, **_: Any) -> Any:
    """Best-effort client-side masking before data leaves the process."""
    if isinstance(data, str):
        return _mask_string(data)
    if isinstance(data, Mapping):
        result: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key)
            if _SENSITIVE_KEY_PATTERN.search(key_text):
                result[key_text] = "***REDACTED***"
            else:
                result[key_text] = mask_data(value)
        return result
    if isinstance(data, list):
        return [mask_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(mask_data(item) for item in data)
    return data


def langfuse_metadata(metadata: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return metadata safe for Langfuse v4 propagated attributes."""
    if not isinstance(metadata, Mapping):
        return {}

    result: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip()
        if not key:
            continue
        result[key] = _coerce_langfuse_metadata_value(raw_value)
    return result


def operation_metadata(
    operation: str,
    *,
    operation_type: str,
    **metadata: Any,
) -> dict[str, str]:
    return langfuse_metadata(
        {
            "operation": operation,
            "operation_type": operation_type,
            **metadata,
        }
    )


def _coerce_langfuse_metadata_value(value: Any) -> str:
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (str, int, float)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        except Exception:
            text = str(value)
    if len(text) > _LANGFUSE_METADATA_VALUE_MAX_CHARS:
        return text[: _LANGFUSE_METADATA_VALUE_MAX_CHARS - 14] + "...<truncated>"
    return text


def _get_langfuse_client() -> Any | None:
    if not langfuse_enabled():
        return None
    if not _langfuse_configured():
        _warn_once(
            "langfuse_missing_keys",
            "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not configured; "
            "Langfuse export is disabled for this process.",
        )
        return None

    global _LANGFUSE_CLIENT
    with _LANGFUSE_CLIENT_LOCK:
        if _LANGFUSE_CLIENT is not None:
            return _LANGFUSE_CLIENT
        try:
            from langfuse import Langfuse
        except Exception as exc:
            _warn_once(
                "langfuse_import_failed",
                "LANGFUSE_ENABLED=true but the langfuse package could not be imported: %s",
                exc,
            )
            return None

        try:
            _LANGFUSE_CLIENT = Langfuse(mask=mask_data)
        except Exception as exc:
            _warn_once(
                "langfuse_init_failed",
                "Failed to initialize Langfuse client; Langfuse export is disabled: %s",
                exc,
            )
            return None
        return _LANGFUSE_CLIENT


def _create_langfuse_handler() -> Any | None:
    if not langfuse_enabled() or not _langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        try:
            from langfuse.callback import CallbackHandler
        except Exception as exc:
            _warn_once(
                "langfuse_callback_import_failed",
                "Langfuse is configured but its LangChain CallbackHandler could not be imported: %s",
                exc,
            )
            return None
    try:
        return CallbackHandler()
    except Exception as exc:
        _warn_once(
            "langfuse_callback_init_failed",
            "Failed to initialize Langfuse CallbackHandler: %s",
            exc,
        )
        return None


@contextlib.contextmanager
def _propagate_langfuse_attributes(
    *,
    user_id: str,
    session_id: str,
    trace_name: str,
    tags: list[str],
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    if not langfuse_enabled() or not _langfuse_configured():
        yield
        return
    try:
        from langfuse import propagate_attributes
    except Exception:
        yield
        return
    try:
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
            tags=tags,
            metadata=langfuse_metadata(metadata),
        ):
            yield
    except TypeError:
        with propagate_attributes(user_id=user_id, session_id=session_id):
            yield


@contextlib.contextmanager
def langchain_observation_context(
    *,
    callbacks: list[Any] | tuple[Any, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> Iterator[None]:
    parent_callbacks = list(_CURRENT_CALLBACKS.get())
    parent_metadata = dict(_CURRENT_METADATA.get())
    parent_tags = list(_CURRENT_TAGS.get())

    next_callbacks = tuple(parent_callbacks + list(callbacks or []))
    next_metadata = {**parent_metadata, **dict(metadata or {})}
    next_tags = tuple(dict.fromkeys(parent_tags + list(tags or [])))

    callbacks_token = _CURRENT_CALLBACKS.set(next_callbacks)
    metadata_token = _CURRENT_METADATA.set(next_metadata)
    tags_token = _CURRENT_TAGS.set(next_tags)
    try:
        yield
    finally:
        _CURRENT_TAGS.reset(tags_token)
        _CURRENT_METADATA.reset(metadata_token)
        _CURRENT_CALLBACKS.reset(callbacks_token)


def build_langchain_config(
    config: Optional[dict[str, Any]] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    run_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    callbacks = list(_CURRENT_CALLBACKS.get())
    context_metadata = dict(_CURRENT_METADATA.get())
    context_tags = list(_CURRENT_TAGS.get())
    if not callbacks and not context_metadata and not context_tags and not metadata and not tags and not run_name:
        return config

    merged: dict[str, Any] = dict(config or {})

    if callbacks:
        existing_callbacks = merged.get("callbacks")
        if existing_callbacks is None:
            merged["callbacks"] = callbacks
        elif isinstance(existing_callbacks, list):
            merged["callbacks"] = [*existing_callbacks, *callbacks]
        elif isinstance(existing_callbacks, tuple):
            merged["callbacks"] = [*existing_callbacks, *callbacks]
        else:
            merged["callbacks"] = [existing_callbacks, *callbacks]

    existing_metadata = merged.get("metadata")
    merged_metadata: dict[str, Any] = {}
    if isinstance(existing_metadata, Mapping):
        merged_metadata.update(dict(existing_metadata))
    merged_metadata.update(context_metadata)
    if metadata:
        merged_metadata.update(metadata)
    if merged_metadata:
        merged["metadata"] = langfuse_metadata(merged_metadata)

    existing_tags = merged.get("tags")
    merged_tags: list[str] = []
    if isinstance(existing_tags, (list, tuple)):
        merged_tags.extend(str(tag) for tag in existing_tags)
    elif isinstance(existing_tags, str):
        merged_tags.append(existing_tags)
    merged_tags.extend(str(tag) for tag in context_tags)
    if tags:
        merged_tags.extend(str(tag) for tag in tags)
    if merged_tags:
        merged["tags"] = list(dict.fromkeys(merged_tags))

    if run_name and "run_name" not in merged:
        merged["run_name"] = run_name

    return merged


def _coerce_text(value: Any) -> str:
    if isinstance(value, BaseMessage):
        content = value.content
        return content if isinstance(content, str) else str(content)
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    return str(value)


def _try_get_encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_text_tokens(text: str) -> int:
    encoder = _try_get_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _messages_token_estimate(messages: Any) -> int:
    total = 0
    if not isinstance(messages, list):
        return _estimate_text_tokens(str(messages))
    for item in messages:
        if isinstance(item, list):
            total += _messages_token_estimate(item)
            continue
        total += _estimate_text_tokens(_coerce_text(item)) + 4
    return max(1, total)


def _generations_text(response: Any) -> str:
    generations = getattr(response, "generations", None)
    if not isinstance(generations, list):
        return ""
    parts: list[str] = []
    for generation_list in generations:
        if not isinstance(generation_list, list):
            generation_list = [generation_list]
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if message is not None:
                parts.append(_coerce_text(message))
                continue
            text = getattr(generation, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _usage_from_mapping(payload: Mapping[str, Any]) -> dict[str, int]:
    prompt_details = payload.get("prompt_tokens_details")
    completion_details = payload.get("completion_tokens_details")
    input_details = payload.get("input_token_details")
    output_details = payload.get("output_token_details")

    usage = {
        "input_tokens": _as_int(
            payload.get("input_tokens", payload.get("prompt_tokens", payload.get("input", 0)))
        ),
        "output_tokens": _as_int(
            payload.get("output_tokens", payload.get("completion_tokens", payload.get("output", 0)))
        ),
        "total_tokens": _as_int(payload.get("total_tokens", payload.get("total", 0))),
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    if isinstance(prompt_details, Mapping):
        usage["cached_tokens"] += _as_int(
            prompt_details.get("cached_tokens", prompt_details.get("cache_read_input_tokens", 0))
        )
    if isinstance(input_details, Mapping):
        usage["cached_tokens"] += _as_int(
            input_details.get("cache_read", input_details.get("cached_tokens", 0))
        )
    if isinstance(completion_details, Mapping):
        usage["reasoning_tokens"] += _as_int(completion_details.get("reasoning_tokens", 0))
    if isinstance(output_details, Mapping):
        usage["reasoning_tokens"] += _as_int(output_details.get("reasoning", 0))

    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _merge_usage(target: dict[str, int], usage: Mapping[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
        target[key] = target.get(key, 0) + _as_int(usage.get(key, 0))


def _extract_usage_and_model(response: Any) -> tuple[dict[str, int], str | None]:
    candidates: list[Mapping[str, Any]] = []
    model: str | None = None
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        candidates.append(llm_output)
        for key in ("model_name", "model"):
            if isinstance(llm_output.get(key), str):
                model = str(llm_output[key])
        for key in ("token_usage", "usage", "usage_metadata"):
            nested = llm_output.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)

    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        for generation_list in generations:
            if not isinstance(generation_list, list):
                generation_list = [generation_list]
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                usage_metadata = getattr(message, "usage_metadata", None)
                if isinstance(usage_metadata, Mapping):
                    candidates.append(usage_metadata)
                response_metadata = getattr(message, "response_metadata", None)
                if isinstance(response_metadata, Mapping):
                    for key in ("model_name", "model"):
                        if model is None and isinstance(response_metadata.get(key), str):
                            model = str(response_metadata[key])
                    candidates.append(response_metadata)
                    token_usage = response_metadata.get("token_usage")
                    if isinstance(token_usage, Mapping):
                        candidates.append(token_usage)

    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    for candidate in candidates:
        extracted = _usage_from_mapping(candidate)
        if extracted["total_tokens"] > 0 or extracted["input_tokens"] > 0 or extracted["output_tokens"] > 0:
            _merge_usage(usage, extracted)
            break
    return usage, model


@dataclass
class _PendingRun:
    input_tokens: int = 0
    model: str | None = None


@dataclass
class TokenUsageCollector(BaseCallbackHandler):
    """Collect model token usage from LangChain callbacks with an estimate fallback."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_calls: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _pending: dict[str, _PendingRun] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        model = _model_from_serialized(serialized)
        input_tokens = sum(_estimate_text_tokens(prompt) for prompt in prompts)
        with self._lock:
            self._pending[str(run_id)] = _PendingRun(input_tokens=input_tokens, model=model)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        model = _model_from_serialized(serialized)
        input_tokens = _messages_token_estimate(messages)
        with self._lock:
            self._pending[str(run_id)] = _PendingRun(input_tokens=input_tokens, model=model)

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        del kwargs
        usage, model = _extract_usage_and_model(response)
        estimated = False
        run_key = str(run_id)
        with self._lock:
            pending = self._pending.pop(run_key, _PendingRun())

        if usage["total_tokens"] <= 0:
            output_tokens = _estimate_text_tokens(_generations_text(response) or "")
            usage = {
                "input_tokens": pending.input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": pending.input_tokens + output_tokens,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            }
            estimated = True

        self.record_usage(usage, model=model or pending.model or "unknown", estimated=estimated)

    def record_usage(
        self,
        usage: Mapping[str, int],
        *,
        model: str = "unknown",
        estimated: bool = False,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += _as_int(usage.get("input_tokens", 0))
            self.output_tokens += _as_int(usage.get("output_tokens", 0))
            self.total_tokens += _as_int(usage.get("total_tokens", 0))
            self.cached_tokens += _as_int(usage.get("cached_tokens", 0))
            self.reasoning_tokens += _as_int(usage.get("reasoning_tokens", 0))
            if estimated:
                self.estimated_calls += 1
            model_key = model or "unknown"
            model_usage = self.by_model.setdefault(
                model_key,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "estimated_calls": 0,
                },
            )
            model_usage["calls"] += 1
            model_usage["input_tokens"] += _as_int(usage.get("input_tokens", 0))
            model_usage["output_tokens"] += _as_int(usage.get("output_tokens", 0))
            model_usage["total_tokens"] += _as_int(usage.get("total_tokens", 0))
            model_usage["cached_tokens"] += _as_int(usage.get("cached_tokens", 0))
            model_usage["reasoning_tokens"] += _as_int(usage.get("reasoning_tokens", 0))
            if estimated:
                model_usage["estimated_calls"] += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "cached_tokens": self.cached_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "estimated_calls": self.estimated_calls,
                "by_model": {key: dict(value) for key, value in self.by_model.items()},
            }


def _model_from_serialized(serialized: Mapping[str, Any] | None) -> str | None:
    if not isinstance(serialized, Mapping):
        return None
    kwargs = serialized.get("kwargs")
    if isinstance(kwargs, Mapping):
        for key in ("model", "model_name"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("model", "model_name", "name"):
        value = serialized.get(key)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class LangfuseTurnObservation:
    enabled: bool
    collector: TokenUsageCollector | None = None
    _span: Any | None = None

    def token_usage(self) -> dict[str, Any]:
        if self.collector is None:
            return {}
        summary = self.collector.summary()
        return summary if summary.get("calls", 0) > 0 else {}

    def update_output(self, output: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.update(output=output)
        except Exception:
            logger.debug("Failed to update Langfuse turn output", exc_info=True)

    def update_error(self, error: BaseException) -> None:
        if self._span is None:
            return
        try:
            self._span.update(
                metadata={
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                }
            )
        except Exception:
            logger.debug("Failed to update Langfuse turn error", exc_info=True)


def _compact_output(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        message = output.get("message")
        return {
            "success": bool(output.get("success")),
            "type": output.get("type"),
            "message_chars": len(message) if isinstance(message, str) else 0,
        }
    if isinstance(output, str):
        return {"text_chars": len(output)}
    return {"type": type(output).__name__}


@contextlib.contextmanager
def observe_chat_turn(
    *,
    user_message: str,
    user_id: str,
    session_id: str,
    workflow_id: str | None,
) -> Iterator[LangfuseTurnObservation]:
    tags = ["academic-copilot"]
    if workflow_id:
        tags.append(f"workflow:{workflow_id}")
    else:
        tags.append("mode:dynamic")

    turn_id = uuid.uuid4().hex
    metadata = operation_metadata(
        OP_CHAT_TURN,
        operation_type="turn",
        app="academic-copilot",
        turn_id=turn_id,
        user_id=user_id,
        session_id=session_id,
        tags=",".join(tags),
        workflow_id=workflow_id or "none",
        message_chars=len(user_message),
    )

    callbacks: list[Any] = []
    collector: TokenUsageCollector | None = None
    if token_observability_enabled():
        collector = TokenUsageCollector()
        callbacks.append(collector)

    client = _get_langfuse_client()
    handler = _create_langfuse_handler() if client is not None else None
    if handler is not None:
        callbacks.append(handler)

    span_cm: contextlib.AbstractContextManager[Any]
    if client is not None and hasattr(client, "start_as_current_observation"):
        input_payload: Any = {"message": user_message, "workflow_id": workflow_id}
        trace_context: dict[str, str] | None = None
        create_trace_id = getattr(client, "create_trace_id", None)
        if callable(create_trace_id):
            try:
                trace_context = {"trace_id": create_trace_id(seed=f"{session_id}:{turn_id}")}
            except Exception:
                trace_context = None

        span_kwargs: dict[str, Any] = {
            "as_type": "span",
            "name": OP_CHAT_TURN,
            "input": input_payload,
            "metadata": metadata,
        }
        if trace_context:
            span_kwargs["trace_context"] = trace_context
        try:
            span_cm = client.start_as_current_observation(**span_kwargs)
        except TypeError:
            try:
                span_cm = client.start_as_current_observation(
                    as_type="span",
                    name=OP_CHAT_TURN,
                    input=input_payload,
                    metadata=metadata,
                )
            except TypeError:
                span_cm = client.start_as_current_observation(
                    as_type="span",
                    name=OP_CHAT_TURN,
                    input=input_payload,
                )
    else:
        span_cm = contextlib.nullcontext(None)

    with span_cm as span:
        with _propagate_langfuse_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name=OP_CHAT_TURN,
            tags=tags,
            metadata=metadata,
        ):
            with langchain_observation_context(
                callbacks=callbacks,
                metadata=metadata,
                tags=tags,
            ):
                yield LangfuseTurnObservation(
                    enabled=client is not None,
                    collector=collector,
                    _span=span,
                )


def flush_langfuse() -> None:
    client = _LANGFUSE_CLIENT
    if client is None:
        return
    try:
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
    except Exception:
        logger.debug("Failed to flush Langfuse client", exc_info=True)


def shutdown_langfuse() -> None:
    client = _LANGFUSE_CLIENT
    if client is None:
        return
    try:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        flush_langfuse()
    except Exception:
        logger.debug("Failed to shutdown Langfuse client", exc_info=True)
