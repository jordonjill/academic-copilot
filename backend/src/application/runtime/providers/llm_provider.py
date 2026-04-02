from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable
from urllib.parse import urlparse

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from src.application.runtime.utils.env_utils import read_env_float, read_env_int


class LLMProvider:
    def __init__(
        self,
        *,
        registry: Any,
        env_placeholder_pattern: Any,
        create_chat_openai: Callable[..., Any] | None = None,
    ) -> None:
        self._registry = registry
        self._env_placeholder_pattern = env_placeholder_pattern
        self._create_chat_openai = create_chat_openai or ChatOpenAI
        self._llm_cache: OrderedDict[tuple[str, str, str, str, str, str], BaseLanguageModel] = OrderedDict()
        self._llm_cache_max_size = max(1, read_env_int("LLM_CACHE_MAX_SIZE", 128, minimum=1))
        self._llm_cache_lock = threading.Lock()
        self._llm_cache_hits = 0
        self._llm_cache_misses = 0

    def resolve_llm(self, spec: Any) -> BaseLanguageModel:
        llm_ref = spec.llm
        profile_name = llm_ref.name.strip()
        profile = self._registry.llms.get(profile_name)
        if profile is None:
            raise RuntimeError(f"Unknown llm profile name: {profile_name}")

        model = profile.model_name
        base_url = profile.base_url or ""
        api_key_env = (profile.api_key_env or "").strip()
        if self._env_placeholder_pattern.search(model):
            raise RuntimeError(
                f"Unresolved env placeholder in model_name for llm profile '{profile_name}': {model}"
            )
        if base_url and self._env_placeholder_pattern.search(base_url):
            raise RuntimeError(
                f"Unresolved env placeholder in base_url for llm profile '{profile_name}': {base_url}"
            )
        if api_key_env and self._env_placeholder_pattern.search(api_key_env):
            raise RuntimeError(
                f"Unresolved env placeholder in api_key_env for llm profile '{profile_name}': {api_key_env}"
            )
        api_key = ""
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"Missing API key env '{api_key_env}' for llm profile '{profile_name}'"
                )
        temperature = (
            float(llm_ref.temperature)
            if llm_ref.temperature is not None
            else float(profile.temperature)
        )
        cache_temperature = f"{round(temperature, 3):.3f}"
        llm_timeout_seconds = read_env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0)
        cache_timeout = f"{llm_timeout_seconds:.3f}"
        user_agent = _resolve_openai_compat_user_agent(base_url)
        cache_user_agent = user_agent or ""

        key = (
            profile_name,
            model,
            base_url,
            cache_temperature,
            cache_timeout,
            cache_user_agent,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": llm_timeout_seconds,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        if user_agent:
            kwargs["default_headers"] = {"User-Agent": user_agent}

        with self._llm_cache_lock:
            cached = self._llm_cache.get(key)
            if cached is not None:
                self._llm_cache_hits += 1
                self._llm_cache.move_to_end(key)
                return cached

            self._llm_cache_misses += 1
            llm = self._create_chat_openai(**kwargs)
            while len(self._llm_cache) >= self._llm_cache_max_size:
                self._llm_cache.popitem(last=False)
            self._llm_cache[key] = llm
            self._llm_cache.move_to_end(key)
            return llm

    def resolve_default_llm(self, resolve_supervisor_spec: Callable[[], Any]) -> BaseLanguageModel:
        supervisor = resolve_supervisor_spec()
        if supervisor is not None:
            return self.resolve_llm(supervisor)
        if self._registry.agents:
            first_spec = next(iter(self._registry.agents.values()))
            return self.resolve_llm(first_spec)
        raise RuntimeError("No agent specs loaded; cannot resolve default llm")

    def cache_metrics(self) -> dict[str, int]:
        with self._llm_cache_lock:
            return {
                "size": len(self._llm_cache),
                "max_size": self._llm_cache_max_size,
                "hits": self._llm_cache_hits,
                "misses": self._llm_cache_misses,
            }


def _resolve_openai_compat_user_agent(base_url: str) -> str:
    normalized_base_url = (base_url or "").strip()
    if not normalized_base_url:
        return ""
    hostname = (urlparse(normalized_base_url).hostname or "").lower()
    if hostname == "api.openai.com":
        return ""
    return os.getenv("OPENAI_COMPAT_USER_AGENT", "AcademicCopilot/1.0").strip()
