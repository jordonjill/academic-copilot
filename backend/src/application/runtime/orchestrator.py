from __future__ import annotations

from typing import List, Optional

MAX_RETRIES_PER_AGENT = 2
_ACCEPT_CONFIRMATION_PHRASES = {"yes", "use workflow", "使用"}
_REJECT_CONFIRMATION_PHRASES = {"no", "don't use", "不使用"}
_TRAILING_PUNCTUATION = ".,!?;:。！？；：，"


class SupervisorOrchestrator:
    def handle_user_input(
        self,
        state: dict,
        text: Optional[str],
        current_turn: Optional[int] = None,
    ) -> dict:
        """Process user input when a workflow suggestion is pending."""
        pending_confirmation = state.get("pending_workflow_confirmation", False)
        if pending_confirmation and self._confirmation_expired(
            state.get("confirmation_expires_at_turn"), current_turn
        ):
            self._clear_pending_confirmation(state)
            state["orchestration_mode"] = "dynamic"
            pending_confirmation = False

        if not text or not pending_confirmation:
            return state

        if self._is_confirmation_response(text):
            return state

        self._clear_pending_confirmation(state)
        state["orchestration_mode"] = "dynamic"
        return state

    def select_next_agent(
        self, state: dict, candidates: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select the next subagent, respecting the retry cap for the last agent."""
        pool = (
            list(candidates)
            if candidates is not None
            else list(state.get("selected_subagents") or [])
        )
        if not pool:
            return None

        last_agent = state.get("last_selected_agent_id")
        counters = self._ensure_retry_counters(state)

        for candidate in pool:
            if (
                candidate == last_agent
                and counters.get(candidate, 0) >= MAX_RETRIES_PER_AGENT
            ):
                continue

            state["last_selected_agent_id"] = candidate
            counters[candidate] = counters.get(candidate, 0) + 1
            return candidate

        return None

    def _clear_pending_confirmation(self, state: dict) -> None:
        state["pending_workflow_confirmation"] = False
        state["suggested_workflow_id"] = None
        state["confirmation_expires_at_turn"] = None

    def _ensure_retry_counters(self, state: dict) -> dict:
        counters = state.get("agent_retry_counters")
        if counters is None:
            counters = {}
            state["agent_retry_counters"] = counters
        return counters

    def _confirmation_expired(
        self, expires_at: Optional[int], current_turn: Optional[int]
    ) -> bool:
        return (
            expires_at is not None
            and current_turn is not None
            and current_turn > expires_at
        )

    def _strip_trailing_punctuation(self, text: str) -> str:
        trimmed = text.rstrip()
        while trimmed and trimmed[-1] in _TRAILING_PUNCTUATION:
            trimmed = trimmed[:-1].rstrip()
        return trimmed

    def _is_confirmation_response(self, text: str) -> bool:
        normalized = self._normalize_confirmation_text(text)
        return (
            normalized in _ACCEPT_CONFIRMATION_PHRASES
            or normalized in _REJECT_CONFIRMATION_PHRASES
        )

    def _normalize_confirmation_text(self, text: str) -> str:
        normalized = text.strip().lower()
        normalized = self._strip_trailing_punctuation(normalized)
        if normalized.endswith(" please"):
            normalized = normalized[: -len(" please")].rstrip()
            normalized = self._strip_trailing_punctuation(normalized)
        return normalized
