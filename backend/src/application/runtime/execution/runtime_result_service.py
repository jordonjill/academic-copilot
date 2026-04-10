from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.application.runtime.contracts.state_types import RuntimeState


class RuntimeResultService:
    """Applies agent outputs and builds final API result payloads."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def apply_agent_output(
        self,
        state: RuntimeState,
        node_name: str,
        agent_id: str,
        text: str,
        parsed: Optional[Dict[str, Any]],
    ) -> None:
        artifacts_state = state.get("artifacts")
        if not isinstance(artifacts_state, dict):
            artifacts_state = {}
            state["artifacts"] = artifacts_state

        shared = artifacts_state.get("shared")
        if not isinstance(shared, dict):
            if shared is not None:
                self._logger.warning(
                    "runtime.artifacts.shared_reset_invalid_type agent_id=%s type=%s",
                    agent_id,
                    type(shared).__name__,
                )
            shared = {}
            artifacts_state["shared"] = shared

        shared.pop(agent_id, None)
        shared[agent_id] = {
            "node": node_name,
            "output_text": text,
            "parsed": parsed,
            "tool_outputs": list(state["io"].get("last_tool_outputs", [])),
        }

        if parsed and isinstance(parsed, dict):
            artifacts_patch = parsed.get("artifacts")
            if isinstance(artifacts_patch, dict):
                patch = dict(artifacts_patch)
                shared_patch = (
                    patch.pop("shared", None)
                    if "shared" in patch or "shared" in artifacts_patch
                    else None
                )
                artifacts_state.update(patch)
                if "shared" in artifacts_patch:
                    if isinstance(shared_patch, dict):
                        target_shared = artifacts_state.get("shared")
                        if not isinstance(target_shared, dict):
                            target_shared = {}
                            artifacts_state["shared"] = target_shared
                        target_shared.update(shared_patch)
                    else:
                        self._logger.warning(
                            "runtime.artifacts.shared_patch_ignored agent_id=%s type=%s",
                            agent_id,
                            type(shared_patch).__name__,
                        )

            final_text = parsed.get("final_text")
            if isinstance(final_text, str) and final_text.strip():
                state["output"]["final_text"] = final_text

            final_structured = parsed.get("final_structured")
            if isinstance(final_structured, dict):
                state["output"]["final_structured"] = final_structured

        if node_name == "reporter" and not state["output"].get("final_text"):
            state["output"]["final_text"] = text

    def best_available_final_text(self, state: RuntimeState) -> Optional[str]:
        final_text = state["output"].get("final_text")
        if isinstance(final_text, str) and final_text.strip():
            return final_text

        shared = state.get("artifacts", {}).get("shared", {})
        if isinstance(shared, dict):
            reporter = shared.get("reporter")
            if isinstance(reporter, dict):
                reporter_text = reporter.get("output_text")
                if isinstance(reporter_text, str) and reporter_text.strip():
                    return reporter_text

            for item in reversed(list(shared.values())):
                if not isinstance(item, dict):
                    continue
                output_text = item.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text
        return None

    def build_result(self, state: RuntimeState) -> Dict[str, Any]:
        final_structured = state["output"].get("final_structured")
        if isinstance(final_structured, dict):
            result = {"success": True, "type": "structured", "data": final_structured}
            if "message" in final_structured and isinstance(final_structured["message"], str):
                result["message"] = final_structured["message"]
            return result

        final_text = (
            state["output"].get("final_text")
            or self.best_available_final_text(state)
            or state["io"].get("last_execution_output")
            or state["io"].get("last_model_output")
        )
        return {
            "success": bool(final_text),
            "type": "chat",
            "message": final_text or "No output produced.",
            "data": {
                "runtime": state.get("runtime", {}),
                "artifacts": state.get("artifacts", {}),
            },
        }
