from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SupervisorAction = Literal["direct_reply", "run_agent", "run_workflow"]
AgentTaskStatus = Literal["success", "needs_clarification", "failed"]
WorkflowRunnerStatus = Literal["success", "partial", "failed"]


class SupervisorDecision(BaseModel):
    action: SupervisorAction
    target: Optional[str] = None
    instruction: Optional[str] = None
    input_artifact_keys: List[str] = Field(default_factory=list)
    done: bool = False
    final_text: Optional[str] = None
    reason: str = ""

    model_config = ConfigDict(extra="forbid")

    @field_validator("input_artifact_keys", mode="before")
    @classmethod
    def _coerce_input_artifact_keys(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        result.append(text)
            return result
        return []


class AgentTaskConstraints(BaseModel):
    output_format: Literal["json"] = "json"
    must_return_artifacts: bool = True

    model_config = ConfigDict(extra="forbid")


class AgentTaskInput(BaseModel):
    task_id: str
    instruction: str
    input_artifacts: Dict[str, Any] = Field(default_factory=dict)
    constraints: AgentTaskConstraints = Field(default_factory=AgentTaskConstraints)

    model_config = ConfigDict(extra="forbid")


class AgentTaskOutput(BaseModel):
    status: AgentTaskStatus = "success"
    final_text: str = ""
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return 0.5
        if raw < 0.0:
            return 0.0
        if raw > 1.0:
            return 1.0
        return raw

    @field_validator("errors", mode="before")
    @classmethod
    def _normalize_errors(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result


class WorkflowRunnerLimits(BaseModel):
    max_steps: int = 20
    max_loops: int = 4

    model_config = ConfigDict(extra="forbid")


class WorkflowRunnerInput(BaseModel):
    workflow_id: str
    instruction: str
    seed_artifacts: Dict[str, Any] = Field(default_factory=dict)
    limits: WorkflowRunnerLimits = Field(default_factory=WorkflowRunnerLimits)

    model_config = ConfigDict(extra="forbid")


class WorkflowRunnerOutput(BaseModel):
    status: WorkflowRunnerStatus = "success"
    final_text: str = ""
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    trace: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
