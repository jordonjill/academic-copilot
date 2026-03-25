from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0

    model_config = ConfigDict(extra="forbid")


class HooksConfig(BaseModel):
    pre_run: Optional[str] = None
    post_run: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class AgentSpec(BaseModel):
    id: str
    name: str
    mode: Literal["chain", "react"]
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    llm: LLMConfig
    hooks: Optional[HooksConfig] = None

    model_config = ConfigDict(extra="forbid")


class WorkflowSpec(BaseModel):
    id: str
    name: str
    entry_node: str
    nodes: Dict[str, dict]
    edges: List[dict]
    limits: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
