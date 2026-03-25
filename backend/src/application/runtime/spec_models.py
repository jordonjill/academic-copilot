from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0


class HooksConfig(BaseModel):
    pre_run: Optional[List[str]] = None
    post_run: Optional[List[str]] = None


class AgentSpec(BaseModel):
    id: str
    name: str
    mode: Literal["chain", "react"]
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    llm: LLMConfig
    hooks: Optional[HooksConfig] = None


class WorkflowSpec(BaseModel):
    id: str
    name: str
    entry_node: str
    nodes: Dict[str, dict]
    edges: List[dict] = Field(default_factory=list)
    limits: Dict[str, int] = Field(default_factory=dict)
