from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMConfig(BaseModel):
    # Strict mode: agent must reference a named LLM profile from config/llms.yaml.
    name: str
    temperature: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class LLMProfileSpec(BaseModel):
    name: str
    model_name: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
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

    @model_validator(mode="after")
    def _validate_tools_for_mode(self) -> "AgentSpec":
        if self.mode == "chain" and self.tools:
            raise ValueError(
                "chain mode does not support tools; set tools to [] or switch mode to 'react'"
            )
        return self


class WorkflowSpec(BaseModel):
    id: str
    name: str
    entry_node: str
    nodes: Dict[str, dict]
    edges: List[dict]
    limits: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_graph(self) -> "WorkflowSpec":
        if self.entry_node not in self.nodes:
            raise ValueError(f"entry_node '{self.entry_node}' is not defined in nodes")

        for idx, edge in enumerate(self.edges):
            if not isinstance(edge, dict):
                raise ValueError(f"edge[{idx}] must be a mapping")
            source = edge.get("from")
            target = edge.get("to")
            if not isinstance(source, str) or not source:
                raise ValueError(f"edge[{idx}].from must be a non-empty string")
            if not isinstance(target, str) or not target:
                raise ValueError(f"edge[{idx}].to must be a non-empty string")
            if source not in self.nodes:
                raise ValueError(f"edge[{idx}].from references unknown node '{source}'")
            if target not in self.nodes:
                raise ValueError(f"edge[{idx}].to references unknown node '{target}'")

        return self
