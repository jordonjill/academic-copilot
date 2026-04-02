from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol, TypeVar

import yaml
from pydantic import ValidationError

from src.application.runtime.spec_models import AgentSpec, LLMProfileSpec, WorkflowSpec


class _HasId(Protocol):
    id: str


_SpecT = TypeVar("_SpecT", bound=_HasId)
_RecordFailure = Callable[[str, Path, Exception | str], None]
_SYSTEM_AGENTS_DIRNAME = "system"
_RESERVED_SYSTEM_AGENT_IDS = {"supervisor"}


@dataclass
class _LoadedAgents:
    merged: Dict[str, AgentSpec]
    user: Dict[str, AgentSpec]
    system: Dict[str, AgentSpec]
    preserve_user_ids: set[str]
    preserve_system_ids: set[str]


class ConfigRegistry:
    def __init__(self, config_root: Path | str) -> None:
        self.config_root = Path(config_root)
        self.config_version = 0
        self.llms: Dict[str, LLMProfileSpec] = {}
        self.agents: Dict[str, AgentSpec] = {}
        self.subagents: Dict[str, AgentSpec] = {}
        self.system_agents: Dict[str, AgentSpec] = {}
        self.workflows: Dict[str, WorkflowSpec] = {}

    def reload(self) -> Dict[str, Any]:
        llms_path = self.config_root / "llms.yaml"
        agents_dir = self.config_root / "agents"
        system_agents_dir = self.config_root / _SYSTEM_AGENTS_DIRNAME
        workflows_dir = self.config_root / "workflows"

        failed_objects: List[Dict[str, str]] = []

        def record_failure(kind: str, path: Path, error: Exception | str) -> None:
            failed_objects.append(
                {
                    "type": kind,
                    "path": str(path),
                    "error": str(error),
                }
            )

        new_llms, preserve_llm_names = self._load_llms(llms_path, record_failure)
        loaded_agents = self._load_agents(
            agents_dir=agents_dir,
            system_agents_dir=system_agents_dir,
            record_failure=record_failure,
        )
        new_agents = loaded_agents.merged
        new_subagents = loaded_agents.user
        new_system_agents = loaded_agents.system
        preserve_agent_ids = loaded_agents.preserve_user_ids | loaded_agents.preserve_system_ids
        new_workflows, preserve_workflow_ids = self._load_workflows(workflows_dir, record_failure)

        for failed_name in preserve_llm_names:
            if failed_name in self.llms and failed_name not in new_llms:
                new_llms[failed_name] = self.llms[failed_name]

        for failed_id in preserve_agent_ids:
            if failed_id in self.agents and failed_id not in new_agents:
                new_agents[failed_id] = self.agents[failed_id]
            if failed_id in self.subagents and failed_id not in new_subagents:
                new_subagents[failed_id] = self.subagents[failed_id]
            if failed_id in self.system_agents and failed_id not in new_system_agents:
                new_system_agents[failed_id] = self.system_agents[failed_id]

        for failed_id in preserve_workflow_ids:
            if failed_id in self.workflows and failed_id not in new_workflows:
                new_workflows[failed_id] = self.workflows[failed_id]

        self.llms = new_llms
        self.agents = new_agents
        self.subagents = new_subagents
        self.system_agents = new_system_agents
        self.workflows = new_workflows
        self.config_version += 1

        return {
            "config_version": self.config_version,
            "loaded_llms": sorted(self.llms.keys()),
            "loaded_agents": sorted(self.agents.keys()),
            "loaded_subagents": sorted(self.subagents.keys()),
            "loaded_system_agents": sorted(self.system_agents.keys()),
            "loaded_workflows": sorted(self.workflows.keys()),
            "failed_objects": failed_objects,
        }

    def _load_llms(
        self,
        llms_path: Path,
        record_failure: _RecordFailure,
    ) -> tuple[Dict[str, LLMProfileSpec], set[str]]:
        new_llms: Dict[str, LLMProfileSpec] = {}
        preserve_names: set[str] = set()

        if not llms_path.exists():
            return new_llms, preserve_names

        try:
            payload = self._load_yaml(llms_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            record_failure("llm", llms_path, exc)
            return new_llms, preserve_names

        raw_llms = payload.get("llms")
        if not isinstance(raw_llms, dict):
            record_failure("llm", llms_path, ValueError("llms.yaml must define mapping: llms: {name: {...}}"))
            return new_llms, preserve_names

        for name, raw in raw_llms.items():
            if not isinstance(name, str):
                record_failure("llm", llms_path, ValueError(f"Invalid llm name key type: {type(name).__name__}"))
                continue
            if not isinstance(raw, dict):
                record_failure("llm", llms_path, ValueError(f"LLM profile '{name}' must be a mapping"))
                preserve_names.add(name)
                continue
            expanded: dict[str, Any] = {}
            try:
                expanded = _expand_env_mapping(raw)
                spec = LLMProfileSpec.model_validate({"name": name, **expanded})
            except (ValidationError, ValueError) as exc:
                unresolved = _collect_unresolved_placeholders(expanded or raw)
                if unresolved:
                    record_failure(
                        "llm",
                        llms_path,
                        ValueError(
                            f"LLM profile '{name}' has unresolved env placeholders: {', '.join(unresolved)}; "
                            f"validation error: {exc}"
                        ),
                    )
                else:
                    record_failure("llm", llms_path, exc)
                preserve_names.add(name)
                continue
            new_llms[name] = spec

        return new_llms, preserve_names

    def _load_agents(
        self,
        *,
        agents_dir: Path,
        system_agents_dir: Path,
        record_failure: _RecordFailure,
    ) -> _LoadedAgents:
        user_agents, preserve_user_ids = self._load_typed_objects(
            kind="agent",
            root=agents_dir,
            record_failure=record_failure,
            validator=AgentSpec.model_validate,
        )
        system_agents, preserve_system_ids = self._load_typed_objects(
            kind="agent",
            root=system_agents_dir,
            record_failure=record_failure,
            validator=AgentSpec.model_validate,
        )

        for reserved_id in sorted(_RESERVED_SYSTEM_AGENT_IDS):
            if reserved_id in user_agents:
                record_failure(
                    "agent",
                    agents_dir,
                    ValueError(
                        f"Agent id '{reserved_id}' is reserved for system use. "
                        f"Move it to '{_SYSTEM_AGENTS_DIRNAME}/' and keep user-editable subagents under 'agents/'."
                    ),
                )
                user_agents.pop(reserved_id, None)
                preserve_user_ids.discard(reserved_id)

        merged_agents: Dict[str, AgentSpec] = dict(system_agents)
        for agent_id, spec in user_agents.items():
            if agent_id in merged_agents:
                record_failure(
                    "agent",
                    agents_dir,
                    ValueError(f"Duplicate agent id between system and user agent catalogs: {agent_id}"),
                )
                continue
            merged_agents[agent_id] = spec

        return _LoadedAgents(
            merged=merged_agents,
            user=user_agents,
            system=system_agents,
            preserve_user_ids=preserve_user_ids,
            preserve_system_ids=preserve_system_ids,
        )

    def _load_workflows(
        self,
        workflows_dir: Path,
        record_failure: _RecordFailure,
    ) -> tuple[Dict[str, WorkflowSpec], set[str]]:
        return self._load_typed_objects(
            kind="workflow",
            root=workflows_dir,
            record_failure=record_failure,
            validator=WorkflowSpec.model_validate,
        )

    def _load_typed_objects(
        self,
        *,
        kind: str,
        root: Path,
        record_failure: _RecordFailure,
        validator: Callable[[dict[str, Any]], _SpecT],
    ) -> tuple[Dict[str, _SpecT], set[str]]:
        loaded: Dict[str, _SpecT] = {}
        seen_ids: dict[str, Path] = {}
        preserve_ids: set[str] = set()
        duplicate_ids: set[str] = set()

        for path in self._iter_yaml_files(root):
            try:
                payload = self._load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                record_failure(kind, path, exc)
                continue

            raw_id = payload.get("id") if isinstance(payload, dict) else None
            try:
                spec = validator(payload)
            except (ValidationError, ValueError) as exc:
                record_failure(kind, path, exc)
                if isinstance(raw_id, str):
                    preserve_ids.add(raw_id)
                continue

            spec_id = getattr(spec, "id", None)
            if not isinstance(spec_id, str) or not spec_id:
                record_failure(kind, path, ValueError(f"Invalid {kind} id: {spec_id!r}"))
                continue

            if spec_id in duplicate_ids:
                record_failure(kind, path, ValueError(f"Duplicate {kind} id: {spec_id}"))
                continue

            if spec_id in seen_ids:
                duplicate_ids.add(spec_id)
                record_failure(kind, path, ValueError(f"Duplicate {kind} id: {spec_id}"))
                record_failure(kind, seen_ids[spec_id], ValueError(f"Duplicate {kind} id: {spec_id}"))
                loaded.pop(spec_id, None)
                continue

            seen_ids[spec_id] = path
            loaded[spec_id] = spec

        return loaded, preserve_ids

    def _iter_yaml_files(self, root: Path) -> List[Path]:
        if not root.exists() or not root.is_dir():
            return []
        return sorted(
            [
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ]
        )

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("YAML must define a mapping at the top level")
        return payload


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_mapping(data: dict[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            expanded[key] = _ENV_VAR_PATTERN.sub(_expand_env_var_non_strict, value)
        else:
            expanded[key] = value
    return expanded


def _expand_env_var_non_strict(match: re.Match[str]) -> str:
    name = match.group(1)
    raw = os.getenv(name)
    if raw is None:
        return match.group(0)
    return raw


def _collect_unresolved_placeholders(data: dict[str, Any]) -> list[str]:
    unresolved: set[str] = set()
    for value in data.values():
        if not isinstance(value, str):
            continue
        for token in _ENV_VAR_PATTERN.findall(value):
            unresolved.add(token)
    return sorted(unresolved)
