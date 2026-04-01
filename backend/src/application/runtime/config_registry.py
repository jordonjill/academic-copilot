from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from src.application.runtime.spec_models import AgentSpec, LLMProfileSpec, WorkflowSpec


class ConfigRegistry:
    def __init__(self, config_root: Path | str) -> None:
        self.config_root = Path(config_root)
        self.config_version = 0
        self.llms: Dict[str, LLMProfileSpec] = {}
        self.agents: Dict[str, AgentSpec] = {}
        self.workflows: Dict[str, WorkflowSpec] = {}

    def reload(self) -> Dict[str, Any]:
        llms_path = self.config_root / "llms.yaml"
        agents_dir = self.config_root / "agents"
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
        new_agents, preserve_agent_ids = self._load_agents(agents_dir, record_failure)
        new_workflows, preserve_workflow_ids = self._load_workflows(workflows_dir, record_failure)

        for failed_name in preserve_llm_names:
            if failed_name in self.llms and failed_name not in new_llms:
                new_llms[failed_name] = self.llms[failed_name]

        for failed_id in preserve_agent_ids:
            if failed_id in self.agents and failed_id not in new_agents:
                new_agents[failed_id] = self.agents[failed_id]

        for failed_id in preserve_workflow_ids:
            if failed_id in self.workflows and failed_id not in new_workflows:
                new_workflows[failed_id] = self.workflows[failed_id]

        self.llms = new_llms
        self.agents = new_agents
        self.workflows = new_workflows
        self.config_version += 1

        return {
            "config_version": self.config_version,
            "loaded_llms": sorted(self.llms.keys()),
            "loaded_agents": sorted(self.agents.keys()),
            "loaded_workflows": sorted(self.workflows.keys()),
            "failed_objects": failed_objects,
        }

    def _load_llms(
        self,
        llms_path: Path,
        record_failure,
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
            try:
                expanded = _expand_env_mapping(raw)
                spec = LLMProfileSpec.model_validate({"name": name, **expanded})
            except ValidationError as exc:
                record_failure("llm", llms_path, exc)
                preserve_names.add(name)
                continue
            new_llms[name] = spec

        return new_llms, preserve_names

    def _load_agents(
        self,
        agents_dir: Path,
        record_failure,
    ) -> tuple[Dict[str, AgentSpec], set[str]]:
        new_agents: Dict[str, AgentSpec] = {}
        seen_ids: dict[str, Path] = {}
        preserve_ids: set[str] = set()
        duplicate_ids: set[str] = set()

        for path in self._iter_yaml_files(agents_dir):
            try:
                payload = self._load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                record_failure("agent", path, exc)
                continue

            raw_id = payload.get("id") if isinstance(payload, dict) else None
            try:
                spec = AgentSpec.model_validate(payload)
            except ValidationError as exc:
                record_failure("agent", path, exc)
                if isinstance(raw_id, str):
                    preserve_ids.add(raw_id)
                continue

            if spec.id in duplicate_ids:
                record_failure("agent", path, ValueError(f"Duplicate agent id: {spec.id}"))
                continue

            if spec.id in seen_ids:
                duplicate_ids.add(spec.id)
                record_failure("agent", path, ValueError(f"Duplicate agent id: {spec.id}"))
                record_failure("agent", seen_ids[spec.id], ValueError(f"Duplicate agent id: {spec.id}"))
                new_agents.pop(spec.id, None)
                continue

            seen_ids[spec.id] = path
            new_agents[spec.id] = spec

        return new_agents, preserve_ids

    def _load_workflows(
        self,
        workflows_dir: Path,
        record_failure,
    ) -> tuple[Dict[str, WorkflowSpec], set[str]]:
        new_workflows: Dict[str, WorkflowSpec] = {}
        seen_ids: dict[str, Path] = {}
        preserve_ids: set[str] = set()
        duplicate_ids: set[str] = set()

        for path in self._iter_yaml_files(workflows_dir):
            try:
                payload = self._load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                record_failure("workflow", path, exc)
                continue

            raw_id = payload.get("id") if isinstance(payload, dict) else None
            try:
                spec = WorkflowSpec.model_validate(payload)
            except ValidationError as exc:
                record_failure("workflow", path, exc)
                if isinstance(raw_id, str):
                    preserve_ids.add(raw_id)
                continue

            if spec.id in duplicate_ids:
                record_failure("workflow", path, ValueError(f"Duplicate workflow id: {spec.id}"))
                continue

            if spec.id in seen_ids:
                duplicate_ids.add(spec.id)
                record_failure("workflow", path, ValueError(f"Duplicate workflow id: {spec.id}"))
                record_failure("workflow", seen_ids[spec.id], ValueError(f"Duplicate workflow id: {spec.id}"))
                new_workflows.pop(spec.id, None)
                continue

            seen_ids[spec.id] = path
            new_workflows[spec.id] = spec

        return new_workflows, preserve_ids

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
            expanded[key] = _ENV_VAR_PATTERN.sub(
                lambda m: os.environ.get(m.group(1), ""),
                value,
            )
        else:
            expanded[key] = value
    return expanded
