from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from src.application.runtime.spec_models import AgentSpec, WorkflowSpec


class ConfigRegistry:
    def __init__(self, config_root: Path | str) -> None:
        self.config_root = Path(config_root)
        self.config_version = 0
        self.agents: Dict[str, AgentSpec] = {}
        self.workflows: Dict[str, WorkflowSpec] = {}

    def reload(self) -> Dict[str, Any]:
        agents_dir = self.config_root / "agents"
        workflows_dir = self.config_root / "workflows"

        new_agents: Dict[str, AgentSpec] = {}
        new_workflows: Dict[str, WorkflowSpec] = {}
        failed_objects: List[Dict[str, str]] = []

        def record_failure(kind: str, path: Path, error: Exception | str) -> None:
            failed_objects.append(
                {
                    "type": kind,
                    "path": str(path),
                    "error": str(error),
                }
            )

        for path in self._iter_yaml_files(agents_dir):
            try:
                payload = self._load_yaml(path)
                spec = AgentSpec.model_validate(payload)
                if spec.id in new_agents:
                    raise ValueError(f"Duplicate agent id: {spec.id}")
                new_agents[spec.id] = spec
            except (ValueError, ValidationError, yaml.YAMLError) as exc:
                record_failure("agent", path, exc)

        for path in self._iter_yaml_files(workflows_dir):
            try:
                payload = self._load_yaml(path)
                spec = WorkflowSpec.model_validate(payload)
                if spec.id in new_workflows:
                    raise ValueError(f"Duplicate workflow id: {spec.id}")
                new_workflows[spec.id] = spec
            except (ValueError, ValidationError, yaml.YAMLError) as exc:
                record_failure("workflow", path, exc)

        self.agents = new_agents
        self.workflows = new_workflows
        self.config_version += 1

        return {
            "config_version": self.config_version,
            "loaded_agents": sorted(self.agents.keys()),
            "loaded_workflows": sorted(self.workflows.keys()),
            "failed_objects": failed_objects,
        }

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
