from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the single YAML config and remember where it (and the project root) live."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent)
    return config


def resolve_paths(config: Mapping[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Return a copy of config with every `paths.*` entry resolved to an absolute path."""
    output = deepcopy(dict(config))
    project_root = Path(root or output.get("_project_root") or Path.cwd()).expanduser().resolve()
    output["_project_root"] = str(project_root)
    resolved: dict[str, str] = {}
    for key, value in output.get("paths", {}).items():
        candidate = Path(value).expanduser()
        resolved[key] = str(candidate if candidate.is_absolute() else project_root / candidate)
    output["paths"] = resolved
    return output


def ensure_artifact_dirs(config: Mapping[str, Any]) -> None:
    for key, value in config.get("paths", {}).items():
        if key == "sociobench_repo":
            continue
        Path(value).mkdir(parents=True, exist_ok=True)


def load_and_prepare(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    config = resolve_paths(config)
    ensure_artifact_dirs(config)
    return config
