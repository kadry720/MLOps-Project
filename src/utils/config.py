"""Configuration helpers shared by pipeline components.

The project keeps operational values in YAML and lets CI/CD override data
locations with environment variables. Centralising path resolution here avoids
hardcoded filesystem assumptions in validation, preprocessing, and training.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


class ConfigurationError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


def resolve_project_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path from config relative to the repository root."""

    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    return expanded if expanded.is_absolute() else project_root / expanded


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dictionary for empty files."""

    path = resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ConfigurationError(f"Expected a mapping in YAML config: {path}")
    return config


def require_mapping(config: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    """Read a nested mapping and raise a clear error when it is absent."""

    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing required mapping '{key}' in {context}.")
    return value


def env_override(value: Any, env_var: str) -> Any:
    """Return an environment override when present, otherwise the config value."""

    override = os.getenv(env_var)
    return override if override not in (None, "") else value
