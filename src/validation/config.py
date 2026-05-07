"""Typed configuration model for the data validation pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config, require_mapping, resolve_project_path
from src.validation.exceptions import ValidationConfigurationError

ENV_INPUT_PATH = "DATA_VALIDATION_INPUT_PATH"
ENV_REFERENCE_PATH = "DATA_VALIDATION_REFERENCE_PATH"
ENV_BASELINE_PATH = "DATA_VALIDATION_BASELINE_PATH"
ENV_REPORT_DIR = "DATA_VALIDATION_REPORT_DIR"
ENV_LOG_DIR = "DATA_VALIDATION_LOG_DIR"


@dataclass(frozen=True)
class ValidationRuntimeContext:
    """Resolved values for one validation execution."""

    dataset_name: str
    input_path: Path
    reference_path: Path | None
    baseline_path: Path
    report_dir: Path
    log_dir: Path
    log_level: str
    log_file: str
    json_log_file: str | None
    log_max_bytes: int
    log_backup_count: int
    log_enable_console: bool
    target_column: str
    id_column: str | None
    create_baseline_if_missing: bool
    fail_on_warnings: bool


@dataclass(frozen=True)
class ValidationSettings:
    """Validated, typed settings from the top-level ``validation`` YAML block."""

    dataset_name: str
    input_data_path: str | Path
    reference_data_path: str | Path | None
    baseline_statistics_path: str | Path
    report_dir: str | Path
    log_dir: str | Path
    log_level: str
    log_file: str
    json_log_file: str | None
    log_max_bytes: int
    log_backup_count: int
    log_enable_console: bool
    target_column: str
    id_column: str | None
    create_baseline_if_missing: bool
    fail_on_warnings: bool

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ValidationSettings":
        validation_config = require_mapping(config, "validation", "validation config")
        dataset_name = str(validation_config.get("dataset_name", "")).strip()
        input_data_path = validation_config.get("input_data_path")
        target_column = str(validation_config.get("target_column", "")).strip()

        if not dataset_name:
            raise ValidationConfigurationError("validation.dataset_name must be configured.")
        if not input_data_path:
            raise ValidationConfigurationError("validation.input_data_path must be configured.")
        if not target_column:
            raise ValidationConfigurationError("validation.target_column must be configured.")

        id_column = validation_config.get("id_column")
        return cls(
            dataset_name=dataset_name,
            input_data_path=input_data_path,
            reference_data_path=validation_config.get("reference_data_path"),
            baseline_statistics_path=validation_config.get(
                "baseline_statistics_path", "data/reference/validation_baseline.json"
            ),
            report_dir=validation_config.get("report_dir", "reports/validation"),
            log_dir=validation_config.get("log_dir", "logs"),
            log_level=str(validation_config.get("log_level", "INFO")),
            log_file=str(validation_config.get("log_file", "data_validation.log")),
            json_log_file=_optional_string(
                validation_config.get("json_log_file", "data_validation.jsonl")
            ),
            log_max_bytes=int(validation_config.get("log_max_bytes", 2_000_000)),
            log_backup_count=int(validation_config.get("log_backup_count", 5)),
            log_enable_console=bool(validation_config.get("log_enable_console", True)),
            target_column=target_column,
            id_column=str(id_column) if id_column else None,
            create_baseline_if_missing=bool(
                validation_config.get("create_baseline_if_missing", True)
            ),
            fail_on_warnings=bool(validation_config.get("fail_on_warnings", False)),
        )

    def resolve_runtime_context(
        self,
        *,
        input_path: str | Path | None = None,
        reference_path: str | Path | None = None,
        fail_on_warnings: bool | None = None,
        apply_environment_overrides: bool = True,
    ) -> ValidationRuntimeContext:
        """Resolve effective paths with precedence: CLI argument, env var, config."""

        return ValidationRuntimeContext(
            dataset_name=self.dataset_name,
            input_path=resolve_project_path(
                _resolve_override(
                    cli_value=input_path,
                    env_var=ENV_INPUT_PATH,
                    config_value=self.input_data_path,
                    apply_environment_overrides=apply_environment_overrides,
                )
            ),
            reference_path=_optional_project_path(
                _resolve_override(
                    cli_value=reference_path,
                    env_var=ENV_REFERENCE_PATH,
                    config_value=self.reference_data_path,
                    apply_environment_overrides=apply_environment_overrides,
                )
            ),
            baseline_path=resolve_project_path(
                _resolve_override(
                    cli_value=None,
                    env_var=ENV_BASELINE_PATH,
                    config_value=self.baseline_statistics_path,
                    apply_environment_overrides=apply_environment_overrides,
                )
            ),
            report_dir=resolve_project_path(
                _resolve_override(
                    cli_value=None,
                    env_var=ENV_REPORT_DIR,
                    config_value=self.report_dir,
                    apply_environment_overrides=apply_environment_overrides,
                )
            ),
            log_dir=resolve_project_path(
                _resolve_override(
                    cli_value=None,
                    env_var=ENV_LOG_DIR,
                    config_value=self.log_dir,
                    apply_environment_overrides=apply_environment_overrides,
                )
            ),
            log_level=self.log_level,
            log_file=self.log_file,
            json_log_file=self.json_log_file,
            log_max_bytes=self.log_max_bytes,
            log_backup_count=self.log_backup_count,
            log_enable_console=self.log_enable_console,
            target_column=self.target_column,
            id_column=self.id_column,
            create_baseline_if_missing=self.create_baseline_if_missing,
            fail_on_warnings=(
                self.fail_on_warnings if fail_on_warnings is None else fail_on_warnings
            ),
        )


@dataclass(frozen=True)
class ValidationConfigBundle:
    """Raw config plus typed settings and source metadata."""

    raw: dict[str, Any]
    settings: ValidationSettings
    source_path: Path | None
    apply_environment_overrides: bool


def load_validation_config(config: Mapping[str, Any] | str | Path) -> ValidationConfigBundle:
    """Load validation config from YAML or an in-memory mapping.

    In-memory mappings are treated as already-resolved test/application config,
    so environment overrides are disabled by default for deterministic tests.
    YAML-backed CLI runs keep environment override support.
    """

    if isinstance(config, Mapping):
        raw_config = dict(config)
        source_path = None
        apply_environment_overrides = False
    else:
        source_path = resolve_project_path(config)
        raw_config = load_yaml_config(source_path)
        apply_environment_overrides = True

    settings = ValidationSettings.from_mapping(raw_config)
    return ValidationConfigBundle(
        raw=raw_config,
        settings=settings,
        source_path=source_path,
        apply_environment_overrides=apply_environment_overrides,
    )


def _resolve_override(
    *,
    cli_value: str | Path | None,
    env_var: str,
    config_value: str | Path | None,
    apply_environment_overrides: bool,
) -> str | Path:
    if cli_value not in (None, ""):
        return cli_value
    if apply_environment_overrides:
        env_value = os.getenv(env_var)
        if env_value not in (None, ""):
            return env_value
    if config_value in (None, ""):
        raise ValidationConfigurationError(
            f"No value configured for {env_var.lower().replace('data_validation_', '')}."
        )
    return config_value


def _optional_project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_project_path(value)


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
