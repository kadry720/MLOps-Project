"""High-level orchestration for the data validation pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar
from uuid import uuid4

import pandas as pd

from src.utils.logger import get_logger, log_context, setup_logger
from src.validation.checks import run_quality_checks
from src.validation.config import (
    ValidationConfigBundle,
    ValidationRuntimeContext,
    load_validation_config,
)
from src.validation.drift import (
    compute_baseline_statistics,
    load_baseline_statistics,
    run_drift_checks,
    save_baseline_statistics,
)
from src.validation.io import load_dataframe
from src.validation.report import write_reports
from src.validation.schema import validate_schema
from src.validation.types import ValidationIssue, ValidationResult

StageResult = TypeVar("StageResult")


class DataValidator:
    """Production-grade, config-driven data validation runner."""

    def __init__(
        self,
        config: Mapping[str, Any] | str | Path = "configs/validation.yaml",
        logger: Logger | None = None,
    ) -> None:
        self.config_bundle: ValidationConfigBundle = load_validation_config(config)
        self.config = self.config_bundle.raw
        self.settings = self.config_bundle.settings
        self.config_path = self.config_bundle.source_path

        bootstrap_context = self.settings.resolve_runtime_context(
            apply_environment_overrides=self.config_bundle.apply_environment_overrides
        )
        setup_logger(
            "data_validation",
            log_dir=bootstrap_context.log_dir,
            level=bootstrap_context.log_level,
            log_file=bootstrap_context.log_file,
            max_bytes=bootstrap_context.log_max_bytes,
            backup_count=bootstrap_context.log_backup_count,
            json_log_file=bootstrap_context.json_log_file,
            enable_console=bootstrap_context.log_enable_console,
        )
        self.logger = logger or get_logger("validator")

    def run(
        self,
        input_path: str | Path | None = None,
        reference_path: str | Path | None = None,
        update_baseline: bool = False,
        fail_on_warnings: bool | None = None,
    ) -> ValidationResult:
        """Run schema, quality, and drift validation and generate reports."""

        context = self.settings.resolve_runtime_context(
            input_path=input_path,
            reference_path=reference_path,
            fail_on_warnings=fail_on_warnings,
            apply_environment_overrides=self.config_bundle.apply_environment_overrides,
        )

        run_id = uuid4().hex
        with log_context(run_id=run_id, dataset=context.dataset_name):
            return self._run_with_context(
                context=context,
                run_id=run_id,
                update_baseline=update_baseline,
            )

    def _run_with_context(
        self,
        *,
        context: ValidationRuntimeContext,
        run_id: str,
        update_baseline: bool,
    ) -> ValidationResult:
        self.logger.info(
            "Starting data validation",
            extra={
                "stage": "orchestration",
                "input_path": str(context.input_path),
                "reference_path": str(context.reference_path) if context.reference_path else None,
            },
        )

        current_df = self._run_stage(
            "load_input",
            lambda: load_dataframe(context.input_path),
            extra={"input_path": str(context.input_path)},
            on_success=lambda dataframe: {
                "row_count": int(len(dataframe)),
                "column_count": int(len(dataframe.columns)),
            },
        )
        issues: list[ValidationIssue] = []
        metrics = self._input_metrics(current_df, context, run_id)

        validated_df, schema_issues = self._run_stage(
            "schema",
            lambda: validate_schema(current_df, self.config),
            on_success=lambda result: {"schema_error_count": len(result[1])},
        )
        issues.extend(schema_issues)
        metrics["schema"] = self._schema_metrics(schema_issues)

        quality_issues, quality_metrics = self._run_stage(
            "quality",
            lambda: run_quality_checks(validated_df, self.config),
            on_success=lambda result: {"quality_issue_count": len(result[0])},
        )
        issues.extend(quality_issues)
        metrics["quality"] = quality_metrics

        reference_df = self._run_stage(
            "reference_data",
            lambda: self._load_reference(context.reference_path),
            extra={
                "reference_path": str(context.reference_path) if context.reference_path else None
            },
            on_success=lambda dataframe: {
                "reference_loaded": dataframe is not None,
                "reference_row_count": int(len(dataframe)) if dataframe is not None else None,
            },
        )
        baseline = self._run_stage(
            "baseline",
            lambda: self._load_or_refresh_baseline(
                context=context,
                validated_df=validated_df,
                reference_df=reference_df,
                update_baseline=update_baseline,
            ),
            extra={"baseline_path": str(context.baseline_path), "update_baseline": update_baseline},
            on_success=lambda baseline_profile: {
                "baseline_available": baseline_profile is not None,
            },
        )

        drift_issues, drift_metrics = self._run_stage(
            "drift",
            lambda: run_drift_checks(
                validated_df,
                self.config,
                baseline=baseline,
                reference_df=reference_df,
            ),
            on_success=lambda result: {"drift_issue_count": len(result[0])},
        )
        issues.extend(drift_issues)
        metrics["drift"] = drift_metrics

        result = ValidationResult(
            dataset_name=context.dataset_name,
            input_path=context.input_path,
            row_count=int(len(validated_df)),
            column_count=int(len(validated_df.columns)),
            issues=issues,
            metrics=metrics,
            baseline_path=context.baseline_path,
            fail_on_warnings=context.fail_on_warnings,
        )
        result = self._run_stage(
            "reporting",
            lambda: write_reports(result, context.report_dir),
            extra={"report_dir": str(context.report_dir)},
            on_success=lambda report_result: {
                "html_report_path": str(report_result.html_report_path),
                "json_report_path": str(report_result.json_report_path),
            },
        )
        self._log_result(result)
        return result

    def _run_stage(
        self,
        stage: str,
        operation: Callable[[], StageResult],
        *,
        extra: dict[str, Any] | None = None,
        on_success: Callable[[StageResult], dict[str, Any]] | None = None,
    ) -> StageResult:
        """Run one validation stage with consistent structured logging."""

        stage_extra = {**(extra or {}), "stage": stage}
        started_at = perf_counter()
        with log_context(stage=stage):
            self.logger.info("Stage started", extra=stage_extra)
            try:
                result = operation()
            except Exception:
                duration_ms = round((perf_counter() - started_at) * 1000, 3)
                self.logger.exception(
                    "Stage failed",
                    extra={**stage_extra, "duration_ms": duration_ms},
                )
                raise

            duration_ms = round((perf_counter() - started_at) * 1000, 3)
            success_extra = on_success(result) if on_success is not None else {}
            self.logger.info(
                "Stage completed",
                extra={**stage_extra, **success_extra, "duration_ms": duration_ms},
            )
            return result

    def _input_metrics(
        self,
        dataframe: pd.DataFrame,
        context: ValidationRuntimeContext,
        run_id: str,
    ) -> dict[str, Any]:
        return {
            "input": {
                "path": str(context.input_path),
                "rows": int(len(dataframe)),
                "columns": int(len(dataframe.columns)),
            },
            "runtime": {
                "run_id": run_id,
                "config_path": str(self.config_path) if self.config_path else "in_memory",
                "reference_path": str(context.reference_path) if context.reference_path else None,
                "baseline_path": str(context.baseline_path),
                "report_dir": str(context.report_dir),
                "log_dir": str(context.log_dir),
                "log_file": context.log_file,
                "json_log_file": context.json_log_file,
                "fail_on_warnings": context.fail_on_warnings,
            },
        }

    def _schema_metrics(self, schema_issues: list[ValidationIssue]) -> dict[str, Any]:
        return {
            "required_column_count": len(self.config.get("schema", {}).get("required_columns", [])),
            "error_count": len(schema_issues),
        }

    def _load_or_refresh_baseline(
        self,
        *,
        context: ValidationRuntimeContext,
        validated_df: pd.DataFrame,
        reference_df: pd.DataFrame | None,
        update_baseline: bool,
    ) -> dict[str, Any] | None:
        baseline = load_baseline_statistics(context.baseline_path)
        should_refresh = update_baseline or (
            baseline is None and context.create_baseline_if_missing
        )
        if not should_refresh:
            self.logger.info(
                "Using existing validation baseline",
                extra={"stage": "baseline", "baseline_path": str(context.baseline_path)},
            )
            return baseline

        baseline_source = reference_df if reference_df is not None else validated_df
        baseline_profile = compute_baseline_statistics(baseline_source, self.config)
        save_baseline_statistics(baseline_profile, context.baseline_path)
        action = "Updated" if update_baseline else "Created"
        self.logger.info(
            "%s validation baseline",
            action,
            extra={"stage": "baseline", "baseline_path": str(context.baseline_path)},
        )
        return baseline_profile.to_dict()

    def _load_reference(self, reference_path: Path | None) -> pd.DataFrame | None:
        if reference_path is None:
            self.logger.warning(
                "No reference data path configured; drift will use baseline statistics.",
                extra={"stage": "reference_data"},
            )
            return None
        if not reference_path.exists():
            self.logger.warning(
                "Reference data not found at %s; drift will use baseline statistics.",
                reference_path,
                extra={"stage": "reference_data", "reference_path": str(reference_path)},
            )
            return None
        self.logger.info(
            "Loading reference data",
            extra={"stage": "reference_data", "reference_path": str(reference_path)},
        )
        return load_dataframe(reference_path)

    def _log_result(self, result: ValidationResult) -> None:
        if result.passed:
            self.logger.info(
                "Data validation passed",
                extra={
                    "stage": "summary",
                    "row_count": result.row_count,
                    "column_count": result.column_count,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "html_report_path": str(result.html_report_path),
                    "json_report_path": str(result.json_report_path),
                },
            )
            return

        self.logger.error(
            "Data validation failed",
            extra={
                "stage": "summary",
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "html_report_path": str(result.html_report_path),
                "json_report_path": str(result.json_report_path),
            },
        )
        for issue in result.errors[:10]:
            self.logger.error(
                "Validation error: %s",
                issue.message,
                extra={
                    "stage": "summary",
                    "check": issue.check,
                    "column": issue.column,
                    "severity": issue.severity,
                },
            )


def run_validation(
    config_path: str | Path = "configs/validation.yaml",
    input_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    update_baseline: bool = False,
    fail_on_warnings: bool | None = None,
) -> ValidationResult:
    """Convenience function used by tests, CLI, and CI/CD."""

    validator = DataValidator(config_path)
    return validator.run(
        input_path=input_path,
        reference_path=reference_path,
        update_baseline=update_baseline,
        fail_on_warnings=fail_on_warnings,
    )
