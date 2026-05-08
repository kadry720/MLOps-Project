"""Configurable data-quality checks beyond static schema validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.validation.types import Severity, ValidationIssue

CheckRunner = Callable[[pd.DataFrame], tuple[list[ValidationIssue], dict[str, Any]]]


@dataclass(frozen=True)
class QualityCheck:
    """A named, executable data-quality check."""

    name: str
    runner: CheckRunner

    def run(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], dict[str, Any]]:
        return self.runner(df)


@dataclass(frozen=True)
class QualityCheckSuite:
    """Ordered collection of quality checks.

    Keeping checks in a suite makes the validator open for extension: new
    checks can be registered here without changing pipeline orchestration.
    """

    checks: Sequence[QualityCheck]

    def run(self, df: pd.DataFrame) -> tuple[list[ValidationIssue], dict[str, Any]]:
        issues: list[ValidationIssue] = []
        metrics: dict[str, Any] = {}
        for check in self.checks:
            check_issues, check_metrics = check.run(df)
            issues.extend(check_issues)
            metrics[check.name] = check_metrics
        return issues, metrics


def run_quality_checks(
    df: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Run all configured quality checks and return issues plus metrics."""

    return build_quality_check_suite(config).run(df)


def build_quality_check_suite(config: Mapping[str, Any]) -> QualityCheckSuite:
    """Build the ordered quality check suite from configuration."""

    schema_config = config.get("schema", {})
    quality_config = config.get("quality", {})
    duplicate_config = quality_config.get("duplicates", {})
    unexpected_config = quality_config.get("unexpected_categories", {})
    imbalance_config = quality_config.get("class_imbalance", {})

    return QualityCheckSuite(
        checks=[
            QualityCheck(
                "missing_values",
                lambda frame: check_null_thresholds(
                    frame,
                    schema_config.get("null_thresholds", {}),
                ),
            ),
            QualityCheck(
                "duplicates",
                lambda frame: check_duplicate_rows(
                    frame,
                    subset=duplicate_config.get("subset", []),
                    max_fraction=float(duplicate_config.get("max_fraction", 0.0)),
                ),
            ),
            QualityCheck(
                "unexpected_categories",
                lambda frame: check_unexpected_categories(
                    frame,
                    schema_config.get("categorical_allowed_values", {}),
                    severity=_configured_severity(unexpected_config.get("fail", True)),
                ),
            ),
            QualityCheck(
                "outliers",
                lambda frame: check_outliers(
                    frame,
                    quality_config.get("outliers", {}),
                ),
            ),
            QualityCheck(
                "class_balance",
                lambda frame: check_class_imbalance(
                    frame,
                    target_column=str(imbalance_config.get("target_column", "")),
                    min_minority_ratio=float(imbalance_config.get("min_minority_ratio", 0.01)),
                ),
            ),
            QualityCheck(
                "freshness",
                lambda frame: check_data_freshness(
                    frame,
                    quality_config.get("freshness", {}),
                ),
            ),
        ]
    )


def check_null_thresholds(
    df: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Fail columns whose missing-value rate exceeds configured thresholds."""

    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}
    row_count = max(len(df), 1)

    for column, max_fraction in thresholds.items():
        if column not in df.columns:
            continue

        missing_count = int(df[column].isna().sum())
        missing_fraction = missing_count / row_count
        metrics[column] = {
            "missing_count": missing_count,
            "missing_fraction": round(missing_fraction, 6),
            "max_allowed_fraction": float(max_fraction),
        }
        if missing_fraction > float(max_fraction):
            issues.append(
                ValidationIssue(
                    check="missing_value_threshold",
                    severity="error",
                    column=column,
                    message=(
                        f"Column '{column}' has {missing_fraction:.2%} missing values; "
                        f"allowed threshold is {float(max_fraction):.2%}."
                    ),
                    details=metrics[column],
                )
            )

    return issues, metrics


def check_duplicate_rows(
    df: pd.DataFrame,
    subset: Sequence[str],
    max_fraction: float,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Detect duplicate rows or duplicate business keys."""

    if df.empty:
        return [], {"duplicate_count": 0, "duplicate_fraction": 0.0, "subset": list(subset)}

    subset_columns = [column for column in subset if column in df.columns]
    if subset and len(subset_columns) != len(subset):
        missing = sorted(set(subset) - set(subset_columns))
        return [
            ValidationIssue(
                check="duplicate_rows",
                severity="error",
                message=f"Duplicate check cannot run because columns are missing: {missing}.",
                details={"configured_subset": list(subset), "missing_columns": missing},
            )
        ], {"duplicate_count": None, "duplicate_fraction": None, "subset": list(subset)}

    duplicated = df.duplicated(subset=subset_columns or None, keep=False)
    duplicate_count = int(duplicated.sum())
    duplicate_fraction = duplicate_count / len(df)
    metrics = {
        "duplicate_count": duplicate_count,
        "duplicate_fraction": round(duplicate_fraction, 6),
        "max_allowed_fraction": max_fraction,
        "subset": subset_columns or "all_columns",
    }

    issues: list[ValidationIssue] = []
    if duplicate_fraction > max_fraction:
        issues.append(
            ValidationIssue(
                check="duplicate_rows",
                severity="error",
                message=(
                    f"Duplicate fraction is {duplicate_fraction:.2%}; "
                    f"allowed threshold is {max_fraction:.2%}."
                ),
                details=metrics,
            )
        )
    return issues, metrics


def check_unexpected_categories(
    df: pd.DataFrame,
    allowed_values: Mapping[str, Sequence[Any]],
    severity: Severity = "error",
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Detect categorical values outside an allow-list."""

    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}

    for column, allowed in allowed_values.items():
        if column not in df.columns:
            continue

        allowed_set = {str(value) for value in allowed}
        observed = set(df[column].dropna().astype(str).unique())
        unexpected = sorted(observed - allowed_set)
        metrics[column] = {
            "allowed_count": len(allowed_set),
            "unexpected_count": len(unexpected),
            "unexpected_values": unexpected[:25],
        }
        if unexpected:
            issues.append(
                ValidationIssue(
                    check="unexpected_category",
                    severity=severity,
                    column=column,
                    message=f"Column '{column}' contains unexpected categories: {unexpected[:10]}.",
                    details=metrics[column],
                )
            )

    return issues, metrics


def check_outliers(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Detect high-fraction numeric outliers using a robust z-score."""

    if not config.get("enabled", True):
        return [], {}

    columns = list(config.get("columns", []))
    threshold = float(config.get("robust_zscore_threshold", 8.0))
    max_fraction = float(config.get("max_fraction", 0.05))
    severity = _configured_severity(config.get("fail", False))
    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 10:
            metrics[column] = {"checked": False, "reason": "fewer_than_10_non_null_values"}
            continue

        scores = _robust_zscore(values)
        outlier_count = int((np.abs(scores) > threshold).sum())
        outlier_fraction = outlier_count / len(values)
        metrics[column] = {
            "checked": True,
            "outlier_count": outlier_count,
            "outlier_fraction": round(outlier_fraction, 6),
            "max_allowed_fraction": max_fraction,
            "robust_zscore_threshold": threshold,
        }
        if outlier_fraction > max_fraction:
            issues.append(
                ValidationIssue(
                    check="outlier_fraction",
                    severity=severity,
                    column=column,
                    message=(
                        f"Column '{column}' has {outlier_fraction:.2%} robust-z outliers; "
                        f"allowed threshold is {max_fraction:.2%}."
                    ),
                    details=metrics[column],
                )
            )

    return issues, metrics


def check_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    min_minority_ratio: float,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Warn when the target distribution is too imbalanced for reliable training."""

    if not target_column or target_column not in df.columns:
        return [], {"checked": False, "reason": "target_column_missing"}

    target = df[target_column].dropna()
    counts = target.value_counts().sort_index()
    distribution = (counts / max(len(target), 1)).round(6).to_dict()
    metrics = {
        "checked": True,
        "target_column": target_column,
        "class_counts": {str(key): int(value) for key, value in counts.to_dict().items()},
        "class_distribution": {str(key): float(value) for key, value in distribution.items()},
        "min_minority_ratio": min_minority_ratio,
    }

    if len(counts) < 2:
        return [
            ValidationIssue(
                check="class_imbalance",
                severity="warning",
                column=target_column,
                message=f"Target column '{target_column}' contains fewer than two classes.",
                details=metrics,
            )
        ], metrics

    minority_ratio = float((counts / len(target)).min())
    metrics["minority_ratio"] = round(minority_ratio, 6)
    if minority_ratio < min_minority_ratio:
        return [
            ValidationIssue(
                check="class_imbalance",
                severity="warning",
                column=target_column,
                message=(
                    f"Minority class ratio is {minority_ratio:.2%}; "
                    f"configured warning threshold is {min_minority_ratio:.2%}."
                ),
                details=metrics,
            )
        ], metrics

    return [], metrics


def check_data_freshness(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Validate parseable event timestamps and report dataset staleness."""

    if not config.get("enabled", True):
        return [], {"checked": False, "reason": "disabled"}

    timestamp_column = str(config.get("timestamp_column", ""))
    required = bool(config.get("required", False))
    if not timestamp_column or timestamp_column not in df.columns:
        severity: Severity = "error" if required else "warning"
        issue = ValidationIssue(
            check="data_freshness",
            severity=severity,
            column=timestamp_column or None,
            message=f"Freshness column '{timestamp_column}' is not available.",
            details={"required": required},
        )
        return [issue], {"checked": False, "reason": "timestamp_column_missing"}

    raw_values = df[timestamp_column]
    parsed = pd.to_datetime(raw_values, errors="coerce", utc=True)
    non_null_count = int(raw_values.notna().sum())
    invalid_count = int(parsed.isna().sum() - raw_values.isna().sum())
    invalid_fraction = invalid_count / max(non_null_count, 1)
    max_invalid_fraction = float(config.get("max_invalid_fraction", 0.0))
    reference_time = _reference_time(config.get("reference_time"))

    metrics: dict[str, Any] = {
        "checked": True,
        "timestamp_column": timestamp_column,
        "non_null_count": non_null_count,
        "invalid_timestamp_count": invalid_count,
        "invalid_timestamp_fraction": round(invalid_fraction, 6),
        "max_invalid_fraction": max_invalid_fraction,
        "reference_time": reference_time.isoformat(),
    }
    issues: list[ValidationIssue] = []

    if invalid_fraction > max_invalid_fraction:
        issues.append(
            ValidationIssue(
                check="invalid_timestamp",
                severity="error",
                column=timestamp_column,
                message=(
                    f"Column '{timestamp_column}' has {invalid_fraction:.2%} invalid timestamps; "
                    f"allowed threshold is {max_invalid_fraction:.2%}."
                ),
                details=metrics,
            )
        )

    valid_values = parsed.dropna()
    if valid_values.empty:
        return issues, metrics

    latest_timestamp = valid_values.max().to_pydatetime()
    max_future_days = float(config.get("max_future_days", 1.0))
    future_fraction = float(
        (valid_values > reference_time + pd.Timedelta(days=max_future_days)).mean()
    )
    metrics["latest_timestamp"] = latest_timestamp.isoformat()
    metrics["future_timestamp_fraction"] = round(future_fraction, 6)

    if future_fraction > float(config.get("max_future_fraction", 0.0)):
        issues.append(
            ValidationIssue(
                check="future_timestamp",
                severity="error",
                column=timestamp_column,
                message=f"Column '{timestamp_column}' contains timestamps too far in the future.",
                details=metrics,
            )
        )

    max_age_days = config.get("max_age_days")
    if max_age_days is not None:
        age_days = (reference_time - valid_values.max()).total_seconds() / 86_400
        metrics["latest_age_days"] = round(float(age_days), 6)
        metrics["max_age_days"] = float(max_age_days)
        if age_days > float(max_age_days):
            issues.append(
                ValidationIssue(
                    check="data_freshness",
                    severity=_configured_severity(config.get("fail_on_stale", True)),
                    column=timestamp_column,
                    message=(
                        f"Latest timestamp is {age_days:.2f} days old; "
                        f"maximum allowed age is {float(max_age_days):.2f} days."
                    ),
                    details=metrics,
                )
            )

    return issues, metrics


def _robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return 0.6745 * (values - median) / mad

    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def _configured_severity(should_fail: Any) -> Severity:
    return "error" if bool(should_fail) else "warning"


def _reference_time(value: Any) -> pd.Timestamp:
    if value in (None, "", "now"):
        return pd.Timestamp(datetime.now(timezone.utc))
    return (
        pd.Timestamp(value).tz_convert("UTC")
        if pd.Timestamp(value).tzinfo
        else pd.Timestamp(value, tz="UTC")
    )
