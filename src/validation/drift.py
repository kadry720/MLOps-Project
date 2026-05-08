"""Schema and distribution drift detection.

Reference data gives the strongest drift signal, so the validator uses it when
available. A compact baseline statistics JSON is also maintained for auditability
and for environments where only summary statistics can be stored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from src.utils.config import resolve_project_path
from src.validation.io import write_json_atomic
from src.validation.schema import required_column_names
from src.validation.types import Severity, ValidationIssue


@dataclass(frozen=True)
class BaselineStatistics:
    """Serializable baseline profile used for future drift comparisons."""

    dataset_name: str
    generated_at: str
    row_count: int
    columns: dict[str, str]
    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]
    target: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generated_at": self.generated_at,
            "row_count": self.row_count,
            "columns": self.columns,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "target": self.target,
        }


def compute_baseline_statistics(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> BaselineStatistics:
    """Create a compact, JSON-serializable profile for a reference dataset."""

    validation_config = config.get("validation", {})
    drift_config = config.get("drift", {})
    numeric_columns = _configured_columns(df, drift_config.get("numeric_columns", []), "number")
    categorical_columns = _configured_columns(
        df, drift_config.get("categorical_columns", []), "object"
    )
    target_column = str(validation_config.get("target_column", ""))

    numeric_stats = {
        column: _numeric_profile(pd.to_numeric(df[column], errors="coerce"))
        for column in numeric_columns
        if column in df.columns
    }
    categorical_stats = {
        column: _categorical_profile(df[column])
        for column in categorical_columns
        if column in df.columns
    }
    target_stats = _categorical_profile(df[target_column]) if target_column in df.columns else {}

    return BaselineStatistics(
        dataset_name=str(validation_config.get("dataset_name", "dataset")),
        generated_at=datetime.now(timezone.utc).isoformat(),
        row_count=int(len(df)),
        columns={column: str(dtype) for column, dtype in df.dtypes.items()},
        numeric=numeric_stats,
        categorical=categorical_stats,
        target=target_stats,
    )


def save_baseline_statistics(baseline: BaselineStatistics, path: str | Path) -> Path:
    """Persist baseline statistics as deterministic, reviewable JSON."""

    return write_json_atomic(path, baseline.to_dict())


def load_baseline_statistics(path: str | Path) -> dict[str, Any] | None:
    """Load baseline JSON if it exists."""

    baseline_path = resolve_project_path(path)
    if not baseline_path.exists():
        return None
    with baseline_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else None


def run_drift_checks(
    current_df: pd.DataFrame,
    config: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Run schema and distribution drift checks from configured thresholds."""

    drift_config = config.get("drift", {})
    if not drift_config.get("enabled", True):
        return [], {"checked": False, "reason": "disabled"}

    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}

    schema_issues, schema_metrics = detect_schema_drift(current_df, config, baseline)
    issues.extend(schema_issues)
    metrics["schema_drift"] = schema_metrics

    if reference_df is not None:
        distribution_issues, distribution_metrics = detect_distribution_drift(
            current_df,
            reference_df,
            drift_config,
        )
    elif baseline is not None:
        distribution_issues, distribution_metrics = detect_distribution_drift_from_baseline(
            current_df,
            baseline,
            drift_config,
        )
    else:
        distribution_issues = [
            ValidationIssue(
                check="distribution_drift",
                severity="warning",
                message="Distribution drift skipped because no reference data or baseline exists.",
            )
        ]
        distribution_metrics = {"checked": False, "reason": "missing_reference"}

    issues.extend(distribution_issues)
    metrics["distribution_drift"] = distribution_metrics
    return issues, metrics


def detect_schema_drift(
    current_df: pd.DataFrame,
    config: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Compare current columns to configured requirements and baseline columns."""

    drift_config = config.get("drift", {}).get("schema_drift", {})
    current_columns = set(current_df.columns)
    required_columns = set(required_column_names(config.get("schema", {})))
    baseline_columns = set((baseline or {}).get("columns", {}).keys())

    if drift_config.get("track", "required") == "all" and baseline_columns:
        tracked_columns = baseline_columns
    else:
        tracked_columns = required_columns

    missing_columns = sorted(tracked_columns - current_columns)
    new_columns = sorted(current_columns - baseline_columns) if baseline_columns else []
    issues: list[ValidationIssue] = []
    metrics = {
        "checked": True,
        "tracked_columns": sorted(tracked_columns),
        "missing_columns": missing_columns,
        "new_columns": new_columns[:100],
        "new_column_count": len(new_columns),
        "baseline_column_count": len(baseline_columns),
        "current_column_count": len(current_columns),
    }

    if missing_columns:
        issues.append(
            ValidationIssue(
                check="schema_drift",
                severity=_severity(drift_config.get("fail_on_missing_columns", True)),
                message=f"Current data is missing tracked columns: {missing_columns}.",
                details=metrics,
            )
        )

    if new_columns:
        issues.append(
            ValidationIssue(
                check="schema_drift",
                severity=_severity(drift_config.get("fail_on_new_columns", False)),
                message=(
                    f"Current data contains {len(new_columns)} columns " "not present in baseline."
                ),
                details=metrics,
            )
        )

    return issues, metrics


def detect_distribution_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_config: Mapping[str, Any],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Run two-sample numeric and categorical drift tests."""

    severity = _severity(drift_config.get("fail_on_distribution_drift", False))
    numeric_issues, numeric_metrics = _numeric_distribution_drift(
        current_df,
        reference_df,
        columns=drift_config.get("numeric_columns", []),
        pvalue_threshold=float(drift_config.get("ks_pvalue_threshold", 0.001)),
        statistic_threshold=float(drift_config.get("ks_statistic_threshold", 0.20)),
        min_samples=int(drift_config.get("min_samples", 20)),
        severity=severity,
    )
    categorical_issues, categorical_metrics = _categorical_distribution_drift(
        current_df,
        reference_df,
        columns=drift_config.get("categorical_columns", []),
        js_threshold=float(drift_config.get("categorical_js_threshold", 0.20)),
        severity=severity,
    )
    return numeric_issues + categorical_issues, {
        "checked": True,
        "mode": "reference_data",
        "numeric": numeric_metrics,
        "categorical": categorical_metrics,
    }


def detect_distribution_drift_from_baseline(
    current_df: pd.DataFrame,
    baseline: Mapping[str, Any],
    drift_config: Mapping[str, Any],
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Compare current data against stored baseline summary statistics."""

    severity = _severity(drift_config.get("fail_on_distribution_drift", False))
    numeric_issues: list[ValidationIssue] = []
    categorical_issues: list[ValidationIssue] = []
    numeric_metrics: dict[str, Any] = {}
    categorical_metrics: dict[str, Any] = {}

    max_mean_shift = float(drift_config.get("max_mean_shift_std", 3.0))
    for column, profile in baseline.get("numeric", {}).items():
        if column not in current_df.columns:
            continue
        current = _numeric_profile(pd.to_numeric(current_df[column], errors="coerce"))
        baseline_std = float(profile.get("std", 0.0) or 0.0)
        denominator = baseline_std if baseline_std > 0 else 1.0
        shift = abs(float(current.get("mean", 0.0)) - float(profile.get("mean", 0.0))) / denominator
        numeric_metrics[column] = {
            "baseline_mean": profile.get("mean"),
            "current_mean": current.get("mean"),
            "mean_shift_std": round(float(shift), 6),
            "max_mean_shift_std": max_mean_shift,
        }
        if shift > max_mean_shift:
            numeric_issues.append(
                ValidationIssue(
                    check="numeric_distribution_drift",
                    severity=severity,
                    column=column,
                    message=(
                        f"Column '{column}' mean shifted by {shift:.2f} "
                        "baseline standard deviations."
                    ),
                    details=numeric_metrics[column],
                )
            )

    js_threshold = float(drift_config.get("categorical_js_threshold", 0.20))
    for column, profile in baseline.get("categorical", {}).items():
        if column not in current_df.columns:
            continue
        baseline_dist = profile.get("distribution", {})
        current_dist = _categorical_distribution(current_df[column])
        distance = _js_distance(baseline_dist, current_dist)
        categorical_metrics[column] = {
            "jensen_shannon_distance": round(distance, 6),
            "threshold": js_threshold,
        }
        if distance > js_threshold:
            categorical_issues.append(
                ValidationIssue(
                    check="categorical_distribution_drift",
                    severity=severity,
                    column=column,
                    message=f"Column '{column}' categorical distribution drift is {distance:.3f}.",
                    details=categorical_metrics[column],
                )
            )

    return numeric_issues + categorical_issues, {
        "checked": True,
        "mode": "baseline_statistics",
        "numeric": numeric_metrics,
        "categorical": categorical_metrics,
    }


def _numeric_distribution_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: Sequence[str],
    pvalue_threshold: float,
    statistic_threshold: float,
    min_samples: int,
    severity: Severity,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}

    for column in columns:
        if column not in current_df.columns or column not in reference_df.columns:
            continue

        current = pd.to_numeric(current_df[column], errors="coerce").dropna()
        reference = pd.to_numeric(reference_df[column], errors="coerce").dropna()
        if len(current) < min_samples or len(reference) < min_samples:
            metrics[column] = {
                "checked": False,
                "reason": "insufficient_samples",
                "current_samples": int(len(current)),
                "reference_samples": int(len(reference)),
            }
            continue

        statistic, pvalue = ks_2samp(reference, current)
        metrics[column] = {
            "checked": True,
            "ks_statistic": round(float(statistic), 6),
            "pvalue": round(float(pvalue), 10),
            "ks_pvalue_threshold": pvalue_threshold,
            "ks_statistic_threshold": statistic_threshold,
        }
        if pvalue < pvalue_threshold and statistic > statistic_threshold:
            issues.append(
                ValidationIssue(
                    check="numeric_distribution_drift",
                    severity=severity,
                    column=column,
                    message=(
                        f"Column '{column}' drifted: KS statistic={statistic:.3f}, "
                        f"p-value={pvalue:.4g}."
                    ),
                    details=metrics[column],
                )
            )

    return issues, metrics


def _categorical_distribution_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    columns: Sequence[str],
    js_threshold: float,
    severity: Severity,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    issues: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}

    for column in columns:
        if column not in current_df.columns or column not in reference_df.columns:
            continue

        reference_dist = _categorical_distribution(reference_df[column])
        current_dist = _categorical_distribution(current_df[column])
        distance = _js_distance(reference_dist, current_dist)
        new_categories = sorted(set(current_dist) - set(reference_dist))
        metrics[column] = {
            "jensen_shannon_distance": round(distance, 6),
            "threshold": js_threshold,
            "new_categories": new_categories[:25],
        }
        if distance > js_threshold:
            issues.append(
                ValidationIssue(
                    check="categorical_distribution_drift",
                    severity=severity,
                    column=column,
                    message=(
                        f"Column '{column}' categorical drift is {distance:.3f}; "
                        f"threshold is {js_threshold:.3f}."
                    ),
                    details=metrics[column],
                )
            )

    return issues, metrics


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    profile: dict[str, Any] = {
        "count": int(clean.count()),
        "missing_fraction": round(float(series.isna().mean()), 6) if len(series) else 0.0,
    }
    if clean.empty:
        return profile

    quantiles = clean.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    profile.update(
        {
            "mean": _safe_float(clean.mean()),
            "std": _safe_float(clean.std(ddof=0)),
            "min": _safe_float(clean.min()),
            "max": _safe_float(clean.max()),
            "quantiles": {str(key): _safe_float(value) for key, value in quantiles.items()},
        }
    )
    return profile


def _categorical_profile(series: pd.Series) -> dict[str, Any]:
    distribution = _categorical_distribution(series)
    return {
        "count": int(series.notna().sum()),
        "missing_fraction": round(float(series.isna().mean()), 6) if len(series) else 0.0,
        "unique_count": int(series.dropna().nunique()),
        "distribution": distribution,
    }


def _categorical_distribution(series: pd.Series) -> dict[str, float]:
    values = series.fillna("__missing__").astype(str)
    if values.empty:
        return {}
    distribution = values.value_counts(normalize=True).head(100)
    return {str(key): round(float(value), 8) for key, value in distribution.to_dict().items()}


def _configured_columns(df: pd.DataFrame, columns: Sequence[str], dtype_group: str) -> list[str]:
    configured = [column for column in columns if column in df.columns]
    if configured:
        return configured
    if dtype_group == "number":
        return df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return df.select_dtypes(exclude=["number", "bool"]).columns.tolist()


def _js_distance(reference: Mapping[str, float], current: Mapping[str, float]) -> float:
    categories = sorted(set(reference) | set(current))
    if not categories:
        return 0.0

    reference_vector = np.array([float(reference.get(category, 0.0)) for category in categories])
    current_vector = np.array([float(current.get(category, 0.0)) for category in categories])
    reference_vector = (
        reference_vector / reference_vector.sum() if reference_vector.sum() else reference_vector
    )
    current_vector = (
        current_vector / current_vector.sum() if current_vector.sum() else current_vector
    )
    return float(jensenshannon(reference_vector, current_vector, base=2.0))


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 8)


def _severity(should_fail: Any) -> Severity:
    return "error" if bool(should_fail) else "warning"
