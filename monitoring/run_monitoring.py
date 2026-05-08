"""Generate Evidently baseline and drift reports for fraud monitoring.

The script compares a reference slice against two production-like slices:

* a clean held-out slice with minimal expected drift
* a perturbed slice with deterministic synthetic drift on multiple features

When the saved best model is available, predictions are added so the Evidently
reports include classification performance alongside data quality and drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
from evidently.report import Report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config, resolve_project_path  # noqa: E402

DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "configs" / "validation.yaml"
PREDICTION_COLUMN = "prediction"


@dataclass(frozen=True)
class MonitoringPaths:
    """Resolved paths used by the monitoring run."""

    source_data: Path
    model: Path
    output_dir: Path
    baseline_report: Path
    drift_report: Path
    summary: Path
    log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Evidently baseline and simulated-drift monitoring reports."
    )
    parser.add_argument("--config", default=str(DEFAULT_PARAMS_PATH), help="Path to params YAML.")
    parser.add_argument(
        "--validation-config",
        default=str(DEFAULT_VALIDATION_PATH),
        help="Path to validation YAML with drift feature lists.",
    )
    parser.add_argument("--data", help="Override source data CSV path.")
    parser.add_argument("--model", help="Override saved model path.")
    parser.add_argument("--output-dir", help="Override Evidently report output directory.")
    parser.add_argument("--sample-size", type=int, help="Rows per reference/current slice.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_yaml_config(args.config)
    validation_config = load_yaml_config(args.validation_config)
    monitoring_config = params.get("monitoring", {})
    data_config = params.get("data", {})
    artifact_config = params.get("artifacts", {})

    paths = monitoring_paths(
        monitoring_config=monitoring_config,
        data_config=data_config,
        artifact_config=artifact_config,
        data_override=args.data,
        model_override=args.model,
        output_dir_override=args.output_dir,
    )
    setup_logging(paths.log)

    sample_size = int(args.sample_size or monitoring_config.get("sample_size", 5000))
    random_state = int(monitoring_config.get("random_state", data_config.get("random_state", 42)))
    target_column = str(data_config.get("target_column", "isFraud"))
    id_column = data_config.get("id_column")

    dataset = pd.read_csv(paths.source_data)
    reference, clean_current = split_reference_and_current(dataset, sample_size, random_state)
    drifted_current, injected_features = simulate_drift(
        clean_current,
        monitoring_config=monitoring_config,
        validation_config=validation_config,
        random_state=random_state,
    )

    reference, model_loaded = add_model_predictions(
        reference, paths.model, target_column, id_column
    )
    clean_current, _ = add_model_predictions(clean_current, paths.model, target_column, id_column)
    drifted_current, _ = add_model_predictions(
        drifted_current, paths.model, target_column, id_column
    )

    numerical_features, categorical_features = monitoring_feature_lists(
        reference,
        validation_config=validation_config,
        monitoring_config=monitoring_config,
        target_column=target_column,
        id_column=id_column,
    )
    report_columns = selected_report_columns(
        numerical_features,
        categorical_features,
        target_column=target_column,
        id_column=id_column,
        include_prediction=model_loaded,
        frames=(reference, clean_current, drifted_current),
    )
    reference = reference[report_columns]
    clean_current = clean_current[report_columns]
    drifted_current = drifted_current[report_columns]

    column_mapping = build_column_mapping(
        target_column=target_column,
        id_column=id_column,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        include_prediction=model_loaded,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = run_evidently_report(reference, clean_current, column_mapping, model_loaded)
    drift_report = run_evidently_report(reference, drifted_current, column_mapping, model_loaded)
    baseline_report.save_html(str(paths.baseline_report))
    drift_report.save_html(str(paths.drift_report))

    baseline_summary = extract_drift_summary(baseline_report.as_dict())
    drift_summary = extract_drift_summary(drift_report.as_dict())
    summary = {
        "source_data": display_path(paths.source_data),
        "model_path": display_path(paths.model),
        "model_performance_included": model_loaded,
        "row_counts": {
            "reference": int(len(reference)),
            "clean_current": int(len(clean_current)),
            "drifted_current": int(len(drifted_current)),
        },
        "injected_drift_features": injected_features,
        "baseline_report": display_path(paths.baseline_report),
        "drift_report": display_path(paths.drift_report),
        "baseline": baseline_summary,
        "drift": drift_summary,
    }
    paths.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    threshold = float(monitoring_config.get("drift_alert_share_threshold", 0.20))
    emit_drift_status("baseline", baseline_summary, threshold)
    emit_drift_status("drift", drift_summary, threshold)

    print(f"Baseline report: {paths.baseline_report}")
    print(f"Drift report: {paths.drift_report}")
    print(f"Monitoring summary: {paths.summary}")


def monitoring_paths(
    monitoring_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    artifact_config: Mapping[str, Any],
    data_override: str | None = None,
    model_override: str | None = None,
    output_dir_override: str | None = None,
) -> MonitoringPaths:
    """Resolve paths from CLI overrides and YAML defaults."""

    output_dir = resolve_project_path(
        output_dir_override or monitoring_config.get("output_dir", "monitoring/evidently_reports")
    )
    baseline_name = str(monitoring_config.get("baseline_report_name", "baseline_report.html"))
    drift_name = str(monitoring_config.get("drift_report_name", "drift_report.html"))
    summary_name = str(monitoring_config.get("summary_name", "monitoring_summary.json"))

    return MonitoringPaths(
        source_data=resolve_project_path(
            data_override or monitoring_config.get("source_data_path", data_config.get("test_path"))
        ),
        model=resolve_project_path(
            model_override
            or monitoring_config.get("model_path", artifact_config.get("best_model_path"))
        ),
        output_dir=output_dir,
        baseline_report=output_dir / baseline_name,
        drift_report=output_dir / drift_name,
        summary=output_dir / summary_name,
        log=resolve_project_path(monitoring_config.get("log_path", "logs/monitoring.log")),
    )


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def display_path(path: Path) -> str:
    """Render repo-local paths for portable generated summaries."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def split_reference_and_current(
    dataset: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic reference and held-out current slices."""

    if len(dataset) < 2:
        raise ValueError("Monitoring data must contain at least two rows.")

    rows_needed = min(len(dataset), max(2, sample_size * 2))
    sampled = dataset.sample(n=rows_needed, random_state=random_state).reset_index(drop=True)
    split_index = max(1, len(sampled) // 2)
    return (
        sampled.iloc[:split_index].reset_index(drop=True),
        sampled.iloc[split_index:].reset_index(drop=True),
    )


def simulate_drift(
    current: pd.DataFrame,
    monitoring_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Inject deterministic synthetic drift on numeric and categorical features."""

    drifted = current.copy()
    rng = np.random.default_rng(random_state)
    injected: list[str] = []

    numeric_columns = _configured_drift_columns(
        monitoring_config,
        validation_config,
        key="drifted_numeric_columns",
        validation_key="numeric_columns",
    )
    for column in numeric_columns:
        if column not in drifted.columns:
            continue
        values = pd.to_numeric(drifted[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        drifted[column] = _drift_numeric_series(column, values, monitoring_config)
        injected.append(column)

    categorical_columns = _configured_drift_columns(
        monitoring_config,
        validation_config,
        key="drifted_categorical_columns",
        validation_key="categorical_columns",
    )
    fraction = float(monitoring_config.get("categorical_drift_fraction", 0.75))
    replacements = dict(monitoring_config.get("categorical_replacements", {}))
    for column in categorical_columns:
        if column not in drifted.columns:
            continue
        replacement = replacements.get(column, f"drifted_{column}")
        mask = rng.random(len(drifted)) < fraction
        if not mask.any() and len(drifted) > 0:
            mask[0] = True
        drifted.loc[mask, column] = replacement
        injected.append(column)

    return drifted, injected


def _configured_drift_columns(
    monitoring_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    key: str,
    validation_key: str,
) -> list[str]:
    configured = monitoring_config.get(key)
    if configured:
        return [str(column) for column in configured]
    return [str(column) for column in validation_config.get("drift", {}).get(validation_key, [])]


def _drift_numeric_series(
    column: str,
    values: pd.Series,
    monitoring_config: Mapping[str, Any],
) -> pd.Series:
    if column == "TransactionAmt":
        multiplier = float(monitoring_config.get("amount_drift_multiplier", 4.0))
        return values * multiplier
    if column == "TransactionDT":
        shift = float(monitoring_config.get("transaction_dt_shift", 14 * 24 * 60 * 60))
        return values + shift
    if column == "card1":
        shift = float(monitoring_config.get("card1_shift", 5000))
        return values + shift

    std = float(values.std(ddof=0) or 1.0)
    median = float(values.median() or 0.0)
    return values + (2.0 * std) + abs(median * 0.10)


def add_model_predictions(
    frame: pd.DataFrame,
    model_path: Path,
    target_column: str,
    id_column: str | None,
) -> tuple[pd.DataFrame, bool]:
    """Attach model predictions when the saved best model is available."""

    if not model_path.exists():
        logging.warning("Best model missing; skipping Evidently model performance report.")
        return frame.copy(), False

    model = joblib.load(model_path)
    feature_frame = frame.drop(columns=[target_column, id_column], errors="ignore")
    predicted = model.predict(feature_frame)
    enriched = frame.copy()
    enriched[PREDICTION_COLUMN] = predicted
    return enriched, True


def monitoring_feature_lists(
    reference: pd.DataFrame,
    validation_config: Mapping[str, Any],
    monitoring_config: Mapping[str, Any],
    target_column: str,
    id_column: str | None,
) -> tuple[list[str], list[str]]:
    """Select report features from validation drift settings, falling back to dtypes."""

    excluded = {target_column, id_column, PREDICTION_COLUMN, None}
    numeric = [
        column
        for column in validation_config.get("drift", {}).get("numeric_columns", [])
        if column in reference.columns and column not in excluded
    ]
    categorical = [
        column
        for column in validation_config.get("drift", {}).get("categorical_columns", [])
        if column in reference.columns and column not in excluded
    ]

    if not numeric:
        numeric = [
            column
            for column in reference.select_dtypes(include=["number", "bool"]).columns
            if column not in excluded
        ]
    if not categorical:
        categorical = [
            column
            for column in reference.select_dtypes(exclude=["number", "bool"]).columns
            if column not in excluded
        ]

    max_features = int(monitoring_config.get("max_report_features", 20))
    if len(numeric) + len(categorical) > max_features:
        numeric_keep = min(len(numeric), max_features)
        categorical_keep = max(0, max_features - numeric_keep)
        numeric = numeric[:numeric_keep]
        categorical = categorical[:categorical_keep]

    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def selected_report_columns(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    target_column: str,
    id_column: str | None,
    include_prediction: bool,
    frames: Sequence[pd.DataFrame],
) -> list[str]:
    available = set.intersection(*(set(frame.columns) for frame in frames))
    ordered = [
        column
        for column in [id_column, target_column, PREDICTION_COLUMN]
        if column and (column != PREDICTION_COLUMN or include_prediction) and column in available
    ]
    ordered.extend(column for column in numerical_features if column in available)
    ordered.extend(column for column in categorical_features if column in available)
    return list(dict.fromkeys(ordered))


def build_column_mapping(
    target_column: str,
    id_column: str | None,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    include_prediction: bool,
) -> ColumnMapping:
    return ColumnMapping(
        target=target_column,
        prediction=PREDICTION_COLUMN if include_prediction else None,
        id=id_column,
        numerical_features=list(numerical_features),
        categorical_features=list(categorical_features),
        task="classification" if include_prediction else None,
        pos_label=1,
    )


def run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping,
    include_model_performance: bool,
) -> Report:
    metrics: list[Any] = [DataQualityPreset(), DataDriftPreset()]
    if include_model_performance:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    return report


def extract_drift_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact drift details from an Evidently report dictionary."""

    for metric in report.get("metrics", []):
        if metric.get("metric") != "DataDriftTable":
            continue

        result = metric.get("result", {})
        drift_by_column = result.get("drift_by_columns", {})
        drifted_features = [
            {
                "feature": column,
                "score": details.get("drift_score"),
                "stattest": details.get("stattest_name"),
                "threshold": details.get("stattest_threshold"),
            }
            for column, details in drift_by_column.items()
            if details.get("drift_detected")
        ]
        return {
            "number_of_columns": int(result.get("number_of_columns", 0)),
            "number_of_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
            "share_of_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
            "dataset_drift": bool(result.get("dataset_drift", False)),
            "drifted_features": drifted_features,
        }

    return {
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "share_of_drifted_columns": 0.0,
        "dataset_drift": False,
        "drifted_features": [],
    }


def emit_drift_status(name: str, summary: Mapping[str, Any], threshold: float) -> None:
    """Print and log a structured status message for the project drift threshold."""

    payload = {
        "event": (
            "data_drift_detected"
            if float(summary["share_of_drifted_columns"]) > threshold
            else "data_drift_within_threshold"
        ),
        "report": name,
        "threshold": threshold,
        "number_of_columns": summary["number_of_columns"],
        "number_of_drifted_columns": summary["number_of_drifted_columns"],
        "share_of_drifted_columns": summary["share_of_drifted_columns"],
        "drifted_features": summary["drifted_features"],
    }
    message = json.dumps(payload, sort_keys=True)
    if payload["event"] == "data_drift_detected":
        logging.warning(message)
        print(f"MONITORING_WARNING {message}")
    else:
        logging.info(message)
        print(f"MONITORING_OK {message}")


if __name__ == "__main__":
    main()
