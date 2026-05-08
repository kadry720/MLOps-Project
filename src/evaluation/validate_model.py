"""Validate the selected fraud model against configured performance thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_config, resolve_project_path  # noqa: E402
from src.utils.config import load_yaml_config  # noqa: E402

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def validate_model(
    config_path: str | Path = "configs/params.yaml",
    validation_config_path: str | Path = "configs/validation.yaml",
    allow_metrics_fallback: bool = False,
) -> dict[str, Any]:
    """Validate model metrics from the saved artifact or exported experiment log."""

    config = load_config(config_path)
    validation_settings = load_yaml_config(validation_config_path)
    data_config = config["data"]
    artifact_config = config["artifacts"]
    validation_config = validation_settings.get("model_validation", {})

    model_path = resolve_project_path(artifact_config["best_model_path"])
    test_path = resolve_project_path(data_config["test_path"])

    if model_path.exists() and test_path.exists():
        result = evaluate_saved_model(
            model_path=model_path,
            test_path=test_path,
            target_column=data_config["target_column"],
            id_column=data_config.get("id_column"),
        )
    elif allow_metrics_fallback:
        result = load_metrics_fallback(resolve_project_path(artifact_config["results_path"]))
    else:
        missing = [str(path) for path in (model_path, test_path) if not path.exists()]
        raise FileNotFoundError(
            "Model validation artifacts are missing. Run dvc pull or dvc repro. "
            f"Missing: {missing}"
        )

    threshold_failures = metric_threshold_failures(result["metrics"], validation_config)
    result["threshold_failures"] = threshold_failures
    result["passed"] = not threshold_failures
    if threshold_failures:
        raise ValueError(f"Model validation failed: {threshold_failures}")

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def evaluate_saved_model(
    model_path: Path,
    test_path: Path,
    target_column: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Load the saved model artifact and evaluate it on the held-out test split."""

    model = joblib.load(model_path)
    dataset = pd.read_csv(test_path)
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is missing from {test_path}.")

    y_true = dataset[target_column]
    X = dataset.drop(columns=[target_column, id_column], errors="ignore")
    y_score = positive_class_scores(model, X)
    if y_score is None:
        y_pred = model.predict(X)
    else:
        threshold = float(getattr(model, "decision_threshold_", 0.5))
        y_pred = (y_score >= threshold).astype(int)

    metrics = classification_metrics(y_true, y_pred, y_score)
    return {
        "source": "saved_model",
        "model_path": str(model_path),
        "test_path": str(test_path),
        "rows": int(len(dataset)),
        "metrics": metrics,
    }


def positive_class_scores(model: Any, X: pd.DataFrame) -> Any:
    """Return positive-class probabilities when the model supports them."""

    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
    return probabilities[:, positive_index]


def classification_metrics(y_true: pd.Series, y_pred: Any, y_score: Any = None) -> dict[str, float]:
    """Compute the model validation metrics used by CI."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def load_metrics_fallback(results_path: Path) -> dict[str, Any]:
    """Read the best exported MLflow result when model artifacts are unavailable."""

    if not results_path.exists():
        raise FileNotFoundError(f"Experiment results file not found: {results_path}")
    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError(f"Experiment results file is empty: {results_path}")

    row = results.iloc[0]
    metrics = {
        metric: float(row[metric])
        for metric in METRIC_NAMES
        if metric in results.columns and pd.notna(row[metric])
    }
    return {
        "source": "metrics_fallback",
        "results_path": str(results_path),
        "model": str(row.get("model", "unknown")),
        "run_id": str(row.get("run_id", "unknown")),
        "metrics": metrics,
    }


def metric_threshold_failures(
    metrics: dict[str, float],
    validation_config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Return metrics that fall below configured minimum thresholds."""

    failures: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        threshold_key = f"min_{metric}"
        if threshold_key not in validation_config:
            continue
        actual = metrics.get(metric)
        expected = float(validation_config[threshold_key])
        if actual is None or actual < expected:
            failures[metric] = {
                "actual": float(actual) if actual is not None else float("nan"),
                "minimum": expected,
            }
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved model performance.")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--validation-config", default="configs/validation.yaml")
    parser.add_argument(
        "--allow-metrics-fallback",
        action="store_true",
        help="Use reports/mlflow_experiment_results.csv when DVC model artifacts are absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_model(
        args.config,
        validation_config_path=args.validation_config,
        allow_metrics_fallback=args.allow_metrics_fallback,
    )


if __name__ == "__main__":
    main()
