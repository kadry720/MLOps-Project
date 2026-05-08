from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.validate_model import (
    evaluate_saved_model,
    load_metrics_fallback,
    metric_threshold_failures,
    validate_model,
)


class DummyProbabilityModel:
    classes_ = np.array([0, 1])
    decision_threshold_ = 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = X["score"].to_numpy(dtype=float)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["score"].to_numpy(dtype=float) >= self.decision_threshold_).astype(int)


def _write_config(tmp_path, results_path: str, model_path: str = "missing.pkl") -> str:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        f"""
data:
  test_path: missing-test.csv
  target_column: isFraud
  id_column: TransactionID
preprocessing: {{}}
training: {{}}
models: {{}}
artifacts:
  best_model_path: {model_path}
  results_path: {results_path}
""",
        encoding="utf-8",
    )
    return str(config_path)


def _write_validation_config(tmp_path) -> str:
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        """
model_validation:
  min_accuracy: 0.90
  min_precision: 0.80
  min_recall: 0.70
  min_f1: 0.75
  min_roc_auc: 0.80
  min_average_precision: 0.75
""",
        encoding="utf-8",
    )
    return str(config_path)


def test_evaluate_saved_model_computes_threshold_metrics(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    test_path = tmp_path / "test.csv"
    joblib.dump(DummyProbabilityModel(), model_path)
    pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "score": [0.1, 0.8, 0.7, 0.2],
            "isFraud": [0, 1, 1, 0],
        }
    ).to_csv(test_path, index=False)

    result = evaluate_saved_model(
        model_path=model_path,
        test_path=test_path,
        target_column="isFraud",
        id_column="TransactionID",
    )

    assert result["source"] == "saved_model"
    assert result["rows"] == 4
    assert result["metrics"]["f1"] == 1.0
    assert result["metrics"]["roc_auc"] == 1.0


def test_metrics_fallback_reads_best_experiment_row(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model": "gradient_boosting",
                "run_id": "run-1",
                "accuracy": 0.98,
                "precision": 0.82,
                "recall": 0.74,
                "f1": 0.78,
                "roc_auc": 0.91,
                "average_precision": 0.80,
            }
        ]
    ).to_csv(results_path, index=False)

    result = load_metrics_fallback(results_path)

    assert result["source"] == "metrics_fallback"
    assert result["model"] == "gradient_boosting"
    assert result["metrics"]["accuracy"] == 0.98


def test_validate_model_can_use_metrics_fallback(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model": "gradient_boosting",
                "run_id": "run-1",
                "accuracy": 0.98,
                "precision": 0.82,
                "recall": 0.74,
                "f1": 0.78,
                "roc_auc": 0.91,
                "average_precision": 0.80,
            }
        ]
    ).to_csv(results_path, index=False)
    config_path = _write_config(tmp_path, str(results_path))
    validation_config_path = _write_validation_config(tmp_path)

    result = validate_model(
        config_path,
        validation_config_path=validation_config_path,
        allow_metrics_fallback=True,
    )

    assert result["passed"] is True
    assert result["threshold_failures"] == {}


def test_metric_threshold_failures_reports_low_metrics() -> None:
    failures = metric_threshold_failures(
        {"precision": 0.4, "recall": 0.9},
        {"min_precision": 0.8, "min_recall": 0.7},
    )

    assert failures == {"precision": {"actual": 0.4, "minimum": 0.8}}


def test_validate_model_fails_when_threshold_is_not_met(tmp_path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model": "weak_model",
                "run_id": "run-2",
                "accuracy": 0.70,
                "precision": 0.20,
                "recall": 0.10,
                "f1": 0.13,
            }
        ]
    ).to_csv(results_path, index=False)
    config_path = _write_config(tmp_path, str(results_path))
    validation_config_path = _write_validation_config(tmp_path)

    with pytest.raises(ValueError, match="Model validation failed"):
        validate_model(
            config_path,
            validation_config_path=validation_config_path,
            allow_metrics_fallback=True,
        )
