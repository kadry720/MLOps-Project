from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.serving.model_service import PredictionService, probability_confidence_interval


class DummyServingModel:
    classes_ = np.array([0, 1])
    decision_threshold_ = 0.4

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = X["TransactionAmt"].fillna(0).astype(float).clip(0, 100) / 100
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["TransactionAmt"].fillna(0).astype(float) >= 40).astype(int).to_numpy()


def _write_serving_config(tmp_path, model_path, reference_path) -> str:
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        f"""
data:
  test_path: {reference_path}
  target_column: isFraud
  id_column: TransactionID
preprocessing: {{}}
training: {{}}
models: {{}}
artifacts:
  best_model_path: {model_path}
  results_path: {tmp_path / "results.csv"}
""",
        encoding="utf-8",
    )
    return str(config_path)


def test_probability_confidence_interval_is_bounded() -> None:
    interval = probability_confidence_interval(
        0.95, confidence_level=0.95, effective_sample_size=10
    )

    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
    assert interval["confidence_level"] == 0.95


def test_prediction_service_loads_sample_and_predicts(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"
    joblib.dump(DummyServingModel(), model_path)
    pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionAmt": [82.0],
            "ProductCD": ["W"],
            "isFraud": [1],
        }
    ).to_csv(reference_path, index=False)
    config_path = _write_serving_config(tmp_path, model_path, reference_path)

    service = PredictionService(config_path=config_path)
    health = service.health()
    sample = service.load_sample()
    result = service.predict({"TransactionAmt": 82.0, "ProductCD": "W", "ignored": "value"})

    assert health["status"] == "ok"
    assert sample == {"ProductCD": "W", "TransactionAmt": 82.0}
    assert result["prediction"] == 1
    assert result["fraud_probability"] == 0.82
    assert result["decision_threshold"] == 0.4
    assert result["ignored_features"] == ["ignored"]


def test_prediction_service_sample_json_is_valid_json(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"
    joblib.dump(DummyServingModel(), model_path)
    pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionAmt": [20.0],
            "ProductCD": ["H"],
            "isFraud": [0],
        }
    ).to_csv(reference_path, index=False)
    config_path = _write_serving_config(tmp_path, model_path, reference_path)

    payload = PredictionService(config_path=config_path).sample_as_json()

    assert json.loads(payload)["ProductCD"] == "H"


def test_prediction_service_rejects_negative_sample_index(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame({"TransactionAmt": [1.0], "isFraud": [0]}).to_csv(reference_path, index=False)
    config_path = _write_serving_config(tmp_path, model_path, reference_path)

    with pytest.raises(ValueError, match="zero or greater"):
        PredictionService(config_path=config_path).load_sample(-1)
