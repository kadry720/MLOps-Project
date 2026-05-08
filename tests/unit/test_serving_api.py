from __future__ import annotations

from fastapi.testclient import TestClient

import src.serving.app as serving_app


class DummyPredictionService:
    def health(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": False,
            "model_path": "models/best_model.pkl",
            "model_exists": True,
            "reference_data_path": "data/splits/test.csv",
            "reference_data_exists": True,
            "expected_feature_count": 3,
            "message": "Serving artifacts are available.",
        }

    def predict(self, features: dict, confidence_level: float = 0.95) -> dict:
        assert features["TransactionAmt"] == 25.0
        return {
            "prediction": 1,
            "label": "Fraud",
            "fraud_probability": 0.82,
            "confidence": 0.82,
            "decision_threshold": 0.35,
            "confidence_interval": {
                "lower": 0.74,
                "upper": 0.90,
                "confidence_level": confidence_level,
                "method": "test_interval",
            },
            "model_path": "models/best_model.pkl",
            "expected_feature_count": 3,
            "missing_feature_count": 0,
            "ignored_features": [],
        }


def test_health_endpoint_reports_serving_artifacts(monkeypatch):
    monkeypatch.setattr(serving_app, "get_model_service", lambda: DummyPredictionService())
    client = TestClient(serving_app.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_exists"] is True
    assert body["expected_feature_count"] == 3


def test_predict_endpoint_returns_probability_and_interval(monkeypatch):
    monkeypatch.setattr(serving_app, "get_model_service", lambda: DummyPredictionService())
    client = TestClient(serving_app.app)

    response = client.post(
        "/predict",
        json={
            "features": {"TransactionAmt": 25.0, "ProductCD": "H", "card1": 7585},
            "confidence_level": 0.95,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["fraud_probability"] == 0.82
    assert body["confidence_interval"]["lower"] == 0.74


def test_predict_endpoint_rejects_empty_feature_payload():
    client = TestClient(serving_app.app)

    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422
