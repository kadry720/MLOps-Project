"""Prometheus metrics for the serving API."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_client.registry import REGISTRY

REQUEST_COUNT = Counter(
    "fraud_serving_http_requests_total",
    "Total HTTP requests handled by the fraud serving API.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "fraud_serving_http_request_duration_seconds",
    "HTTP request latency for the fraud serving API.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICTION_COUNT = Counter(
    "fraud_serving_predictions_total",
    "Total fraud predictions by predicted class.",
    ["prediction", "label"],
)
PREDICTION_ERRORS = Counter(
    "fraud_serving_prediction_errors_total",
    "Total prediction errors by error type.",
    ["error_type"],
)
PREDICTION_PROBABILITY = Histogram(
    "fraud_serving_prediction_fraud_probability",
    "Fraud-class probability emitted by the model.",
    buckets=(0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1.0),
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_serving_prediction_confidence",
    "Model confidence for the predicted class.",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)
MISSING_FEATURES = Histogram(
    "fraud_serving_prediction_missing_features",
    "Number of expected raw features missing from prediction requests.",
    buckets=(0, 1, 5, 10, 25, 50, 100, 200, 432, 600),
)
HEALTH_STATUS = Gauge(
    "fraud_serving_health_status",
    "Serving health status, where 1 is ok and 0 is degraded.",
)
MODEL_EXISTS = Gauge(
    "fraud_serving_model_artifact_exists",
    "Whether the configured best model artifact exists.",
)
REFERENCE_DATA_EXISTS = Gauge(
    "fraud_serving_reference_data_exists",
    "Whether the configured reference test split exists.",
)
EXPECTED_FEATURES = Gauge(
    "fraud_serving_expected_features",
    "Number of raw features expected by the serving pipeline.",
)
MODEL_LOADED = Gauge(
    "fraud_serving_model_loaded",
    "Whether the model artifact has been loaded into memory.",
)
MODEL_EVALUATION_SCORE = Gauge(
    "fraud_serving_model_evaluation_score",
    "Saved evaluation scores for the selected best model.",
    ["model", "metric"],
)
MODEL_EVALUATION_AVAILABLE = Gauge(
    "fraud_serving_model_evaluation_available",
    "Whether saved model evaluation metrics were loaded.",
)

EVALUATION_METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "roc_auc",
    "average_precision",
    "decision_threshold",
)


def metrics_response() -> Response:
    """Render the Prometheus exposition format."""

    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def update_health_metrics(health: dict) -> None:
    """Record artifact and service health gauges."""

    HEALTH_STATUS.set(1 if health.get("status") == "ok" else 0)
    MODEL_EXISTS.set(1 if health.get("model_exists") else 0)
    REFERENCE_DATA_EXISTS.set(1 if health.get("reference_data_exists") else 0)
    MODEL_LOADED.set(1 if health.get("model_loaded") else 0)
    EXPECTED_FEATURES.set(float(health.get("expected_feature_count") or 0))


def update_evaluation_metrics(results_path: str | Path) -> dict[str, Any]:
    """Load the top row from the MLflow comparison report into Prometheus gauges."""

    path = Path(results_path)
    if not path.exists():
        MODEL_EVALUATION_AVAILABLE.set(0)
        return {"available": False, "path": str(path), "metrics": {}}

    results = pd.read_csv(path)
    if results.empty or "model" not in results.columns:
        MODEL_EVALUATION_AVAILABLE.set(0)
        return {"available": False, "path": str(path), "metrics": {}}

    row = results.iloc[0]
    model_name = str(row["model"])
    loaded_metrics: dict[str, float] = {}
    for metric in EVALUATION_METRIC_COLUMNS:
        if metric not in results.columns or pd.isna(row[metric]):
            continue
        value = float(row[metric])
        MODEL_EVALUATION_SCORE.labels(model=model_name, metric=metric).set(value)
        loaded_metrics[metric] = value

    MODEL_EVALUATION_AVAILABLE.set(1 if loaded_metrics else 0)
    return {
        "available": bool(loaded_metrics),
        "path": str(path),
        "model": model_name,
        "metrics": loaded_metrics,
    }


def record_prediction(result: dict) -> None:
    """Record model prediction metrics."""

    prediction = str(result["prediction"])
    label = str(result["label"])
    PREDICTION_COUNT.labels(prediction=prediction, label=label).inc()
    PREDICTION_PROBABILITY.observe(float(result["fraud_probability"]))
    PREDICTION_CONFIDENCE.observe(float(result["confidence"]))
    MISSING_FEATURES.observe(float(result["missing_feature_count"]))


async def observe_http_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Middleware helper that records request counts and latency."""

    start_time = time.perf_counter()
    path = request.url.path
    method = request.method
    status_code = "500"

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        REQUEST_LATENCY.labels(method=method, path=path).observe(time.perf_counter() - start_time)
        REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
