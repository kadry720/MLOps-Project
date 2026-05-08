"""FastAPI application for serving the fraud detection model."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Response, status

from src.serving.metrics import (
    PREDICTION_ERRORS,
    metrics_response,
    observe_http_request,
    record_prediction,
    update_evaluation_metrics,
    update_health_metrics,
)
from src.serving.model_service import PredictionService
from src.serving.schemas import ErrorResponse, HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Fraud Detection Serving API",
    version="1.0.0",
    description="Health and prediction endpoints for the saved best fraud model.",
)


@lru_cache(maxsize=1)
def get_model_service() -> PredictionService:
    return PredictionService()


app.middleware("http")(observe_http_request)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return model and reference-data availability."""

    health_result = get_model_service().health()
    update_health_metrics(health_result)
    update_evaluation_metrics(get_model_service().evaluation_results_path)
    return HealthResponse(**health_result)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics."""

    return metrics_response()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
    tags=["prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict fraud risk for one raw transaction sample."""

    try:
        result = get_model_service().predict(
            features=request.features,
            confidence_level=request.confidence_level,
        )
    except ValueError as exc:
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    record_prediction(result)
    return PredictionResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host=os.getenv("SERVING_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVING_PORT", "8000")),
        reload=os.getenv("SERVING_RELOAD", "true").lower() == "true",
    )
