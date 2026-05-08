"""FastAPI application for serving the fraud detection model."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, status

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


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return model and reference-data availability."""

    return HealthResponse(**get_model_service().health())


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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    return PredictionResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host=os.getenv("SERVING_HOST", "127.0.0.1"),
        port=int(os.getenv("SERVING_PORT", "8000")),
        reload=os.getenv("SERVING_RELOAD", "true").lower() == "true",
    )
