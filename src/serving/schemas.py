"""Pydantic request and response schemas for model serving."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

FeatureValue = str | int | float | bool | None


class PredictionRequest(BaseModel):
    """Single-row prediction request using raw transaction feature values."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "features": {
                    "TransactionDT": 762552,
                    "TransactionAmt": 25.0,
                    "ProductCD": "H",
                    "card1": 7585,
                    "card4": "visa",
                    "card6": "credit",
                    "P_emaildomain": "gmail.com",
                },
                "confidence_level": 0.95,
            }
        },
    )

    features: dict[str, FeatureValue] = Field(
        ...,
        min_length=1,
        description="Raw transaction features. Missing model columns are imputed by the pipeline.",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.5,
        le=0.999,
        description="Confidence level used for the probability interval.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: dict[str, FeatureValue]) -> dict[str, FeatureValue]:
        cleaned: dict[str, FeatureValue] = {}
        for key, value in features.items():
            if not key or not key.strip():
                raise ValueError("Feature names must be non-empty strings.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Feature '{key}' must be finite when numeric.")
            cleaned[key.strip()] = value
        return cleaned


class ConfidenceInterval(BaseModel):
    """Probability interval for the fraud class."""

    lower: float = Field(..., ge=0.0, le=1.0)
    upper: float = Field(..., ge=0.0, le=1.0)
    confidence_level: float = Field(..., ge=0.5, le=0.999)
    method: str


class PredictionResponse(BaseModel):
    """Prediction result returned by the API."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., ge=0, le=1)
    label: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision_threshold: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: ConfidenceInterval
    model_path: str
    expected_feature_count: int
    missing_feature_count: int
    ignored_features: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response for the serving application."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    model_exists: bool
    reference_data_path: str
    reference_data_exists: bool
    expected_feature_count: int
    message: str


class ErrorResponse(BaseModel):
    """Standard error response for failed predictions."""

    detail: str | dict[str, Any]
