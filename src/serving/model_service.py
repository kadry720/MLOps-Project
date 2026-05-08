"""Model loading and prediction utilities shared by FastAPI and Gradio."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import load_config, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "params.yaml"
DEFAULT_EFFECTIVE_SAMPLE_SIZE = 100


def _as_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _positive_class_probability(model: Any, features: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    classes = list(getattr(model, "classes_", []))
    positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
    return float(probabilities[0, positive_index])


def probability_confidence_interval(
    probability: float,
    confidence_level: float,
    effective_sample_size: int = DEFAULT_EFFECTIVE_SAMPLE_SIZE,
) -> dict[str, float | str]:
    """Create a bounded normal-approximation interval around a class probability."""

    z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    variance = probability * (1.0 - probability) / max(1, effective_sample_size)
    margin = z_score * math.sqrt(variance)
    return {
        "lower": max(0.0, probability - margin),
        "upper": min(1.0, probability + margin),
        "confidence_level": confidence_level,
        "method": f"normal_probability_interval_n{effective_sample_size}",
    }


class PredictionService:
    """Load the saved best model and run validated single-row predictions."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        model_path: str | Path | None = None,
        reference_data_path: str | Path | None = None,
    ) -> None:
        self.config = load_config(config_path)
        data_config = self.config["data"]
        artifact_config = self.config["artifacts"]

        self.target_column = data_config["target_column"]
        self.id_column = data_config.get("id_column")
        self.model_path = resolve_project_path(model_path or artifact_config["best_model_path"])
        self.reference_data_path = resolve_project_path(
            reference_data_path or data_config["test_path"]
        )
        self._model: Any | None = None
        self.feature_columns = self._load_feature_columns()
        self.reference_dtypes = self._load_reference_dtypes()

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Best model artifact is missing: {self.model_path}. Run dvc pull or dvc repro."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def health(self) -> dict[str, Any]:
        model_exists = self.model_path.exists()
        reference_data_exists = self.reference_data_path.exists()
        model_loaded = self._model is not None
        if model_exists and reference_data_exists and self.feature_columns:
            status = "ok"
            message = "Serving artifacts are available."
        else:
            status = "degraded"
            message = "Model or reference split is missing; run dvc pull or dvc repro."

        return {
            "status": status,
            "model_loaded": model_loaded,
            "model_path": str(self.model_path),
            "model_exists": model_exists,
            "reference_data_path": str(self.reference_data_path),
            "reference_data_exists": reference_data_exists,
            "expected_feature_count": len(self.feature_columns),
            "message": message,
        }

    def _load_feature_columns(self) -> list[str]:
        if not self.reference_data_path.exists():
            return []
        columns = pd.read_csv(self.reference_data_path, nrows=0).columns.tolist()
        return [column for column in columns if column not in {self.target_column, self.id_column}]

    def _load_reference_dtypes(self) -> dict[str, Any]:
        if not self.reference_data_path.exists():
            return {}
        reference = pd.read_csv(self.reference_data_path, nrows=250)
        return {
            column: dtype
            for column, dtype in reference.dtypes.items()
            if column in self.feature_columns
        }

    def _coerce_feature_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError(
                f"Reference split is missing or empty: {self.reference_data_path}. "
                "The service needs it to align raw prediction columns."
            )

        row = {column: features.get(column, np.nan) for column in self.feature_columns}
        frame = pd.DataFrame([row], columns=self.feature_columns)

        for column, dtype in self.reference_dtypes.items():
            if column not in frame:
                continue
            if pd.api.types.is_numeric_dtype(dtype):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = frame[column].astype("object").where(frame[column].notna(), np.nan)
        return frame

    def predict(
        self,
        features: dict[str, Any],
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        frame = self._coerce_feature_frame(features)
        model = self.model
        probability = _positive_class_probability(model, frame)

        if probability is None:
            prediction = int(model.predict(frame)[0])
            probability = float(prediction)
            decision_threshold = 0.5
        else:
            decision_threshold = float(getattr(model, "decision_threshold_", 0.5))
            prediction = int(probability >= decision_threshold)

        confidence = probability if prediction == 1 else 1.0 - probability
        known_features = set(self.feature_columns)
        ignored_features = sorted(feature for feature in features if feature not in known_features)
        missing_features = [feature for feature in self.feature_columns if feature not in features]

        return {
            "prediction": prediction,
            "label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": probability,
            "confidence": confidence,
            "decision_threshold": decision_threshold,
            "confidence_interval": probability_confidence_interval(
                probability,
                confidence_level,
            ),
            "model_path": str(self.model_path),
            "expected_feature_count": len(self.feature_columns),
            "missing_feature_count": len(missing_features),
            "ignored_features": ignored_features,
        }

    def load_sample(self, row_index: int = 0) -> dict[str, Any]:
        if row_index < 0:
            raise ValueError("Sample row index must be zero or greater.")
        if not self.reference_data_path.exists():
            raise FileNotFoundError(f"Reference split is missing: {self.reference_data_path}")

        sample = pd.read_csv(
            self.reference_data_path,
            skiprows=range(1, row_index + 1),
            nrows=1,
        )
        if sample.empty:
            raise ValueError(f"No sample exists at row index {row_index}.")

        sample = sample.drop(columns=[self.target_column, self.id_column], errors="ignore")
        record = sample.iloc[0].to_dict()
        return {key: _as_jsonable(value) for key, value in record.items()}

    def sample_as_json(self, row_index: int = 0) -> str:
        return json.dumps(self.load_sample(row_index), indent=2, sort_keys=True)
