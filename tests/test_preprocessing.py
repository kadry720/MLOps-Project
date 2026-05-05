import numpy as np
import pandas as pd

from src.data.preprocessing import (
    build_full_preprocessing_pipeline,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_features,
)


def _test_config() -> dict:
    return {
        "preprocessing": {
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "constant",
            "categorical_fill_value": "missing",
            "scaler": "standard",
            "encoder_handle_unknown": "ignore",
            "encoder_max_categories": None,
            "encoder_min_frequency": None,
            "use_feature_selection": False,
            "feature_selection_k": 50,
            "use_polynomial_features": False,
            "polynomial_degree": 2,
        }
    }


def _as_array(transformed):
    return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)


def test_numeric_missing_values_are_imputed_and_scaled():
    X = pd.DataFrame(
        {
            "amount": [10.0, np.nan, 30.0],
            "distance": [1.0, 2.0, np.nan],
        }
    )
    preprocessor = build_full_preprocessing_pipeline(
        numeric_features=["amount", "distance"],
        categorical_features=[],
        config=_test_config(),
    )

    transformed = _as_array(fit_preprocessor(preprocessor, X).transform(X))

    assert not np.isnan(transformed).any()
    assert np.allclose(transformed.mean(axis=0), 0.0)


def test_categorical_missing_values_are_imputed_and_encoded():
    X = pd.DataFrame({"card_type": ["visa", None, "mastercard", "visa"]})
    preprocessor = build_full_preprocessing_pipeline(
        numeric_features=[],
        categorical_features=["card_type"],
        config=_test_config(),
    )

    transformed = _as_array(fit_preprocessor(preprocessor, X).transform(X))

    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= 2


def test_unknown_categories_are_handled_safely():
    train = pd.DataFrame({"card_type": ["visa", "mastercard"]})
    test = pd.DataFrame({"card_type": ["amex"]})
    preprocessor = build_full_preprocessing_pipeline(
        numeric_features=[],
        categorical_features=["card_type"],
        config=_test_config(),
    )

    fit_preprocessor(preprocessor, train)
    transformed = _as_array(transform_features(preprocessor, test))

    assert transformed.shape[0] == 1
    assert transformed.shape[1] >= 1
    assert not np.isnan(transformed).any()


def test_preprocessor_serialization_roundtrip(tmp_path):
    X = pd.DataFrame(
        {
            "amount": [10.0, 20.0, np.nan],
            "card_type": ["visa", "mastercard", None],
        }
    )
    preprocessor = build_full_preprocessing_pipeline(
        numeric_features=["amount"],
        categorical_features=["card_type"],
        config=_test_config(),
    )
    fit_preprocessor(preprocessor, X)
    original = _as_array(transform_features(preprocessor, X))

    artifact_path = tmp_path / "preprocessing_pipeline.pkl"
    save_preprocessor(preprocessor, artifact_path)
    loaded_preprocessor = load_preprocessor(artifact_path)
    loaded = _as_array(transform_features(loaded_preprocessor, X))

    assert loaded.shape == original.shape
    assert np.allclose(loaded, original)
