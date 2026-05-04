import pandas as pd

from src.data.preprocessing import build_preprocessor, get_column_types, split_features_target


def test_split_features_target_removes_target_and_identifier():
    dataset = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "amount": [20.0, 35.5, 12.0],
            "card_type": ["visa", "mastercard", "visa"],
            "isFraud": [0, 1, 0],
        }
    )

    features, target = split_features_target(dataset, "isFraud", id_column="TransactionID")

    assert "isFraud" not in features.columns
    assert "TransactionID" not in features.columns
    assert target.tolist() == [0, 1, 0]


def test_get_column_types_detects_numeric_and_categorical_columns():
    features = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
            "count": [1, 2],
            "card_type": ["visa", "mastercard"],
            "email_domain": ["gmail.com", "yahoo.com"],
        }
    )

    numeric_columns, categorical_columns = get_column_types(features)

    assert numeric_columns == ["amount", "count"]
    assert categorical_columns == ["card_type", "email_domain"]


def test_build_preprocessor_fits_and_transforms_mixed_data():
    params = {
        "preprocessing": {
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "constant",
            "categorical_fill_value": "missing",
            "onehot_handle_unknown": "ignore",
            "onehot_max_categories": None,
            "onehot_min_frequency": None,
            "scaler": "standard",
        }
    }
    features = pd.DataFrame(
        {
            "amount": [10.0, None, 30.0],
            "distance": [1.0, 2.0, None],
            "card_type": ["visa", None, "mastercard"],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=["amount", "distance"],
        categorical_features=["card_type"],
        params=params,
    )
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 4
    assert not pd.isna(transformed).any()
