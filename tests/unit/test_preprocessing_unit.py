import pandas as pd

from src.data.preprocessing import (
    FraudFeatureEngineer,
    SafeSelectKBest,
    build_preprocessor,
    get_column_types,
    identify_column_types,
    load_raw_data,
    split_features_target,
)


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
        config=params,
    )
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 4
    assert not pd.isna(transformed).any()


def test_fraud_feature_engineer_adds_configured_features():
    config = {
        "preprocessing": {
            "use_feature_engineering": True,
            "use_missing_pattern_features": True,
            "missing_indicator_prefixes": ["card", "addr"],
            "log_transform_features": ["TransactionAmt"],
            "use_transaction_time_features": True,
            "transaction_time_column": "TransactionDT",
            "use_frequency_encoding": True,
            "frequency_encode_features": ["P_emaildomain", "card1"],
            "use_amount_group_features": True,
            "amount_column": "TransactionAmt",
            "amount_groupby_features": ["ProductCD", "card4"],
            "use_target_mean_encoding": True,
            "target_mean_smoothing": 2.0,
            "target_mean_encode_features": ["ProductCD", "card4"],
            "target_mean_encode_interactions": [["card4", "card6"]],
            "email_domain_features": ["P_emaildomain", "R_emaildomain"],
            "interaction_features": [["card4", "card6"], ["P_emaildomain", "R_emaildomain"]],
            "categorical_like_numeric_features": ["card1"],
        }
    }
    features = pd.DataFrame(
        {
            "TransactionAmt": [20.0, 100.5, 250.0],
            "TransactionDT": [3600, 86400, 172800],
            "ProductCD": ["W", "C", "W"],
            "card1": [1234, None, 4321],
            "card4": ["visa", "mastercard", "visa"],
            "card6": ["debit", "credit", "debit"],
            "addr1": [100.0, None, 200.0],
            "P_emaildomain": ["gmail.com", "yahoo.com", None],
            "R_emaildomain": ["gmail.com", "hotmail.com", None],
        }
    )
    target = pd.Series([0, 1, 0])

    engineered = FraudFeatureEngineer(config).fit(features, target).transform(features)

    expected_columns = {
        "missing_total_count",
        "card_missing_count",
        "TransactionAmt_log1p",
        "TransactionAmt_cents",
        "TransactionDT_hour",
        "P_emaildomain_frequency",
        "ProductCD_target_mean",
        "card4_x_card6_target_mean",
        "TransactionAmt_to_ProductCD_median",
        "P_emaildomain_root",
        "P_emaildomain_suffix",
        "email_domain_match",
        "card4_x_card6",
        "P_emaildomain_x_R_emaildomain",
    }
    assert expected_columns.issubset(engineered.columns)
    assert engineered["card1"].dtype == object
    assert engineered["email_domain_match"].tolist() == [1, 0, 0]


def test_identify_column_types_respects_categorical_like_numeric_overrides():
    features = pd.DataFrame({"card1": [1000, 2000], "amount": [10.0, 20.0]})

    numeric, categorical = identify_column_types(
        features,
        {"preprocessing": {"categorical_like_numeric_features": ["card1"]}},
    )

    assert numeric == ["amount"]
    assert categorical == ["card1"]


def test_safe_select_kbest_caps_requested_feature_count():
    selector = SafeSelectKBest(k=10)
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 0, 0]})
    y = pd.Series([0, 0, 1, 1])

    transformed = selector.fit(X, y).transform(X)

    assert transformed.shape == (4, 2)
    assert selector.get_support(indices=True).tolist() == [0, 1]


def test_load_raw_data_merges_identity_when_available(tmp_path):
    transaction_path = tmp_path / "transaction.csv"
    identity_path = tmp_path / "identity.csv"
    pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "isFraud": [0, 1],
            "TransactionAmt": [10.0, 20.0],
        }
    ).to_csv(transaction_path, index=False)
    pd.DataFrame({"TransactionID": [1], "DeviceType": ["desktop"]}).to_csv(
        identity_path,
        index=False,
    )

    merged = load_raw_data(transaction_path, identity_path, id_column="TransactionID")

    assert merged.loc[0, "DeviceType"] == "desktop"
    assert pd.isna(merged.loc[1, "DeviceType"])
