from __future__ import annotations

import pandas as pd

from monitoring.run_monitoring import (
    extract_drift_summary,
    selected_report_columns,
    simulate_drift,
    split_reference_and_current,
)


def test_split_reference_and_current_is_deterministic() -> None:
    dataset = pd.DataFrame({"value": range(10)})

    first_reference, first_current = split_reference_and_current(dataset, 3, random_state=42)
    second_reference, second_current = split_reference_and_current(dataset, 3, random_state=42)

    assert first_reference.equals(second_reference)
    assert first_current.equals(second_current)
    assert len(first_reference) == 3
    assert len(first_current) == 3


def test_simulate_drift_changes_configured_features() -> None:
    current = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 20.0, 30.0],
            "TransactionDT": [100, 200, 300],
            "card1": [1000, 2000, 3000],
            "ProductCD": ["W", "W", "R"],
            "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com"],
            "card4": ["visa", "mastercard", "visa"],
        }
    )
    monitoring_config = {
        "drifted_numeric_columns": ["TransactionAmt", "TransactionDT", "card1"],
        "drifted_categorical_columns": ["ProductCD", "P_emaildomain", "card4"],
        "categorical_drift_fraction": 1.0,
        "categorical_replacements": {
            "ProductCD": "C",
            "P_emaildomain": "drifted.example.com",
            "card4": "visa",
        },
    }

    drifted, injected = simulate_drift(
        current,
        monitoring_config=monitoring_config,
        validation_config={},
        random_state=42,
    )

    assert set(injected) == {
        "TransactionAmt",
        "TransactionDT",
        "card1",
        "ProductCD",
        "P_emaildomain",
        "card4",
    }
    assert drifted["TransactionAmt"].tolist() == [40.0, 80.0, 120.0]
    assert (drifted["TransactionDT"] > current["TransactionDT"]).all()
    assert (drifted["card1"] > current["card1"]).all()
    assert drifted["ProductCD"].eq("C").all()
    assert drifted["P_emaildomain"].eq("drifted.example.com").all()


def test_extract_drift_summary_reads_evidently_table() -> None:
    report = {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "number_of_columns": 3,
                    "number_of_drifted_columns": 1,
                    "share_of_drifted_columns": 1 / 3,
                    "dataset_drift": False,
                    "drift_by_columns": {
                        "TransactionAmt": {
                            "drift_detected": True,
                            "drift_score": 0.001,
                            "stattest_name": "K-S p_value",
                            "stattest_threshold": 0.05,
                        },
                        "ProductCD": {"drift_detected": False, "drift_score": 0.9},
                    },
                },
            }
        ]
    }

    summary = extract_drift_summary(report)

    assert summary["number_of_columns"] == 3
    assert summary["number_of_drifted_columns"] == 1
    assert summary["share_of_drifted_columns"] == 1 / 3
    assert summary["drifted_features"] == [
        {
            "feature": "TransactionAmt",
            "score": 0.001,
            "stattest": "K-S p_value",
            "threshold": 0.05,
        }
    ]


def test_selected_report_columns_keeps_only_common_columns() -> None:
    reference = pd.DataFrame(columns=["isFraud", "prediction", "TransactionAmt", "ProductCD"])
    current = pd.DataFrame(columns=["isFraud", "prediction", "TransactionAmt", "card4"])

    columns = selected_report_columns(
        numerical_features=["TransactionAmt"],
        categorical_features=["ProductCD", "card4"],
        target_column="isFraud",
        id_column="TransactionID",
        include_prediction=True,
        frames=(reference, current),
    )

    assert columns == ["isFraud", "prediction", "TransactionAmt"]
