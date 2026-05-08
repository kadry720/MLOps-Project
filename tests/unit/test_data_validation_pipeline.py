from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_yaml_config
from src.validation.checks import build_quality_check_suite, check_null_thresholds
from src.validation.config import ValidationSettings
from src.validation.drift import detect_distribution_drift
from src.validation.validator import DataValidator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _validation_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_yaml_config("configs/validation.yaml"))
    config["validation"]["baseline_statistics_path"] = str(tmp_path / "baseline.json")
    config["validation"]["report_dir"] = str(tmp_path / "reports")
    config["validation"]["log_dir"] = str(tmp_path / "logs")
    config["validation"]["input_data_path"] = str(
        PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv"
    )
    config["validation"]["reference_data_path"] = str(
        PROJECT_ROOT / "tests/data/reference_fraud_transactions.csv"
    )
    return config


def test_schema_validation_fails_when_required_column_is_missing(tmp_path):
    config = _validation_config(tmp_path)
    df = pd.read_csv(PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv")
    bad_path = tmp_path / "missing_amount.csv"
    df.drop(columns=["TransactionAmt"]).to_csv(bad_path, index=False)
    config["validation"]["input_data_path"] = str(bad_path)

    result = DataValidator(config).run()

    assert not result.passed
    assert any(issue.check == "pandera_schema" for issue in result.errors)
    assert result.json_report_path is not None
    assert result.json_report_path.exists()


def test_null_threshold_check_reports_excessive_missing_values():
    df = pd.DataFrame({"TransactionAmt": [10.0, None, None, 15.0]})

    issues, metrics = check_null_thresholds(df, {"TransactionAmt": 0.25})

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert metrics["TransactionAmt"]["missing_fraction"] == 0.5


def test_quality_check_suite_exposes_stable_metric_sections(tmp_path):
    config = _validation_config(tmp_path)
    df = pd.read_csv(PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv")

    issues, metrics = build_quality_check_suite(config).run(df)

    assert issues == []
    assert set(metrics) == {
        "missing_values",
        "duplicates",
        "unexpected_categories",
        "outliers",
        "class_balance",
        "freshness",
    }


def test_distribution_drift_detection_can_fail_the_gate():
    reference = pd.DataFrame(
        {
            "TransactionAmt": np.linspace(20, 100, 80),
            "ProductCD": ["W"] * 60 + ["C"] * 20,
        }
    )
    current = pd.DataFrame(
        {
            "TransactionAmt": np.linspace(500, 900, 80),
            "ProductCD": ["S"] * 60 + ["H"] * 20,
        }
    )

    issues, metrics = detect_distribution_drift(
        current,
        reference,
        {
            "fail_on_distribution_drift": True,
            "numeric_columns": ["TransactionAmt"],
            "categorical_columns": ["ProductCD"],
            "ks_pvalue_threshold": 0.05,
            "ks_statistic_threshold": 0.20,
            "categorical_js_threshold": 0.10,
            "min_samples": 20,
        },
    )

    assert metrics["checked"] is True
    assert any(issue.severity == "error" for issue in issues)
    assert {issue.check for issue in issues} >= {
        "numeric_distribution_drift",
        "categorical_distribution_drift",
    }


def test_pipeline_failure_behavior_generates_reports(tmp_path):
    config = _validation_config(tmp_path)
    df = pd.read_csv(PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv")
    df.loc[:5, "TransactionAmt"] = np.nan
    bad_path = tmp_path / "bad_nulls.csv"
    df.to_csv(bad_path, index=False)
    config["validation"]["input_data_path"] = str(bad_path)

    result = DataValidator(config).run()

    assert not result.passed
    assert any(issue.check == "missing_value_threshold" for issue in result.errors)
    assert result.html_report_path is not None
    assert result.html_report_path.exists()
    assert result.json_report_path is not None
    assert result.json_report_path.exists()


def test_runtime_context_uses_cli_over_env_over_config(monkeypatch, tmp_path):
    config = _validation_config(tmp_path)
    settings = ValidationSettings.from_mapping(config)
    cli_input = tmp_path / "cli.csv"
    env_input = tmp_path / "env.csv"
    monkeypatch.setenv("DATA_VALIDATION_INPUT_PATH", str(env_input))

    context = settings.resolve_runtime_context(
        input_path=cli_input,
        apply_environment_overrides=True,
    )

    assert context.input_path == cli_input


def test_in_memory_validator_config_is_not_overridden_by_environment(monkeypatch, tmp_path):
    config = _validation_config(tmp_path)
    df = pd.read_csv(PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv")
    df.loc[:5, "TransactionAmt"] = np.nan
    bad_path = tmp_path / "bad_nulls.csv"
    df.to_csv(bad_path, index=False)
    config["validation"]["input_data_path"] = str(bad_path)
    monkeypatch.setenv(
        "DATA_VALIDATION_INPUT_PATH",
        str(PROJECT_ROOT / "tests/data/sample_fraud_transactions.csv"),
    )

    result = DataValidator(config).run()

    assert not result.passed
    assert any(issue.check == "missing_value_threshold" for issue in result.errors)
