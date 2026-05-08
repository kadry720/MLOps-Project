"""Pandera schema construction for fraud-detection datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors

from src.validation.types import ValidationIssue

PANDERA_DTYPES: dict[str, Any] = {
    "bool": bool,
    "boolean": bool,
    "float": float,
    "float64": float,
    "int": int,
    "int64": int,
    "integer": int,
    "number": float,
    "object": object,
    "str": str,
    "string": str,
}


def required_column_names(config: Mapping[str, Any]) -> list[str]:
    """Return configured required columns in declaration order."""

    return [str(column["name"]) for column in config.get("required_columns", [])]


def _pandera_dtype(dtype_name: str) -> Any:
    try:
        return PANDERA_DTYPES[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Pandera dtype configured: {dtype_name}") from exc


def _range_checks(column_name: str, config: Mapping[str, Any]) -> list[pa.Check]:
    ranges = config.get("numeric_ranges", {})
    if column_name not in ranges:
        return []

    configured_range = ranges[column_name] or {}
    checks: list[pa.Check] = []
    if configured_range.get("min") is not None:
        checks.append(
            pa.Check.ge(
                configured_range["min"],
                error=f"{column_name} must be >= {configured_range['min']}",
            )
        )
    if configured_range.get("max") is not None:
        checks.append(
            pa.Check.le(
                configured_range["max"],
                error=f"{column_name} must be <= {configured_range['max']}",
            )
        )
    return checks


def _categorical_checks(column_name: str, config: Mapping[str, Any]) -> list[pa.Check]:
    allowed_values = config.get("categorical_allowed_values", {})
    if column_name not in allowed_values:
        return []

    values = list(allowed_values[column_name])
    return [
        pa.Check.isin(
            values,
            ignore_na=True,
            error=f"{column_name} contains values outside the allowed set: {values}",
        )
    ]


def build_pandera_schema(config: Mapping[str, Any]) -> pa.DataFrameSchema:
    """Build a reusable Pandera schema from YAML configuration.

    The schema is intentionally non-strict by default because the IEEE-CIS raw
    data has hundreds of columns. Required business-critical fields are enforced
    here, while full schema drift is reported separately against a baseline.
    """

    schema_config = config.get("schema", config)
    columns: dict[str, pa.Column] = {}
    for column_config in schema_config.get("required_columns", []):
        name = str(column_config["name"])
        dtype = _pandera_dtype(str(column_config.get("dtype", "object")))
        nullable = bool(column_config.get("nullable", False))
        checks = [
            *_range_checks(name, schema_config),
            *_categorical_checks(name, schema_config),
        ]
        columns[name] = pa.Column(
            dtype,
            checks=checks,
            nullable=nullable,
            required=True,
            coerce=bool(schema_config.get("coerce", True)),
        )

    return pa.DataFrameSchema(
        columns,
        strict=bool(schema_config.get("strict", False)),
        coerce=bool(schema_config.get("coerce", True)),
        ordered=bool(schema_config.get("ordered", False)),
        name=str(schema_config.get("name", "fraud_transaction_schema")),
    )


def validate_schema(
    df: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Validate a DataFrame with Pandera and convert failures into issues."""

    schema = build_pandera_schema(config)
    try:
        validated = schema.validate(df, lazy=True)
        return validated, []
    except SchemaErrors as exc:
        return df, _schema_errors_to_issues(exc)
    except SchemaError as exc:
        return df, [
            ValidationIssue(
                check="pandera_schema",
                severity="error",
                column=getattr(exc, "schema", None).name if getattr(exc, "schema", None) else None,
                message=str(exc),
            )
        ]


def _schema_errors_to_issues(exc: SchemaErrors, limit: int = 30) -> list[ValidationIssue]:
    failure_cases = exc.failure_cases
    issues: list[ValidationIssue] = []

    if not isinstance(failure_cases, pd.DataFrame) or failure_cases.empty:
        return [
            ValidationIssue(
                check="pandera_schema",
                severity="error",
                message=str(exc),
            )
        ]

    grouped = failure_cases.head(limit).to_dict(orient="records")
    for failure in grouped:
        column = failure.get("column")
        check = failure.get("check")
        failure_case = failure.get("failure_case")
        index = failure.get("index")
        message = (
            f"Schema validation failed for column '{column}' on check '{check}' "
            f"at index '{index}' with value '{failure_case}'."
        )
        issues.append(
            ValidationIssue(
                check="pandera_schema",
                severity="error",
                column=None if pd.isna(column) else str(column),
                message=message,
                details={key: _json_safe(value) for key, value in failure.items()},
            )
        )

    remaining = len(failure_cases) - len(grouped)
    if remaining > 0:
        issues.append(
            ValidationIssue(
                check="pandera_schema",
                severity="error",
                message=f"Pandera found {remaining} additional schema failure cases.",
                details={"additional_failure_cases": remaining},
            )
        )
    return issues


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
