"""Filesystem and dataset IO helpers for validation runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import resolve_project_path
from src.validation.exceptions import ValidationArtifactError, ValidationInputError

SUPPORTED_DATA_EXTENSIONS = {".csv", ".parquet", ".pq", ".json", ".jsonl"}


def load_dataframe(path: str | Path, *, allow_empty: bool = False) -> pd.DataFrame:
    """Load a validation dataset from a supported tabular file format."""

    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        raise ValidationInputError(f"Validation input data not found: {resolved_path}")
    if not resolved_path.is_file():
        raise ValidationInputError(f"Validation input path is not a file: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValidationInputError(
            f"Unsupported validation data format '{suffix}'. Supported formats: {supported}."
        )

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(resolved_path)
        elif suffix in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(resolved_path)
        else:
            dataframe = pd.read_json(resolved_path, lines=suffix == ".jsonl")
    except Exception as exc:  # noqa: BLE001 - convert library detail into domain error.
        raise ValidationInputError(
            f"Could not load validation data from {resolved_path}: {exc}"
        ) from exc

    if dataframe.empty and not allow_empty:
        raise ValidationInputError(f"Validation input data is empty: {resolved_path}")
    return dataframe


def write_text_atomic(path: str | Path, content: str) -> Path:
    """Atomically write text to avoid partially-written CI artifacts."""

    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(output_path)
    except Exception as exc:  # noqa: BLE001 - artifact writes must fail with context.
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ValidationArtifactError(
            f"Could not write validation artifact {output_path}: {exc}"
        ) from exc
    return output_path


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write deterministic JSON."""

    return write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str),
    )
