"""Reusable preprocessing pipeline for the IEEE fraud detection project.

The preprocessing layer is deliberately separated from notebooks and model
code so the same fitted artifact can be used during training and serving.
That eliminates train/serve skew: missing-value handling, scaling, and
categorical encoding are learned once from the training split and then applied
consistently everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"


def resolve_project_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path relative to the repository root."""

    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load the project parameter file.

    Centralizing configuration keeps preprocessing, DVC, training, and serving
    aligned. Paths, split settings, imputation strategy, encoding behavior, and
    model parameters should come from ``configs/params.yaml`` rather than being
    hardcoded in scripts.
    """

    path = resolve_project_path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    if "data" not in params:
        raise ValueError("configs/params.yaml must define a 'data' section.")
    return params


def _configured_nrows(sample_size: Any) -> int | None:
    if sample_size in (None, "null"):
        return None
    sample_size = int(sample_size)
    return sample_size if sample_size > 0 else None


def _load_raw_dataset(params: dict[str, Any]) -> pd.DataFrame:
    """Load and merge raw DVC data when the processed file is unavailable."""

    data_cfg = params["data"]
    transaction_path = resolve_project_path(data_cfg["raw_transaction_path"])
    identity_path = resolve_project_path(data_cfg["raw_identity_path"])
    id_column = data_cfg["id_column"]
    nrows = _configured_nrows(data_cfg.get("sample_size"))

    if not transaction_path.exists():
        raise FileNotFoundError(
            "Processed data is missing and raw transaction data was not found. "
            f"Run 'python -m dvc pull' and check: {transaction_path}"
        )

    transaction_df = pd.read_csv(transaction_path, nrows=nrows)
    if identity_path.exists():
        identity_df = pd.read_csv(identity_path)
        dataset = transaction_df.merge(identity_df, on=id_column, how="left")
    else:
        print(f"[preprocess] Identity file not found, using transaction data only: {identity_path}")
        dataset = transaction_df

    processed_path = resolve_project_path(data_cfg["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_path, index=False)
    print(f"[preprocess] Created processed dataset from raw DVC files: {processed_path}")
    return dataset


def load_dataset(params: dict[str, Any]) -> pd.DataFrame:
    """Load the model-ready dataset.

    The preferred input is ``data.processed_path``. If that file is not present
    after cloning, the function falls back to the DVC-tracked raw IEEE fraud
    transaction and identity files and creates the processed CSV. This keeps the
    pipeline reproducible from a clean checkout while preserving the EDA output
    path expected by the project.
    """

    data_cfg = params["data"]
    processed_path = resolve_project_path(data_cfg["processed_path"])
    nrows = _configured_nrows(data_cfg.get("sample_size"))

    if processed_path.exists() and processed_path.stat().st_size > 0:
        print(f"[preprocess] Loading processed dataset: {processed_path}")
        return pd.read_csv(processed_path, nrows=nrows)

    print(f"[preprocess] Processed dataset not found at {processed_path}; loading raw DVC data.")
    return _load_raw_dataset(params)


def split_features_target(
    dataset: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into model features and target labels.

    The transaction identifier is removed from features when present because it
    is an identifier, not a generalizable signal. Keeping IDs out of the feature
    matrix reduces leakage risk and improves serving compatibility.
    """

    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    drop_columns = [target_column]
    if id_column and id_column in dataset.columns:
        drop_columns.append(id_column)

    features = dataset.drop(columns=drop_columns)
    target = dataset[target_column]
    return features, target


def get_column_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names detected from dtypes."""

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def _build_one_hot_encoder(preprocessing_cfg: dict[str, Any]) -> OneHotEncoder:
    encoder_kwargs: dict[str, Any] = {
        "handle_unknown": preprocessing_cfg["onehot_handle_unknown"],
    }

    if preprocessing_cfg.get("onehot_max_categories") is not None:
        encoder_kwargs["max_categories"] = preprocessing_cfg["onehot_max_categories"]
    if preprocessing_cfg.get("onehot_min_frequency") is not None:
        encoder_kwargs["min_frequency"] = preprocessing_cfg["onehot_min_frequency"]

    try:
        return OneHotEncoder(sparse_output=False, **encoder_kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **encoder_kwargs)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
) -> ColumnTransformer:
    """Build the scikit-learn preprocessing transformer.

    Missing-value imputation protects downstream estimators from nulls.
    Numeric scaling keeps distance-sensitive models such as logistic regression
    well-conditioned. Categorical one-hot encoding turns text labels into a
    numeric representation while ``handle_unknown='ignore'`` keeps serving
    robust to categories not seen during training.
    """

    preprocessing_cfg = params["preprocessing"]
    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy=preprocessing_cfg["numeric_imputer_strategy"]),
        )
    ]

    scaler_name = preprocessing_cfg.get("scaler")
    if scaler_name == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler_name in (None, "none"):
        pass
    else:
        raise ValueError(f"Unsupported scaler configured: {scaler_name}")

    categorical_imputer_kwargs: dict[str, Any] = {
        "strategy": preprocessing_cfg["categorical_imputer_strategy"],
    }
    if preprocessing_cfg["categorical_imputer_strategy"] == "constant":
        categorical_imputer_kwargs["fill_value"] = preprocessing_cfg["categorical_fill_value"]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(**categorical_imputer_kwargs)),
            ("encoder", _build_one_hot_encoder(preprocessing_cfg)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _stratify_target(target: pd.Series) -> pd.Series | None:
    class_counts = target.value_counts(dropna=False)
    if len(class_counts) < 2 or class_counts.min() < 2:
        return None
    return target


def save_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and save reproducible train/test split CSV files."""

    data_cfg = params["data"]
    train_path = resolve_project_path(data_cfg["train_path"])
    test_path = resolve_project_path(data_cfg["test_path"])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=_stratify_target(target),
    )

    train_data = x_train.copy()
    train_data[data_cfg["target_column"]] = y_train
    test_data = x_test.copy()
    test_data[data_cfg["target_column"]] = y_test

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"[preprocess] Saved training split: {train_path} ({len(train_data):,} rows)")
    print(f"[preprocess] Saved test split: {test_path} ({len(test_data):,} rows)")
    return train_data, test_data


def main() -> None:
    """Fit and persist the preprocessing pipeline on training data only."""

    params = load_params()
    data_cfg = params["data"]
    artifact_path = resolve_project_path(params["artifacts"]["preprocessing_pipeline_path"])

    dataset = load_dataset(params)
    features, target = split_features_target(
        dataset,
        target_column=data_cfg["target_column"],
        id_column=data_cfg.get("id_column"),
    )
    train_data, _ = save_train_test_split(features, target, params)
    x_train, _ = split_features_target(
        train_data,
        target_column=data_cfg["target_column"],
        id_column=data_cfg.get("id_column"),
    )

    numeric_columns, categorical_columns = get_column_types(x_train)
    column_transformer = build_preprocessor(numeric_columns, categorical_columns, params)
    preprocessing_pipeline = Pipeline(steps=[("preprocessor", column_transformer)])
    preprocessing_pipeline.fit(x_train)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing_pipeline, artifact_path)
    print(f"[preprocess] Saved fitted preprocessing pipeline: {artifact_path}")


if __name__ == "__main__":
    main()
