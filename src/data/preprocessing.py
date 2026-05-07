"""Reusable preprocessing utilities for the IEEE-CIS fraud project.

This module owns Component 2 of the MLOps deliverable. It builds a single
serializable scikit-learn ``Pipeline`` containing the preprocessing transformer
and any optional advanced preprocessing steps. SMOTE is intentionally kept out
of this saved preprocessing artifact because it changes the number of samples
and must be applied only to training folds inside the model-training pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "params.yaml"


def resolve_project_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path from config relative to the repository root."""

    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file used by preprocessing and training."""

    path = resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required_sections = {"data", "preprocessing", "training", "models", "artifacts"}
    missing_sections = sorted(required_sections - set(config))
    if missing_sections:
        raise ValueError(f"Configuration is missing required sections: {missing_sections}")
    return config


def _configured_nrows(sample_size: Any) -> int | None:
    if sample_size in (None, "null"):
        return None
    sample_size = int(sample_size)
    return sample_size if sample_size > 0 else None


def load_raw_data(
    transaction_path: str | Path,
    identity_path: str | Path,
    sample_size: int | None = None,
    id_column: str = "TransactionID",
) -> pd.DataFrame:
    """Load and merge the raw IEEE-CIS transaction and identity files.

    The transaction file is required because it contains the target column. The
    identity file is optional; when unavailable, the transaction data is still
    returned so notebooks can fail gracefully after a missing DVC pull.
    """

    transaction_file = resolve_project_path(transaction_path)
    identity_file = resolve_project_path(identity_path)
    nrows = _configured_nrows(sample_size)

    if not transaction_file.exists():
        raise FileNotFoundError(
            "Raw transaction data is missing. Run 'dvc pull' and check: "
            f"{transaction_file}"
        )

    transaction_df = pd.read_csv(transaction_file, nrows=nrows)
    if identity_file.exists():
        identity_df = pd.read_csv(identity_file)
        return transaction_df.merge(identity_df, on=id_column, how="left")

    print(f"[preprocess] Identity file not found; using transaction data only: {identity_file}")
    return transaction_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target, dropping the identifier."""

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    drop_columns = [target_column]
    if id_column and id_column in df.columns:
        drop_columns.append(id_column)

    return df.drop(columns=drop_columns), df[target_column]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic fraud features before column-wise preprocessing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def _category_values(series: pd.Series) -> pd.Series:
        return series.astype("object").where(~series.isna(), "__missing__").astype(str)

    def _combined_category_values(self, X: pd.DataFrame, columns: list[str]) -> pd.Series:
        combined = self._category_values(X[columns[0]])
        for column in columns[1:]:
            combined = combined + "_" + self._category_values(X[column])
        return combined

    def _target_mean_map(
        self,
        values: pd.Series,
        target: pd.Series,
        smoothing: float,
        global_mean: float,
    ) -> dict[str, float]:
        frame = pd.DataFrame({"value": values, "target": target}).dropna(subset=["target"])
        if frame.empty:
            return {}
        stats = frame.groupby("value")["target"].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        return smoothed.to_dict()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        preprocessing_config = _preprocessing_config(self.config)

        self.frequency_maps_: dict[str, dict[str, float]] = {}
        if preprocessing_config.get("use_frequency_encoding", False):
            for column in preprocessing_config.get("frequency_encode_features", []):
                if column in X.columns:
                    values = self._category_values(X[column])
                    self.frequency_maps_[column] = values.value_counts(normalize=True).to_dict()

        self.amount_global_median_ = 0.0
        self.amount_group_medians_: dict[str, dict[str, float]] = {}
        amount_column = preprocessing_config.get("amount_column", "TransactionAmt")
        if preprocessing_config.get("use_amount_group_features", False) and amount_column in X.columns:
            amounts = pd.to_numeric(X[amount_column], errors="coerce")
            if amounts.notna().any():
                self.amount_global_median_ = float(amounts.median())
            for group_column in preprocessing_config.get("amount_groupby_features", []):
                if group_column in X.columns:
                    groups = self._category_values(X[group_column])
                    medians = amounts.groupby(groups).median().dropna()
                    self.amount_group_medians_[group_column] = medians.to_dict()

        self.global_target_mean_ = 0.0
        self.target_mean_maps_: dict[str, dict[str, float]] = {}
        self.target_interaction_mean_maps_: dict[str, dict[str, float]] = {}
        if preprocessing_config.get("use_target_mean_encoding", False) and y is not None:
            if isinstance(y, pd.Series):
                target = pd.to_numeric(y.reindex(X.index), errors="coerce")
            else:
                target = pd.to_numeric(pd.Series(y, index=X.index), errors="coerce")

            if target.notna().any():
                self.global_target_mean_ = float(target.mean())
                smoothing = float(preprocessing_config.get("target_mean_smoothing", 20.0))
                for column in preprocessing_config.get("target_mean_encode_features", []):
                    if column in X.columns:
                        values = self._category_values(X[column])
                        self.target_mean_maps_[column] = self._target_mean_map(
                            values,
                            target,
                            smoothing,
                            self.global_target_mean_,
                        )

                for columns in preprocessing_config.get("target_mean_encode_interactions", []):
                    if len(columns) >= 2 and all(column in X.columns for column in columns):
                        feature_name = "_x_".join(columns)
                        values = self._combined_category_values(X, columns)
                        self.target_interaction_mean_maps_[feature_name] = self._target_mean_map(
                            values,
                            target,
                            smoothing,
                            self.global_target_mean_,
                        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        preprocessing_config = _preprocessing_config(self.config)
        engineered = X.copy()
        original_columns = engineered.columns.tolist()

        if not preprocessing_config.get("use_feature_engineering", False):
            return engineered

        if preprocessing_config.get("use_missing_pattern_features", False):
            missing_prefixes = preprocessing_config.get("missing_indicator_prefixes", [])
            original_frame = engineered[original_columns]
            engineered["missing_total_count"] = original_frame.isna().sum(axis=1)
            engineered["missing_total_ratio"] = original_frame.isna().mean(axis=1)
            for prefix in missing_prefixes:
                prefix_columns = [column for column in original_columns if column.startswith(prefix)]
                if prefix_columns:
                    engineered[f"{prefix}_missing_count"] = engineered[prefix_columns].isna().sum(axis=1)
                    engineered[f"{prefix}_missing_ratio"] = engineered[prefix_columns].isna().mean(axis=1)

        for column in preprocessing_config.get("log_transform_features", []):
            if column in engineered.columns:
                values = pd.to_numeric(engineered[column], errors="coerce")
                engineered[f"{column}_log1p"] = np.log1p(values.clip(lower=0))
                engineered[f"{column}_cents"] = (values.fillna(0) * 100).round().mod(100)
                engineered[f"{column}_is_round"] = (engineered[f"{column}_cents"] == 0).astype(int)

        time_column = preprocessing_config.get("transaction_time_column", "TransactionDT")
        if preprocessing_config.get("use_transaction_time_features", False) and time_column in engineered.columns:
            seconds = pd.to_numeric(engineered[time_column], errors="coerce")
            hour = np.floor(seconds / 3600.0).mod(24)
            weekday = np.floor(seconds / 86400.0).mod(7)
            engineered[f"{time_column}_day"] = seconds / 86400.0
            engineered[f"{time_column}_week"] = np.floor(seconds / 604800.0)
            engineered[f"{time_column}_hour"] = hour
            engineered[f"{time_column}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            engineered[f"{time_column}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
            engineered[f"{time_column}_weekday"] = weekday
            engineered[f"{time_column}_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
            engineered[f"{time_column}_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

        for column, mapping in getattr(self, "frequency_maps_", {}).items():
            if column in engineered.columns:
                values = self._category_values(engineered[column])
                engineered[f"{column}_frequency"] = values.map(mapping).fillna(0.0).astype(float)

        global_target_mean = float(getattr(self, "global_target_mean_", 0.0) or 0.0)
        for column, mapping in getattr(self, "target_mean_maps_", {}).items():
            if column in engineered.columns:
                values = self._category_values(engineered[column])
                engineered[f"{column}_target_mean"] = values.map(mapping).fillna(global_target_mean).astype(float)

        for feature_name, mapping in getattr(self, "target_interaction_mean_maps_", {}).items():
            columns = feature_name.split("_x_")
            if all(column in engineered.columns for column in columns):
                values = self._combined_category_values(engineered, columns)
                engineered[f"{feature_name}_target_mean"] = values.map(mapping).fillna(global_target_mean).astype(float)

        amount_column = preprocessing_config.get("amount_column", "TransactionAmt")
        if amount_column in engineered.columns:
            amounts = pd.to_numeric(engineered[amount_column], errors="coerce")
            global_median = float(getattr(self, "amount_global_median_", 0.0) or 0.0)
            fallback_median = global_median if global_median > 0 else 1.0
            for group_column, mapping in getattr(self, "amount_group_medians_", {}).items():
                if group_column in engineered.columns:
                    groups = self._category_values(engineered[group_column])
                    group_medians = groups.map(mapping).astype(float).fillna(fallback_median)
                    denominator = group_medians.replace(0, np.nan)
                    ratio = (amounts / denominator).replace([np.inf, -np.inf], np.nan)
                    engineered[f"{amount_column}_to_{group_column}_median"] = ratio
                    engineered[f"{amount_column}_minus_{group_column}_median"] = amounts - group_medians

        for column in preprocessing_config.get("email_domain_features", []):
            if column in engineered.columns:
                domains = self._category_values(engineered[column]).str.lower()
                parts = domains.str.split(".", n=1, expand=True)
                engineered[f"{column}_root"] = parts[0].where(domains != "__missing__", "__missing__")
                engineered[f"{column}_suffix"] = domains.str.rsplit(".", n=1).str[-1]

        if {"P_emaildomain", "R_emaildomain"}.issubset(engineered.columns):
            payer_email = self._category_values(engineered["P_emaildomain"]).str.lower()
            receiver_email = self._category_values(engineered["R_emaildomain"]).str.lower()
            engineered["email_domain_match"] = (
                (payer_email == receiver_email) & (payer_email != "__missing__")
            ).astype(int)

        for interaction in preprocessing_config.get("interaction_features", []):
            if len(interaction) != 2:
                continue
            left, right = interaction
            if left in engineered.columns and right in engineered.columns:
                left_values = self._category_values(engineered[left])
                right_values = self._category_values(engineered[right])
                engineered[f"{left}_x_{right}"] = left_values + "_" + right_values

        for column in preprocessing_config.get("categorical_like_numeric_features", []):
            if column in engineered.columns:
                values = engineered[column].to_numpy(dtype=object)
                engineered[column] = pd.Series(
                    [np.nan if pd.isna(value) else str(value) for value in values],
                    index=engineered.index,
                    dtype=object,
                )

        return engineered


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """Select up to k features without failing when fewer columns are available."""

    def __init__(self, score_func: Any = f_classif, k: int | str = 10):
        self.score_func = score_func
        self.k = k

    def fit(self, X: Any, y: pd.Series | None = None) -> "SafeSelectKBest":
        if self.k == "all":
            effective_k: int | str = "all"
        else:
            effective_k = min(int(self.k), X.shape[1])
        self.selector_ = SelectKBest(score_func=self.score_func, k=effective_k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        return self.selector_.transform(X)

    def get_support(self, indices: bool = False) -> Any:
        return self.selector_.get_support(indices=indices)


def identify_column_types(X: pd.DataFrame, config: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns from pandas dtypes."""

    preprocessing_config = _preprocessing_config(config or {})
    categorical_overrides = {
        column
        for column in preprocessing_config.get("categorical_like_numeric_features", [])
        if column in X.columns
    }
    numeric_features = [
        column
        for column in X.select_dtypes(include=["number", "bool"]).columns.tolist()
        if column not in categorical_overrides
    ]
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("preprocessing", config)


def _build_one_hot_encoder(preprocessing_config: dict[str, Any]) -> OneHotEncoder:
    encoder_kwargs: dict[str, Any] = {
        "handle_unknown": preprocessing_config.get(
            "encoder_handle_unknown",
            preprocessing_config.get("onehot_handle_unknown", "ignore"),
        )
    }

    max_categories = preprocessing_config.get(
        "encoder_max_categories",
        preprocessing_config.get("onehot_max_categories"),
    )
    min_frequency = preprocessing_config.get(
        "encoder_min_frequency",
        preprocessing_config.get("onehot_min_frequency"),
    )
    if max_categories is not None:
        encoder_kwargs["max_categories"] = max_categories
    if min_frequency is not None:
        encoder_kwargs["min_frequency"] = min_frequency

    try:
        return OneHotEncoder(sparse_output=False, **encoder_kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **encoder_kwargs)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
    use_column_selectors: bool = False,
) -> ColumnTransformer:
    """Build the ColumnTransformer for imputation, scaling, and encoding."""

    preprocessing_config = _preprocessing_config(config)
    numeric_imputer_kwargs: dict[str, Any] = {
        "strategy": preprocessing_config["numeric_imputer_strategy"],
        "add_indicator": bool(preprocessing_config.get("numeric_add_missing_indicator", False)),
    }
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(**numeric_imputer_kwargs))]

    scaler_name = preprocessing_config.get("scaler")
    if scaler_name == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler_name == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    elif scaler_name in (None, "none"):
        pass
    else:
        raise ValueError(f"Unsupported scaler configured: {scaler_name}")

    categorical_imputer_kwargs: dict[str, Any] = {
        "strategy": preprocessing_config["categorical_imputer_strategy"],
        "add_indicator": bool(preprocessing_config.get("categorical_add_missing_indicator", False)),
    }
    if preprocessing_config["categorical_imputer_strategy"] == "constant":
        categorical_imputer_kwargs["fill_value"] = preprocessing_config["categorical_fill_value"]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(**categorical_imputer_kwargs)),
            ("encoder", _build_one_hot_encoder(preprocessing_config)),
        ]
    )

    transformers: list[tuple[str, Pipeline, Any]] = []
    if use_column_selectors:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                make_column_selector(dtype_include=["number", "bool"]),
            )
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=["number", "bool"]),
            )
        )
    else:
        if numeric_features:
            transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_features))
        if categorical_features:
            transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
) -> Pipeline:
    """Build a single serializable sklearn Pipeline for all preprocessing.

    The base step is always a ColumnTransformer with numeric imputation,
    optional scaling, categorical imputation, and one-hot encoding. Optional
    polynomial features and feature selection are controlled by config.
    """

    preprocessing_config = _preprocessing_config(config)
    use_feature_engineering = preprocessing_config.get("use_feature_engineering", False)
    steps: list[tuple[str, Any]] = []

    if use_feature_engineering:
        steps.append(("feature_engineering", FraudFeatureEngineer(config)))

    steps.append(
        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features,
                config,
                use_column_selectors=use_feature_engineering,
            ),
        )
    )

    if preprocessing_config.get("use_polynomial_features", False):
        steps.append(
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=int(preprocessing_config["polynomial_degree"]),
                    include_bias=False,
                ),
            )
        )

    if preprocessing_config.get("use_variance_threshold", False):
        steps.append(("variance_threshold", VarianceThreshold()))

    if preprocessing_config.get("use_feature_selection", False):
        steps.append(
            (
                "feature_selection",
                SafeSelectKBest(
                    score_func=f_classif,
                    k=preprocessing_config["feature_selection_k"],
                ),
            )
        )

    return Pipeline(steps=steps)


def fit_preprocessor(preprocessor: Pipeline, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> Pipeline:
    """Fit the preprocessing pipeline on training data only."""

    if y_train is None:
        preprocessor.fit(X_train)
    else:
        preprocessor.fit(X_train, y_train)
    return preprocessor


def transform_features(preprocessor: Pipeline, X: pd.DataFrame) -> Any:
    """Transform features using a fitted preprocessing pipeline."""

    return preprocessor.transform(X)


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    """Serialize a fitted preprocessing pipeline with joblib."""

    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    return output_path


def load_preprocessor(path: str | Path) -> Pipeline:
    """Load a joblib-serialized preprocessing pipeline."""

    input_path = resolve_project_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessing artifact not found: {input_path}")
    return joblib.load(input_path)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load raw data from config and optionally persist a processed merge."""

    data_config = config["data"]
    dataset = load_raw_data(
        transaction_path=data_config["raw_transaction_path"],
        identity_path=data_config["raw_identity_path"],
        sample_size=data_config.get("sample_size"),
        id_column=data_config["id_column"],
    )

    processed_path = data_config.get("processed_path")
    if processed_path:
        output_path = resolve_project_path(processed_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
    return dataset


def _stratify_target(target: pd.Series) -> pd.Series | None:
    class_counts = target.value_counts(dropna=False)
    if len(class_counts) < 2 or class_counts.min() < 2:
        return None
    return target


def save_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and save reproducible train/test split CSV files."""

    data_config = config["data"]
    train_path = resolve_project_path(data_config["train_path"])
    test_path = resolve_project_path(data_config["test_path"])

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=_stratify_target(target),
    )

    train_data = X_train.copy()
    train_data[data_config["target_column"]] = y_train
    test_data = X_test.copy()
    test_data[data_config["target_column"]] = y_test

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    return train_data, test_data


def prepare_dataset(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Merge raw DVC inputs into the configured processed dataset."""

    config = load_config(config_path)
    dataset = load_dataset(config)
    output_path = resolve_project_path(config["data"]["processed_path"])
    print(f"[prepare] Saved processed dataset: {output_path} ({dataset.shape[0]} rows, {dataset.shape[1]} columns)")
    return output_path


def preprocess_dataset(config_path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, Path]:
    """Create deterministic train/test split CSVs from the processed dataset."""

    config = load_config(config_path)
    data_config = config["data"]
    processed_path = resolve_project_path(data_config["processed_path"])
    if processed_path.exists():
        dataset = pd.read_csv(processed_path)
    else:
        dataset = load_dataset(config)

    X, y = split_features_target(dataset, data_config["target_column"], data_config["id_column"])
    train_data, test_data = save_train_test_split(X, y, config)
    train_path = resolve_project_path(data_config["train_path"])
    test_path = resolve_project_path(data_config["test_path"])
    print(f"[preprocess] Saved train split: {train_path} ({train_data.shape[0]} rows)")
    print(f"[preprocess] Saved test split: {test_path} ({test_data.shape[0]} rows)")
    return train_path, test_path


def fit_preprocessing_artifact(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Fit and save the standalone preprocessing artifact from the train split."""

    config = load_config(config_path)
    data_config = config["data"]
    train_path = resolve_project_path(data_config["train_path"])
    if train_path.exists():
        train_data = pd.read_csv(train_path)
        X_train, y_train = split_features_target(
            train_data,
            data_config["target_column"],
            data_config.get("id_column"),
        )
    else:
        dataset = load_dataset(config)
        X, y = split_features_target(dataset, data_config["target_column"], data_config["id_column"])
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=data_config["test_size"],
            random_state=data_config["random_state"],
            stratify=_stratify_target(y),
        )

    numeric_features, categorical_features = identify_column_types(X_train, config)
    preprocessor = build_full_preprocessing_pipeline(numeric_features, categorical_features, config)
    fit_preprocessor(preprocessor, X_train, y_train)
    artifact_path = save_preprocessor(preprocessor, config["artifacts"]["preprocessing_pipeline_path"])
    print(f"[featurize] Saved fitted preprocessing pipeline: {artifact_path}")
    return artifact_path


def main(argv: list[str] | None = None) -> None:
    """Run one or all preprocessing DVC stages."""

    parser = argparse.ArgumentParser(description="Run preprocessing stages for the DVC pipeline.")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("all", "prepare", "preprocess", "featurize"),
        default="all",
        help="Stage to run. Defaults to the full preprocessing flow.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args(argv)

    if args.stage in ("all", "prepare"):
        prepare_dataset(args.config)
    if args.stage in ("all", "preprocess"):
        preprocess_dataset(args.config)
    if args.stage in ("all", "featurize"):
        fit_preprocessing_artifact(args.config)


# Backward-compatible aliases used by older project files/tests.
load_params = load_config
get_column_types = identify_column_types


if __name__ == "__main__":
    main()
