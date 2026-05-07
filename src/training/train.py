"""Train fraud detection models with MLflow experiment tracking.

This script covers Component 3 and also creates the fitted Component 2
preprocessing artifact. All tunable values are loaded from
``configs/params.yaml``. SMOTE is applied only inside the training pipeline so
synthetic samples are created for training folds, never for the held-out test
set or the saved standalone preprocessing artifact.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from mlflow.exceptions import MlflowException
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (  # noqa: E402
    build_full_preprocessing_pipeline,
    identify_column_types,
    load_config,
    load_raw_data,
    resolve_project_path,
    split_features_target,
)


SCORING_ALIASES = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f1_score": "f1",
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "f2_score": make_scorer(fbeta_score, beta=2, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "balanced_random_forest": BalancedRandomForestClassifier,
    "easy_ensemble": EasyEnsembleClassifier,
    "balanced_bagging": BalancedBaggingClassifier,
    "rus_boost": RUSBoostClassifier,
}


def _jsonify(value: Any) -> str | int | float | bool:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, prefix=f"{name}."))
        else:
            flattened[name] = _jsonify(value)
    return flattened


def _log_params(params: dict[str, str | int | float | bool]) -> None:
    """Log MLflow params in small batches to avoid backend request limits."""

    items = list(params.items())
    for start in range(0, len(items), 100):
        mlflow.log_params(dict(items[start : start + 100]))


def _metric_name(configured_metric: str) -> str:
    if configured_metric == "f1_score":
        return "f1"
    if configured_metric == "f2_score":
        return "f2"
    return configured_metric


def _stratify_target(target: pd.Series) -> pd.Series | None:
    class_counts = target.value_counts(dropna=False)
    if len(class_counts) < 2 or class_counts.min() < 2:
        return None
    return target


def _ensure_artifact_directories(config: dict[str, Any]) -> None:
    artifact_config = config["artifacts"]
    for key in (
        "preprocessing_pipeline_path",
        "best_model_path",
        "results_path",
        "cv_results_dir",
        "plots_dir",
        "classification_reports_dir",
        "screenshots_dir",
    ):
        path = resolve_project_path(artifact_config[key])
        directory = path if path.suffix == "" else path.parent
        directory.mkdir(parents=True, exist_ok=True)


def _set_mlflow_experiment(config: dict[str, Any]) -> None:
    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    try:
        mlflow.set_experiment(mlflow_config["experiment_name"])
    except (MlflowException, OSError, ConnectionError) as exc:
        raise RuntimeError(
            "Could not connect to the MLflow tracking server. Run "
            "'docker compose up -d mlflow' and verify http://localhost:5000."
        ) from exc


def load_and_split_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load raw DVC data, merge identity data, and create a train/test split."""

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

    X, y = split_features_target(
        dataset,
        target_column=data_config["target_column"],
        id_column=data_config["id_column"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=_stratify_target(y),
    )

    train_path = data_config.get("train_path")
    test_path = data_config.get("test_path")
    if train_path and test_path:
        train_df = X_train.copy()
        train_df[data_config["target_column"]] = y_train
        test_df = X_test.copy()
        test_df[data_config["target_column"]] = y_test
        train_output = resolve_project_path(train_path)
        test_output = resolve_project_path(test_path)
        train_output.parent.mkdir(parents=True, exist_ok=True)
        test_output.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)

    return X_train, y_train, X_test, y_test


def load_training_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load prepared DVC split files, falling back to raw-data splitting for ad hoc runs."""

    data_config = config["data"]
    train_path = resolve_project_path(data_config["train_path"])
    test_path = resolve_project_path(data_config["test_path"])
    if train_path.exists() and test_path.exists():
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        X_train, y_train = split_features_target(
            train_data,
            data_config["target_column"],
            data_config.get("id_column"),
        )
        X_test, y_test = split_features_target(
            test_data,
            data_config["target_column"],
            data_config.get("id_column"),
        )
        return X_train, y_train, X_test, y_test

    return load_and_split_data(config)


def build_sampler(y_train: pd.Series, config: dict[str, Any]) -> SMOTE | None:
    """Create a SMOTE sampler from config when class counts make it valid."""

    preprocessing_config = config["preprocessing"]
    if not preprocessing_config.get("use_smote", False):
        return None

    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())
    if len(class_counts) < 2 or minority_count <= 1:
        print("[train] SMOTE skipped because the training target has too few minority samples.")
        return None

    configured_neighbors = int(preprocessing_config["smote_k_neighbors"])
    return SMOTE(
        sampling_strategy=preprocessing_config["smote_sampling_strategy"],
        random_state=preprocessing_config["smote_random_state"],
        k_neighbors=min(configured_neighbors, minority_count - 1),
    )


def build_model_search_spaces(config: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Build enabled estimators and GridSearchCV grids from config."""

    search_spaces: dict[str, tuple[Any, dict[str, list[Any]]]] = {}
    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", False):
            continue
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported model configured: {model_name}")
        estimator = MODEL_BUILDERS[model_name](**model_config.get("params", {}))
        search_spaces[model_name] = (estimator, model_config.get("grid", {}))

    if len(search_spaces) < 3:
        raise ValueError("At least three enabled model experiments are required.")
    return search_spaces


def build_training_pipeline(preprocessor: Any, sampler: SMOTE | None, classifier: Any) -> ImbalancedPipeline:
    """Create the imbalanced-learn training pipeline."""

    steps: list[tuple[str, Any]] = []
    if hasattr(preprocessor, "steps"):
        steps.extend(preprocessor.steps)
    else:
        steps.append(("preprocessing", preprocessor))

    if sampler is not None:
        steps.append(("smote", sampler))
    steps.append(("classifier", classifier))
    return ImbalancedPipeline(steps=steps)


def _positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = model.predict_proba(X)
    except ValueError:
        return None

    if probabilities.shape[1] < 2:
        return None

    classes = list(getattr(model, "classes_", []))
    positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
    return probabilities[:, positive_index]


def _predict_with_threshold(model: Any, X: pd.DataFrame, threshold: float | None) -> np.ndarray:
    probabilities = _positive_probabilities(model, X)
    if threshold is None or probabilities is None:
        return model.predict(X)
    return (probabilities >= threshold).astype(int)


def _score_predictions(y_true: pd.Series, predictions: np.ndarray, metric_name: str) -> float:
    if metric_name == "accuracy":
        return accuracy_score(y_true, predictions)
    if metric_name == "precision":
        return precision_score(y_true, predictions, zero_division=0)
    if metric_name == "recall":
        return recall_score(y_true, predictions, zero_division=0)
    if metric_name == "f1":
        return f1_score(y_true, predictions, zero_division=0)
    if metric_name == "f2":
        return fbeta_score(y_true, predictions, beta=2, zero_division=0)
    raise ValueError(f"Unsupported threshold tuning metric: {metric_name}")


def _select_decision_threshold(model: Any, X_train: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]) -> tuple[float | None, float]:
    training_config = config["training"]
    if not training_config.get("threshold_tuning", False):
        return None, float("nan")

    probabilities = _positive_probabilities(model, X_train)
    if probabilities is None:
        return None, float("nan")

    threshold_metric = _metric_name(training_config.get("threshold_metric", training_config["scoring_metric"]))
    candidates = sorted(
        {
            float(threshold)
            for threshold in training_config.get("threshold_grid", [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            if 0 < float(threshold) < 1
        }
    )
    if not candidates:
        return None, float("nan")

    best_threshold = candidates[0]
    best_score = float("-inf")
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        score = _score_predictions(y_train, predictions, threshold_metric)
        if score > best_score:
            best_threshold = threshold
            best_score = score

    return best_threshold, best_score


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate a trained classifier on the untouched test split."""

    predictions = _predict_with_threshold(model, X_test, threshold)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "f2": fbeta_score(y_test, predictions, beta=2, zero_division=0),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "decision_threshold": 0.5 if threshold is None else threshold,
    }

    probabilities = _positive_probabilities(model, X_test)
    if probabilities is not None and y_test.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
        metrics["average_precision"] = average_precision_score(y_test, probabilities)
    return metrics


def _create_cv(y_train: pd.Series, requested_folds: int, random_state: int) -> StratifiedKFold:
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Training requires at least two classes with at least two rows each for CV.")

    return StratifiedKFold(
        n_splits=min(int(requested_folds), int(class_counts.min())),
        shuffle=True,
        random_state=random_state,
    )


def _save_cv_results(grid_search: GridSearchCV, model_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_cv_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(output_path, index=False)
    return output_path


def _save_score_curve(grid_search: GridSearchCV, model_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_loss_curve.png"
    cv_results = pd.DataFrame(grid_search.cv_results_).reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    x_values = range(1, len(cv_results) + 1)
    plt.plot(x_values, cv_results["mean_test_score"], marker="o", linewidth=2)
    if "std_test_score" in cv_results:
        lower = cv_results["mean_test_score"] - cv_results["std_test_score"]
        upper = cv_results["mean_test_score"] + cv_results["std_test_score"]
        plt.fill_between(x_values, lower, upper, alpha=0.18)
    best_index = int(grid_search.best_index_) + 1
    plt.axvline(best_index, color="red", linestyle="--", linewidth=1.5, label="Best parameter set")
    plt.title("Grid Search Validation Score Curve")
    plt.xlabel("Grid search parameter combination")
    plt.ylabel(f"Mean CV {grid_search.scoring}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _save_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
    threshold: float | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    predictions = _predict_with_threshold(model, X_test, threshold)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(sorted(y_test.unique())))
    plt.xticks(tick_marks, sorted(y_test.unique()))
    plt.yticks(tick_marks, sorted(y_test.unique()))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            plt.text(column_index, row_index, value, ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _save_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
    threshold: float | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_classification_report.txt"
    predictions = _predict_with_threshold(model, X_test, threshold)
    threshold_text = "model_default" if threshold is None else f"{threshold:.3f}"
    report = f"decision_threshold: {threshold_text}\n\n"
    report += classification_report(y_test, predictions, zero_division=0)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def _save_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, output_dir: Path) -> Path | None:
    probabilities = _positive_probabilities(model, X_test)
    if probabilities is None or y_test.nunique() != 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _save_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
) -> Path | None:
    probabilities = _positive_probabilities(model, X_test)
    if probabilities is None or y_test.nunique() != 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {average_precision:.3f}")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _log_run_artifacts(
    grid_search: GridSearchCV,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
    threshold: float | None,
) -> None:
    artifact_config = config["artifacts"]
    cv_results_path = _save_cv_results(
        grid_search,
        model_name,
        resolve_project_path(artifact_config["cv_results_dir"]),
    )
    score_curve_path = _save_score_curve(
        grid_search,
        model_name,
        resolve_project_path(artifact_config["plots_dir"]),
    )
    confusion_matrix_path = _save_confusion_matrix(
        grid_search.best_estimator_,
        X_test,
        y_test,
        model_name,
        resolve_project_path(artifact_config["plots_dir"]),
        threshold,
    )
    classification_report_path = _save_classification_report(
        grid_search.best_estimator_,
        X_test,
        y_test,
        model_name,
        resolve_project_path(artifact_config["classification_reports_dir"]),
        threshold,
    )
    roc_curve_path = _save_roc_curve(
        grid_search.best_estimator_,
        X_test,
        y_test,
        model_name,
        resolve_project_path(artifact_config["plots_dir"]),
    )
    precision_recall_curve_path = _save_precision_recall_curve(
        grid_search.best_estimator_,
        X_test,
        y_test,
        model_name,
        resolve_project_path(artifact_config["plots_dir"]),
    )

    mlflow.log_artifact(str(cv_results_path), artifact_path="cv_results")
    mlflow.log_artifact(str(score_curve_path), artifact_path="loss_curves")
    mlflow.log_artifact(str(confusion_matrix_path), artifact_path="confusion_matrices")
    mlflow.log_artifact(str(classification_report_path), artifact_path="classification_reports")
    if roc_curve_path is not None:
        mlflow.log_artifact(str(roc_curve_path), artifact_path="roc_curves")
    if precision_recall_curve_path is not None:
        mlflow.log_artifact(str(precision_recall_curve_path), artifact_path="precision_recall_curves")


def main(config_path: str | Path = "configs/params.yaml") -> pd.DataFrame:
    """Run training, log MLflow experiments, and return comparison results."""

    config = load_config(config_path)
    _ensure_artifact_directories(config)
    _set_mlflow_experiment(config)

    training_config = config["training"]
    scoring_metric = _metric_name(training_config["scoring_metric"])
    scoring = SCORING_ALIASES.get(scoring_metric)
    if scoring is None:
        raise ValueError(f"Unsupported scoring metric configured: {scoring_metric}")

    X_train, y_train, X_test, y_test = load_training_data(config)
    numeric_features, categorical_features = identify_column_types(X_train, config)

    sampler = build_sampler(y_train, config)
    cv = _create_cv(y_train, training_config["cv_folds"], config["data"]["random_state"])
    search_spaces = build_model_search_spaces(config)
    results: list[dict[str, Any]] = []

    best_score = float("-inf")
    best_model: Any | None = None
    best_model_name = ""

    for model_name, (classifier, param_grid) in search_spaces.items():
        parameter_sets = 1
        for values in param_grid.values():
            parameter_sets *= len(values)
        total_cv_fits = parameter_sets * cv.n_splits
        print(
            f"[train] Starting experiment: {model_name} "
            f"({parameter_sets} parameter sets, {total_cv_fits} CV fits)",
            flush=True,
        )
        model_preprocessor = build_full_preprocessing_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
        )
        pipeline = build_training_pipeline(model_preprocessor, sampler, classifier)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=training_config["n_jobs"],
            refit=True,
            return_train_score=True,
            verbose=int(training_config.get("grid_search_verbose", 0)),
        )

        start_time = time.perf_counter()
        with mlflow.start_run(run_name=model_name) as run:
            _log_params(
                {
                    **_flatten_params(config["data"], prefix="data."),
                    **_flatten_params(config["preprocessing"], prefix="preprocessing."),
                    **_flatten_params(config["training"], prefix="training."),
                    **_flatten_params(config["mlflow"], prefix="mlflow."),
                    **_flatten_params(config["artifacts"], prefix="artifacts."),
                    **_flatten_params(config["models"][model_name].get("params", {}), prefix=f"{model_name}.params."),
                    **_flatten_params(param_grid, prefix=f"{model_name}.grid."),
                    "model_name": model_name,
                    "dataset_shape": f"{len(X_train) + len(X_test)}x{X_train.shape[1]}",
                    "train_shape": f"{X_train.shape[0]}x{X_train.shape[1]}",
                    "test_shape": f"{X_test.shape[0]}x{X_test.shape[1]}",
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "train_columns": X_train.shape[1],
                    "numeric_feature_count": len(numeric_features),
                    "categorical_feature_count": len(categorical_features),
                    "smote_enabled": bool(config["preprocessing"].get("use_smote", False)),
                    "cv_folds": cv.n_splits,
                }
            )

            grid_search.fit(X_train, y_train)
            training_time = time.perf_counter() - start_time
            threshold, threshold_train_score = _select_decision_threshold(
                grid_search.best_estimator_,
                X_train,
                y_train,
                config,
            )
            metrics = evaluate_classifier(grid_search.best_estimator_, X_test, y_test, threshold)
            metrics["threshold_train_score"] = threshold_train_score
            setattr(grid_search.best_estimator_, "decision_threshold_", metrics["decision_threshold"])
            mlflow.log_metrics(metrics)
            _log_params(_flatten_params(grid_search.best_params_, prefix="best."))
            _log_run_artifacts(grid_search, model_name, X_test, y_test, config, threshold)
            mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model")

            row = {
                "model": model_name,
                "best_params": json.dumps(grid_search.best_params_, sort_keys=True),
                "decision_threshold": metrics["decision_threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "training_time_seconds": training_time,
                "run_id": run.info.run_id,
            }
            results.append(row)

            score = metrics[scoring_metric]
            print(
                f"[train] Finished {model_name}: run_id={run.info.run_id}, "
                f"{scoring_metric}={score:.4f}, recall={metrics['recall']:.4f}, "
                f"threshold={metrics['decision_threshold']:.2f}, best_params={grid_search.best_params_}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model_path = resolve_project_path(config["artifacts"]["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_path)
    print(f"[train] Saved best model '{best_model_name}' to {best_model_path}")

    results_df = pd.DataFrame(results).sort_values(scoring_metric, ascending=False)
    results_path = resolve_project_path(config["artifacts"]["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"[train] Saved experiment comparison results: {results_path}")
    print(results_df.round(4).to_string(index=False))
    return results_df


if __name__ == "__main__":
    main()
