"""Train fraud detection models with MLflow experiment tracking.

Fraud detection is highly imbalanced, so model selection is driven by F1-score,
recall, precision, and ROC-AUC rather than accuracy alone. SMOTE is applied
inside the training pipeline, after preprocessing and only on training folds,
because it changes the number of samples and must never touch the held-out test
set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from mlflow.exceptions import MlflowException
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_params, resolve_project_path, split_features_target  # noqa: E402


SCORING_ALIASES = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
}


def _jsonify(value: Any) -> str | int | float | bool:
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


def load_splits(params: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/test split files created by the preprocessing stage."""

    data_cfg = params["data"]
    train_path = resolve_project_path(data_cfg["train_path"])
    test_path = resolve_project_path(data_cfg["test_path"])

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train/test split files are missing. Run 'python src/data/preprocessing.py' first."
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    x_train, y_train = split_features_target(
        train_data,
        target_column=data_cfg["target_column"],
        id_column=data_cfg.get("id_column"),
    )
    x_test, y_test = split_features_target(
        test_data,
        target_column=data_cfg["target_column"],
        id_column=data_cfg.get("id_column"),
    )
    return x_train, y_train, x_test, y_test


def load_preprocessing_pipeline(params: dict[str, Any]) -> Any:
    """Load the fitted preprocessing artifact produced by DVC."""

    artifact_path = resolve_project_path(params["artifacts"]["preprocessing_pipeline_path"])
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Preprocessing pipeline artifact is missing. "
            f"Run preprocessing first and check: {artifact_path}"
        )
    return joblib.load(artifact_path)


def build_sampler(y_train: pd.Series, params: dict[str, Any]) -> SMOTE | None:
    """Build a SMOTE sampler when class counts allow it."""

    preprocessing_cfg = params["preprocessing"]
    if not preprocessing_cfg.get("use_smote", False):
        return None

    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())
    if len(class_counts) < 2 or minority_count <= 1:
        print("[train] SMOTE skipped because the training target has too few minority samples.")
        return None

    configured_neighbors = int(preprocessing_cfg.get("smote_k_neighbors", 5))
    k_neighbors = min(configured_neighbors, minority_count - 1)
    return SMOTE(
        sampling_strategy=preprocessing_cfg.get("smote_sampling_strategy", "auto"),
        random_state=params["training"]["random_state"],
        k_neighbors=k_neighbors,
    )


def build_model_search_spaces(params: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Create model estimators and GridSearchCV parameter grids from config."""

    model_cfg = params["models"]
    random_state = params["training"]["random_state"]
    n_jobs = params["training"].get("n_jobs")

    logistic_cfg = model_cfg["logistic_regression"]
    random_forest_cfg = model_cfg["random_forest"]
    gradient_boosting_cfg = model_cfg["gradient_boosting"]

    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=logistic_cfg["max_iter"],
                solver=logistic_cfg["solver"],
                class_weight=logistic_cfg.get("class_weight"),
                random_state=random_state,
            ),
            {"classifier__C": logistic_cfg["C_values"]},
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight=random_forest_cfg.get("class_weight"),
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            {
                "classifier__n_estimators": random_forest_cfg["n_estimators"],
                "classifier__max_depth": random_forest_cfg["max_depth"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "classifier__n_estimators": gradient_boosting_cfg["n_estimators"],
                "classifier__learning_rate": gradient_boosting_cfg["learning_rate"],
                "classifier__max_depth": gradient_boosting_cfg["max_depth"],
            },
        ),
    }


def build_training_pipeline(preprocessor: Any, sampler: SMOTE | None, classifier: Any) -> ImbalancedPipeline:
    """Create an imbalanced-learn pipeline with preprocessing, optional SMOTE, and a classifier."""

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if sampler is not None:
        steps.append(("smote", sampler))
    steps.append(("classifier", classifier))
    return ImbalancedPipeline(steps=steps)


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a classifier on the untouched test split."""

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }

    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return metrics


def _create_cv(y_train: pd.Series, requested_folds: int, random_state: int) -> StratifiedKFold:
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Training requires at least two classes with at least two rows each for CV.")

    n_splits = min(int(requested_folds), int(class_counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _save_and_log_cv_results(grid_search: GridSearchCV, model_name: str, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cv_results_path = artifact_dir / f"{model_name}_cv_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)
    mlflow.log_artifact(str(cv_results_path), artifact_path="cv_results")


def _log_loss_curve_if_available(model: Any, model_name: str, artifact_dir: Path) -> None:
    classifier = model.named_steps.get("classifier")
    train_score = getattr(classifier, "train_score_", None)
    if train_score is None:
        return

    artifact_dir.mkdir(parents=True, exist_ok=True)
    curve_path = artifact_dir / f"{model_name}_loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_score) + 1), train_score)
    plt.title(f"{model_name} training loss")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    mlflow.log_artifact(str(curve_path), artifact_path="loss_curves")


def _set_mlflow_experiment(params: dict[str, Any]) -> None:
    training_cfg = params["training"]
    mlflow.set_tracking_uri(training_cfg["tracking_uri"])
    try:
        mlflow.set_experiment(training_cfg["experiment_name"])
    except MlflowException as exc:
        raise RuntimeError(
            "Could not connect to the MLflow tracking server. "
            "Start it with 'docker compose up mlflow' and verify http://localhost:5000."
        ) from exc


def run_training() -> None:
    """Run all configured experiments, log them to MLflow, and save the best model."""

    params = load_params()
    training_cfg = params["training"]
    primary_metric = training_cfg["primary_metric"]
    scoring = SCORING_ALIASES.get(primary_metric)
    if scoring is None:
        raise ValueError(f"Unsupported primary metric configured: {primary_metric}")

    x_train, y_train, x_test, y_test = load_splits(params)
    preprocessor = load_preprocessing_pipeline(params)
    sampler = build_sampler(y_train, params)
    cv = _create_cv(y_train, training_cfg["cv_folds"], training_cfg["random_state"])
    artifact_dir = resolve_project_path(params["artifacts"]["experiment_artifacts_dir"])
    best_model_path = resolve_project_path(params["artifacts"]["best_model_path"])

    _set_mlflow_experiment(params)

    best_score = float("-inf")
    best_model: Any | None = None
    best_model_name = ""
    search_spaces = build_model_search_spaces(params)

    for model_name, (classifier, param_grid) in search_spaces.items():
        print(f"[train] Starting experiment: {model_name}")
        pipeline = build_training_pipeline(preprocessor, sampler, classifier)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=training_cfg.get("n_jobs"),
            refit=True,
        )

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "primary_metric": primary_metric,
                    "scoring": scoring,
                    "cv_folds": cv.n_splits,
                    "use_smote": params["preprocessing"].get("use_smote", False),
                }
            )
            mlflow.log_params(_flatten_params(params["models"][model_name], prefix=f"{model_name}."))
            mlflow.log_params(_flatten_params(param_grid, prefix="grid."))

            grid_search.fit(x_train, y_train)
            metrics = evaluate_classifier(grid_search.best_estimator_, x_test, y_test)
            mlflow.log_params(_flatten_params(grid_search.best_params_, prefix="best."))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model")
            _save_and_log_cv_results(grid_search, model_name, artifact_dir)
            _log_loss_curve_if_available(grid_search.best_estimator_, model_name, artifact_dir)

            score = metrics[primary_metric]
            print(
                f"[train] Finished {model_name}: run_id={run.info.run_id}, "
                f"{primary_metric}={score:.4f}, best_params={grid_search.best_params_}"
            )
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_path)
    print(
        f"[train] Saved best model '{best_model_name}' to {best_model_path} "
        f"with {primary_metric}={best_score:.4f}"
    )


if __name__ == "__main__":
    run_training()
