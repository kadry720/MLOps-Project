"""Prefect orchestration for the MLOps training pipeline.

The flow maps the existing DVC stages into a DAG:
validate data -> prepare -> preprocess -> featurize -> train -> evaluate -> optional registry.
Each task delegates to the same scripts tracked by ``dvc.yaml`` so orchestration
does not create a second implementation of the pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from prefect import flow, get_run_logger, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_command(command: Sequence[str]) -> None:
    """Run a project command and stream output to the active terminal."""

    logger = get_run_logger()
    logger.info("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


@task(name="validate_data")
def validate_data() -> str:
    """Run schema, quality, and drift validation before pipeline execution."""

    _run_command(
        [
            sys.executable,
            "src/pipeline/validate_pipeline.py",
            "--config",
            "configs/validation.yaml",
        ]
    )
    return "reports/validation/validation_summary.json"


@task(name="prepare")
def prepare_data() -> str:
    """Build the cleaned dataset from raw DVC inputs."""

    _run_command([sys.executable, "-m", "dvc", "repro", "prepare"])
    return "data/processed/cleaned_data.csv"


@task(name="preprocess")
def preprocess_data() -> list[str]:
    """Create deterministic train/test split files."""

    _run_command([sys.executable, "-m", "dvc", "repro", "preprocess"])
    return ["data/splits/train.csv", "data/splits/test.csv"]


@task(name="featurize")
def featurize_data() -> str:
    """Fit and save the preprocessing artifact used by training and serving."""

    _run_command([sys.executable, "-m", "dvc", "repro", "featurize"])
    return "models/preprocessing_pipeline.pkl"


@task(name="train")
def train_models() -> list[str]:
    """Run MLflow model experiments and save the best model."""

    _run_command([sys.executable, "-m", "dvc", "repro", "train"])
    return ["models/best_model.pkl", "reports/mlflow_experiment_results.csv"]


@task(name="evaluate")
def evaluate_model() -> str:
    """Validate the selected model against configured performance thresholds."""

    _run_command(
        [
            sys.executable,
            "src/evaluation/validate_model.py",
            "--config",
            "configs/params.yaml",
            "--validation-config",
            "configs/validation.yaml",
            "--allow-metrics-fallback",
        ]
    )
    return "reports/mlflow_experiment_results.csv"


@task(name="register_model")
def register_best_model() -> str:
    """Promote the best MLflow model through Staging and Production."""

    _run_command([sys.executable, "src/training/register_model.py"])
    return "FraudDetectionBestModel"


@flow(name="mlops-training-pipeline")
def mlops_training_pipeline(run_training: bool = True, register_model: bool = False) -> None:
    """Map DVC pipeline stages into a Prefect DAG.

    Args:
        run_training: When false, run only data preparation and featurization.
        register_model: When true, register/promote the best model after training.
    """

    validated = validate_data.submit()
    prepared = prepare_data.submit(wait_for=[validated])
    preprocessed = preprocess_data.submit(wait_for=[prepared])
    featurized = featurize_data.submit(wait_for=[preprocessed])

    if not run_training:
        return

    trained = train_models.submit(wait_for=[featurized])
    evaluated = evaluate_model.submit(wait_for=[trained])
    if register_model:
        register_best_model.submit(wait_for=[evaluated])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Prefect MLOps training DAG.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Run prepare, preprocess, and featurize only.",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register and promote the best MLflow model after training.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    mlops_training_pipeline(
        run_training=not args.skip_training,
        register_model=args.register_model,
    )
