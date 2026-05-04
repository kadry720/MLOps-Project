"""Register and promote the best MLflow model.

The final project requires explicit Model Registry evidence. This script finds
the best completed MLflow run for the configured experiment, registers its
serialized model artifact, then promotes the new version from None to Staging
and Production through the MLflow API.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_params  # noqa: E402


def _wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(name=model_name, version=version)
        if model_version.status == "READY":
            return
        time.sleep(2)
    raise TimeoutError(f"Model version {model_name} v{version} was not ready within {timeout_seconds}s.")


def register_best_model() -> None:
    """Register the best run and promote it to Staging and Production."""

    params = load_params()
    training_cfg = params["training"]
    tracking_uri = training_cfg["tracking_uri"]
    experiment_name = training_cfg["experiment_name"]
    primary_metric = training_cfg["primary_metric"]
    registered_model_name = training_cfg["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except MlflowException as exc:
        raise RuntimeError(
            "Could not connect to MLflow. Start it with 'docker compose up mlflow' first."
        ) from exc

    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{primary_metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No finished MLflow runs found for experiment: {experiment_name}")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    metric_value = best_run.data.metrics.get(primary_metric)
    metric_text = "missing" if metric_value is None else f"{metric_value:.4f}"
    print(
        f"[registry] Registering best run {best_run.info.run_id} "
        f"with {primary_metric}={metric_text}"
    )

    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    _wait_until_ready(client, registered_model_name, model_version.version)

    archive_existing = bool(training_cfg.get("archive_existing_versions", True))
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=archive_existing,
    )
    print(f"[registry] Promoted {registered_model_name} v{model_version.version}: None -> Staging")

    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=archive_existing,
    )
    print(f"[registry] Promoted {registered_model_name} v{model_version.version}: Staging -> Production")
    print(f"[registry] Registered model: {registered_model_name}, version: {model_version.version}")


if __name__ == "__main__":
    register_best_model()
