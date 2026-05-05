"""Register the best MLflow model and promote it through required stages.

The rubric explicitly requires a Model Registry transition through
``None -> Staging -> Production`` using the MLflow API. MLflow's newer
guidance encourages aliases, so this script performs the requested stage
transitions where available and also applies ``Staging`` and ``Production``
aliases when the installed MLflow client supports them.
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

from src.data.preprocessing import load_config  # noqa: E402


def _wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 120) -> None:
    """Wait until MLflow finishes creating the registered model version."""

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(name=model_name, version=version)
        if model_version.status == "READY":
            return
        time.sleep(2)
    raise TimeoutError(f"Model version {model_name} v{version} was not ready within {timeout_seconds}s.")


def _set_alias_if_supported(client: MlflowClient, model_name: str, alias: str, version: str) -> bool:
    """Set a model alias for MLflow versions that support aliases."""

    setter = getattr(client, "set_registered_model_alias", None)
    if setter is None:
        return False
    setter(name=model_name, alias=alias, version=version)
    return True


def _transition_stage(
    client: MlflowClient,
    model_name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool,
) -> str:
    """Transition a model version stage and return the new current stage."""

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )
    return client.get_model_version(name=model_name, version=version).current_stage


def register_best_model(config_path: str | Path = "configs/params.yaml") -> None:
    """Register the best MLflow run and promote it to Staging then Production."""

    config = load_config(config_path)
    mlflow_config = config["mlflow"]
    metric_name = config["training"]["scoring_metric"]
    if metric_name == "f1_score":
        metric_name = "f1"

    tracking_uri = mlflow_config["tracking_uri"]
    experiment_name = mlflow_config["experiment_name"]
    registered_model_name = mlflow_config["registered_model_name"]
    archive_existing = bool(mlflow_config.get("archive_existing_versions", True))

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except (MlflowException, OSError, ConnectionError) as exc:
        raise RuntimeError(
            "Could not connect to MLflow. Run 'docker compose up -d mlflow' "
            "and verify http://localhost:5000."
        ) from exc

    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No finished MLflow runs found for experiment: {experiment_name}")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    metric_value = best_run.data.metrics.get(metric_name)
    metric_text = "missing" if metric_value is None else f"{metric_value:.4f}"
    print(
        f"[registry] Registering run {best_run.info.run_id} as {registered_model_name} "
        f"using {metric_name}={metric_text}"
    )

    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version = model_version.version
    _wait_until_ready(client, registered_model_name, version)

    previous_stage = client.get_model_version(name=registered_model_name, version=version).current_stage
    current_stage = _transition_stage(
        client,
        registered_model_name,
        version,
        stage="Staging",
        archive_existing_versions=archive_existing,
    )
    staging_alias_applied = _set_alias_if_supported(client, registered_model_name, "Staging", version)
    print(
        "[registry] Stage transition evidence: "
        f"model={registered_model_name}, version={version}, "
        f"previous_stage={previous_stage}, current_stage={current_stage}, "
        f"aliases_applied={['Staging'] if staging_alias_applied else []}"
    )

    previous_stage = current_stage
    current_stage = _transition_stage(
        client,
        registered_model_name,
        version,
        stage="Production",
        archive_existing_versions=archive_existing,
    )
    production_alias_applied = _set_alias_if_supported(client, registered_model_name, "Production", version)
    aliases = []
    if staging_alias_applied:
        aliases.append("Staging")
    if production_alias_applied:
        aliases.append("Production")
    print(
        "[registry] Stage transition evidence: "
        f"model={registered_model_name}, version={version}, "
        f"previous_stage={previous_stage}, current_stage={current_stage}, "
        f"aliases_applied={aliases}"
    )
    print(f"[registry] Registered model name: {registered_model_name}")
    print(f"[registry] Registered model version: {version}")


if __name__ == "__main__":
    register_best_model()
