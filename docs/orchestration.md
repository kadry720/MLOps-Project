# Prefect Orchestration

Bonus B Part 1 uses Prefect to map the existing DVC training pipeline into a DAG.
The orchestration entrypoint is:

```bash
python src/orchestration/flow.py
```

## Environment

Prefect is included in `requirements.txt`:

```text
prefect==2.19.5
```

Use the project environment before running orchestration commands:

```bash
conda activate mlopsproject
```

Start MLflow before running the full training DAG:

```bash
docker compose up -d mlflow
```

## DAG Mapping

The Prefect flow maps the same stages defined in `dvc.yaml`:

```text
prepare -> preprocess -> featurize -> train -> register model
```

The tasks delegate to DVC commands so the DAG uses the existing reproducible
pipeline rather than a separate implementation:

```bash
python -m dvc repro prepare
python -m dvc repro preprocess
python -m dvc repro featurize
python -m dvc repro train
```

## Commands

Run the data-preparation portion only:

```bash
python src/orchestration/flow.py --skip-training
```

Run the full training DAG:

```bash
python src/orchestration/flow.py
```

Run training and then promote the best MLflow model:

```bash
python src/orchestration/flow.py --register-model
```

Optionally start the Prefect UI locally:

```bash
prefect server start
```

The UI is available at `http://127.0.0.1:4200`.
