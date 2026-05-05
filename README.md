# End-to-End MLOps Pipeline

DDSC611 Machine Learning Operations final project for IEEE fraud detection.

## Preprocessing and Training Workflow

The preprocessing stage builds a reusable scikit-learn artifact that performs
missing-value imputation, feature scaling, and categorical encoding. SMOTE is
kept in the training pipeline because it creates synthetic rows and must only be
applied to training folds. Training logs comparable runs to MLflow and promotes
the best registered model through Staging and Production.

1. Pull latest code:

   ```bash
   git pull origin main
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Pull DVC artifacts:

   ```bash
   python -m dvc pull
   ```

   The current DVC remote is DagsHub S3-compatible, so `requirements.txt`
   includes `dvc[s3]`.

4. Start MLflow:

   ```bash
   docker compose up mlflow
   ```

   The MLflow UI is available at `http://localhost:5000`.

5. Run preprocessing:

   ```bash
   python src/data/preprocessing.py
   ```

6. Run training:

   ```bash
   python src/training/train.py
   ```

7. Register the best model:

   ```bash
   python src/training/register_model.py
   ```

8. Run tests:

   ```bash
   pytest tests/unit/test_preprocessing.py
   ```

9. Reproduce the DVC pipeline:

   ```bash
   dvc repro
   ```

## Report Notes

- Preprocessing parameters, artifact paths, MLflow settings, and model search
  spaces live in `configs/params.yaml`.
- `models/preprocessing_pipeline.pkl` is the fitted preprocessing artifact used
  to prevent train/serve skew.
- Fraud data is imbalanced, so F1-score and recall are emphasized over accuracy.
- MLflow records hyperparameters, best GridSearchCV settings, metrics, CV result
  files, loss curves when the estimator exposes them, and serialized model
  artifacts.
