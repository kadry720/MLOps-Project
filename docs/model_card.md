# Model Card: IEEE-CIS Fraud Detection

## Model Overview

This model predicts whether an e-commerce transaction is fraudulent using the
IEEE-CIS fraud detection dataset. The current best saved artifact is
`models/best_model.pkl`, selected from MLflow experiments by the configured
primary metric `f1`.

- Task: binary classification
- Positive class: `isFraud = 1`
- Best model family: `gradient_boosting`
- Registered model name: `FraudDetectionBestModel`
- Serving path: FastAPI app in `src/serving/`
- Model artifact path: `models/best_model.pkl`
- Experiment log: `docs/experiment_log.csv`

## Intended Use

The model is intended for an MLOps demonstration of fraud-risk scoring,
experiment tracking, model serving, monitoring, and retraining workflows. It can
rank transactions by fraud probability and support analyst triage in a simulated
fraud-detection system.

The model should not be used as the only basis for blocking payments, closing
accounts, or taking irreversible action against a customer. Production use would
require additional calibration, fairness review, business threshold review,
human escalation paths, and live monitoring.

## Training Data

The source data is the IEEE-CIS Fraud Detection dataset from Kaggle. The raw
transaction file contains 590,540 rows and 394 columns. The raw identity file
contains 144,233 rows and 41 columns. Raw data is not committed to Git; it is
managed by DVC.

For the current training run, `configs/params.yaml` uses a deterministic
100,000-row sample from the merged raw data. The processed sample has a fraud
rate of 2.561%. The split is stratified into:

- Training split: 80,000 rows, fraud rate 2.56125%
- Test split: 20,000 rows, fraud rate 2.56%

## Preprocessing

The model is trained through a serialized scikit-learn compatible pipeline. The
pipeline includes:

- Median imputation for numeric features
- Constant-token imputation for categorical features
- Robust scaling for numeric features
- One-hot encoding for categorical features with unknown-category handling
- Deterministic feature engineering for transaction amount, time, missingness,
  frequency encoding, target mean encoding, and interactions
- Variance thresholding and feature selection

SMOTE is configured but disabled for the current run. The advanced preprocessing
requirement is met through feature selection.

## Model Selection

MLflow tracks the model family, search grid, best hyperparameters, metrics,
plots, classification reports, CV outputs, and serialized model artifacts.
GridSearchCV is used for HPO. The current experiment comparison includes at
least these model families:

- Gradient Boosting
- Hist Gradient Boosting
- Random Forest
- Extra Trees
- Balanced Bagging
- Balanced Random Forest
- Easy Ensemble
- RUSBoost

The best run in `reports/mlflow_experiment_results.csv` is:

- Model: `gradient_boosting`
- Run ID: `faffd97210db4d6bb9d0ae8741a40ce8`
- Parameters: `learning_rate=0.05`, `max_depth=5`, `n_estimators=250`,
  `subsample=1.0`
- Decision threshold: `0.35`

## Evaluation Metrics

Evaluation was performed on the held-out test split with 20,000 rows.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.98585 |
| Precision | 0.81370 |
| Recall | 0.58008 |
| F1 | 0.67731 |
| F2 | 0.61542 |
| ROC AUC | 0.91142 |
| Average precision | 0.68081 |

Class-level report:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Legitimate (`0`) | 0.99 | 1.00 | 0.99 | 19,488 |
| Fraud (`1`) | 0.81 | 0.58 | 0.68 | 512 |

## Subgroup Metrics

Subgroup metrics were computed on the held-out test split using the saved best
model and decision threshold `0.35`.

### ProductCD

| Group | Rows | Fraud Rate | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C | 2,203 | 0.08579 | 0.78846 | 0.65079 | 0.71304 |
| H | 3,103 | 0.01837 | 0.86111 | 0.54386 | 0.66667 |
| R | 2,754 | 0.01162 | 0.70833 | 0.53125 | 0.60714 |
| S | 452 | 0.01991 | 1.00000 | 0.66667 | 0.80000 |
| W | 11,488 | 0.01959 | 0.83916 | 0.53333 | 0.65217 |

### Card Type

| Group | Rows | Fraud Rate | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| credit | 7,017 | 0.03520 | 0.80214 | 0.60729 | 0.69124 |
| debit | 12,980 | 0.02042 | 0.82584 | 0.55472 | 0.66366 |

## Limitations

- The model is trained on a static historical sample and may degrade under new
  fraud patterns.
- Fraud labels are highly imbalanced, so recall is materially lower than
  precision at the selected threshold.
- Features are anonymized and engineered, which limits interpretability and
  root-cause analysis.
- Some groups have smaller support, so subgroup metrics can be unstable.
- The model has not been calibrated for real financial loss, manual-review
  capacity, or customer-impact constraints.

## Ethical Considerations

Fraud models can create customer friction, false declines, and uneven treatment
across transaction types or customer cohorts. Any production system should:

- Use model scores as decision support, not automatic final judgment
- Maintain human review for high-impact actions
- Monitor false-positive and false-negative rates by meaningful business groups
- Provide an appeal or correction process for affected users
- Avoid using protected attributes or proxies without a fairness review

## Monitoring

The repository includes Evidently reports in `monitoring/evidently_reports/` and
Prometheus/Grafana monitoring for the serving API. The drift report injects
deterministic drift into six features and triggers the configured warning when
more than 20% of monitored features drift.

Operational monitoring should track:

- Data drift and data quality failures
- Prediction volume and prediction error rate
- Fraud probability and confidence distributions
- Model artifact availability
- Subgroup performance when labels become available

## Maintenance

Retraining should be considered when drift exceeds the configured threshold,
when live fraud recall drops below the accepted operating point, or when the
business changes fraud-review capacity. New model candidates should be logged in
MLflow, compared against the current production model, and promoted through the
registry workflow in `src/training/register_model.py`.
