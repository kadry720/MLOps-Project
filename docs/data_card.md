# Data Card: IEEE-CIS Fraud Detection

## Dataset Overview

This project uses the IEEE-CIS Fraud Detection dataset from Kaggle to build a
binary fraud classifier. The target column is `isFraud`, where `1` means fraud
and `0` means legitimate.

The dataset contains transaction-level records with payment, card, address,
email-domain, identity, device, and anonymized engineered features. Raw data is
not committed to Git and is managed through DVC.

## Source and Access

- Source: Kaggle IEEE-CIS Fraud Detection competition dataset
- Raw transaction file: `data/raw/train_transaction.csv`
- Raw identity file: `data/raw/train_identity.csv`
- DVC pointer: `data/raw.dvc`
- DVC remote: configured in `.dvc/config`

The raw data is subject to Kaggle's dataset and competition terms. Team members
and reviewers should download or pull the data only through approved access
paths and should not commit raw CSV files to Git.

## Dataset Size

Local artifact statistics:

| Artifact | Rows | Columns | Notes |
| --- | ---: | ---: | --- |
| `data/raw/train_transaction.csv` | 590,540 | 394 | Contains target and transaction features |
| `data/raw/train_identity.csv` | 144,233 | 41 | Identity/device features keyed by `TransactionID` |
| `data/processed/cleaned_data.csv` | 100,000 | 434 | Deterministic configured training sample |
| `data/splits/train.csv` | 80,000 | 433 | Stratified training split |
| `data/splits/test.csv` | 20,000 | 433 | Stratified held-out test split |

Fraud rates:

| Artifact | Fraud Rate | Fraud Rows |
| --- | ---: | ---: |
| Raw transaction data | 3.49900% | 20,663 |
| Processed sample | 2.56100% | 2,561 |
| Train split | 2.56125% | 2,049 |
| Test split | 2.56000% | 512 |

## Key Columns

The validation schema tracks the following required columns:

| Column | Type | Description |
| --- | --- | --- |
| `TransactionID` | integer | Business key in raw and processed data |
| `isFraud` | integer | Binary target |
| `TransactionDT` | numeric | Relative transaction time |
| `TransactionAmt` | numeric | Transaction amount |
| `ProductCD` | string | Product code |
| `card1` | numeric | Anonymized card feature |
| `card4` | string | Card network |
| `card6` | string | Card type |
| `P_emaildomain` | string | Purchaser email domain |

The full model input has hundreds of additional anonymized numeric and
categorical features from transaction and identity tables.

## Collection and Labeling

The original data was released for the IEEE-CIS fraud detection benchmark.
Labels are historical fraud outcomes. The project does not control original
label creation, investigation policy, or time delay between transaction and
fraud confirmation.

## Preprocessing Decisions

The DVC pipeline creates reproducible artifacts through these stages:

1. `prepare`: merge and clean raw transaction and identity data into
   `data/processed/cleaned_data.csv`.
2. `preprocess`: create deterministic stratified train/test splits in
   `data/splits/`.
3. `featurize`: fit and save the preprocessing pipeline as
   `models/preprocessing_pipeline.pkl`.
4. `train`: train candidate models and save `models/best_model.pkl`.

Preprocessing includes:

- Median imputation for numeric values
- Constant-token imputation for categorical values
- Robust scaling
- One-hot encoding with unknown-category handling
- Transaction amount and time features
- Missingness pattern features
- Frequency encoding and target mean encoding
- Interaction features
- Feature selection

All tunable values are centralized in `configs/params.yaml`.

## Validation and Quality Checks

The project uses Pandera and custom validation checks configured in
`configs/validation.yaml`. The validation gate checks:

- Required columns and data types
- Numeric ranges
- Allowed categorical values
- Missing value thresholds
- Duplicate `TransactionID` values
- Freshness and timestamp sanity where available
- Outlier fractions
- Class imbalance
- Schema drift
- Numeric distribution drift using Kolmogorov-Smirnov tests
- Categorical distribution drift using Jensen-Shannon distance

Validation reports are written to `reports/validation/` and logs are written to
`logs/`.

## Drift Simulation

Monitoring compares a reference slice with:

- A clean held-out slice for the baseline report
- A deterministic perturbed slice for the drift report

The drift report injects shifts into:

- `TransactionAmt`
- `TransactionDT`
- `card1`
- `ProductCD`
- `P_emaildomain`
- `card4`

The current generated drift report flags 6 of 9 monitored columns as drifted,
which exceeds the configured 20% warning threshold.

## Known Biases and Risks

- The dataset is highly imbalanced, so models may under-detect fraud unless
  thresholds are tuned for recall.
- The data is historical and may not represent future fraud strategies.
- Some groups, such as smaller product or card-network segments, have lower
  support and less stable metrics.
- Anonymized features can hide proxy variables for geography, issuer, device, or
  customer segment.
- Labels may reflect prior fraud-detection policies and investigation capacity.

## Privacy and Security

Although the dataset is anonymized, it represents transaction behavior and
should be treated as sensitive. Raw CSV files and model artifacts are excluded
from Git and managed by DVC. Any shared reports should avoid exposing raw
transaction rows unless required for debugging and approved by the team.

## Licensing Notes

The dataset should be used according to Kaggle's terms and the IEEE-CIS
competition rules. This repository stores DVC pointers and derived project
artifacts, not the raw dataset itself.

## Recommended Use

This dataset is suitable for:

- MLOps pipeline development
- Fraud-detection modeling experiments
- Drift simulation and monitoring demonstrations
- Model-serving and CI/CD exercises

It is not sufficient by itself for production fraud decisions without additional
legal, privacy, fairness, calibration, and operational review.
