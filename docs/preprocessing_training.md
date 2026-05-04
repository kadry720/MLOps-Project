# Preprocessing and Experiment Tracking

## Why this preprocessing design is used

The preprocessing pipeline is a serializable scikit-learn object so training and
serving can apply the exact same transformations. Numeric missing values are
imputed with the median because fraud features often contain skewed monetary and
count distributions. Categorical missing values are filled with a stable
`missing` token so the model can learn whether absence itself is informative.
Standard scaling is included for models such as logistic regression, and one-hot
encoding converts categorical values into model-ready numeric features while
ignoring unseen production categories.

SMOTE is intentionally placed in the training pipeline instead of the standalone
preprocessor. It changes the number of samples by synthesizing minority-class
examples, so it must be applied only to the training folds during GridSearchCV.
This avoids contaminating validation or test data.

## Why MLflow is used

MLflow stores every experiment run with the model family, search grid, best
hyperparameters, test metrics, CV result artifacts, optional loss curves, and the
serialized model artifact. The registration script then selects the run with the
highest configured primary metric and promotes that model through Staging and
Production using `MlflowClient`, which gives clear evidence for the final report.
