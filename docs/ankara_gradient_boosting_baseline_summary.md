# Ankara Gradient Boosting Baseline

## Dataset

- Rows: 102,745
- Positive station cells: 46
- Predictor features: 14
- Spatial folds: 5
- Spatial block size: 5 km

## Model

The nonlinear baseline uses scikit-learn HistGradientBoostingClassifier.

Fixed configuration:

- learning rate: 0.05
- maximum iterations: 150
- maximum leaf nodes: 15
- minimum samples per leaf: 100
- L2 regularization: 1.0
- early stopping: disabled
- balanced training sample weights

No hyperparameter search was performed.

Disabling internal early stopping prevents the estimator from creating
a random internal validation split inside the predefined spatial
training folds.

Charging-derived features are excluded from the predictors.

## Spatial OOF Performance

- Pooled average precision: 0.063939
- Pooled ROC-AUC: 0.938261
- Mean fold average precision: 0.091578
- Fold AP standard deviation: 0.078839
- Mean fold ROC-AUC: 0.944689
- Fold ROC-AUC standard deviation: 0.062621

## Threshold 0.5 Diagnostic

- Precision: 0.031026
- Recall: 0.282609
- F1: 0.055914
- Predicted-positive cells: 419

Balanced sample weights change the effective class distribution seen
during fitting.

The resulting scores should therefore be treated as ranking scores
rather than calibrated real-world charging-station probabilities.

## Ranking Performance

### Top 1%

- Cells inspected: 1,028
- Existing-station cells recovered: 22
- Recall: 0.478261
- Lift: 47.80x

### Top 5%

- Cells inspected: 5,138
- Existing-station cells recovered: 38
- Recall: 0.826087
- Lift: 16.52x

## Fold-Level Results

- Fold 0: 10 positives, AP 0.068652, ROC-AUC 0.975909
- Fold 1: 9 positives, AP 0.180936, ROC-AUC 0.995245
- Fold 2: 9 positives, AP 0.167324, ROC-AUC 0.990301
- Fold 3: 9 positives, AP 0.036416, ROC-AUC 0.912301
- Fold 4: 9 positives, AP 0.004564, ROC-AUC 0.849686

## Interpretation Policy

This model is compared directly with Logistic Regression and Random
Forest using exactly the same predefined 5-km spatial folds.

Model selection should consider:

- pooled average precision
- fold AP stability
- top-1-percent recall
- top-5-percent recall
- lift over prevalence

ROC-AUC remains a secondary metric.

Accuracy is not used as a primary metric.

## Outputs

- `data/processed/ankara_gradient_boosting_baseline_oof_predictions.csv`
- `data/processed/ankara_gradient_boosting_baseline_fold_metrics.csv`
- `docs/ankara_gradient_boosting_baseline_pr_curve.png`

## Generated At

2026-08-10T09:11:47.121003+00:00
