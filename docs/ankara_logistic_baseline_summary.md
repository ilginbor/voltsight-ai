# Ankara Logistic Regression Baseline

## Dataset

- Rows: 102,745
- Positive station cells: 46
- Predictor features: 14
- Spatial folds: 5
- Spatial block size: 5 km

## Model

The baseline is an L2-regularized logistic regression using standardized
road and parking predictors.

`class_weight="balanced"` is used because the existing-station target is
extremely imbalanced.

Charging-derived variables are not used as predictors.

## Spatial Out-of-Fold Performance

- Logistic average precision / PR-AUC: 0.031053
- Dummy average precision: 0.000440
- Logistic ROC-AUC: 0.967786
- Dummy ROC-AUC: 0.491300

### Threshold 0.5 Diagnostic

- Precision: 0.006997
- Recall: 0.891304
- F1: 0.013884
- Predicted-positive cells: 5,860

Because class weighting changes the effective class distribution seen
during fitting, the logistic scores should not be interpreted as
calibrated real-world station probabilities.

The 0.5 threshold is therefore reported only as a diagnostic.

## Ranking Performance

### Top 1%

- Cells inspected: 1,028
- Existing-station cells recovered: 25
- Precision: 0.024319
- Recall: 0.543478
- Lift over prevalence: 54.32x

### Top 5%

- Cells inspected: 5,138
- Existing-station cells recovered: 41
- Precision: 0.007980
- Recall: 0.891304
- Lift over prevalence: 17.82x

## Fold-Level Results

- Fold 0: 20,549 validation rows, 10 positives, AP 0.0276, ROC-AUC 0.9428
- Fold 1: 20,549 validation rows, 9 positives, AP 0.0324, ROC-AUC 0.9808
- Fold 2: 20,549 validation rows, 9 positives, AP 0.0518, ROC-AUC 0.9833
- Fold 3: 20,549 validation rows, 9 positives, AP 0.0899, ROC-AUC 0.9767
- Fold 4: 20,549 validation rows, 9 positives, AP 0.0506, ROC-AUC 0.9673

## Largest Standardized Coefficients

- `distance_to_main_road_m`: -11.9248
- `distance_to_nearest_parking_m`: -1.4283
- `road_segment_count`: -0.7278
- `parking_count_within_500m`: +0.7211
- `parking_count`: -0.7060
- `road_length_m`: +0.6412
- `road_density_km_per_km2`: +0.6332
- `main_road_length_m`: -0.4013
- `main_road_segment_count`: +0.2567
- `parking_area_ratio`: +0.2184

Coefficient magnitude is descriptive rather than causal.

Correlated road and parking variables can redistribute coefficient
magnitude among one another.

## Evaluation Policy

Accuracy is intentionally not used as a primary model metric.

With only 46 positive cells among
102,745 total rows, a trivial negative classifier would
produce extremely high apparent accuracy while identifying no station
cells.

Primary evaluation focuses on ranking quality and rare-class retrieval:

- average precision / PR-AUC
- precision
- recall
- F1
- top-ranked-cell recall
- lift over prevalence

ROC-AUC is reported as a secondary metric.

## Spatial Validation Limitation

The predefined 5-km folds keep cells within the same spatial block
together.

Adjacent blocks can still belong to different folds, so this procedure
reduces local spatial dependence but does not eliminate every possible
form of spatial autocorrelation.

## Outputs

- `data/processed/ankara_logistic_baseline_oof_predictions.csv`
- `data/processed/ankara_logistic_baseline_fold_metrics.csv`
- `data/processed/ankara_logistic_baseline_coefficients.csv`
- `docs/ankara_logistic_baseline_pr_curve.png`
- `docs/ankara_logistic_baseline_coefficients.png`

## Generated At

2026-08-10T08:52:12.097081+00:00
