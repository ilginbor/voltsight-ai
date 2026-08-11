# Ankara Feature-Group Ablation

## Purpose

This experiment isolates the contribution of road and parking predictor
families while preserving the existing Ankara baseline model settings and
the predefined 5-km spatial cross-validation folds.

No hyperparameter search is performed.

## Dataset

- Rows: 102,745
- Positive station cells: 46
- Spatial folds: 5
- Road predictors: 6
- Parking predictors: 8
- Combined predictors: 14

## Feature Groups

### Road only

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`

### Parking only

- `parking_count`
- `parking_area_m2`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`

## Models

The experiment reuses the fixed baseline configurations for:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Class-imbalance handling is also preserved. Logistic Regression and Random
Forest retain their class-weight settings, while HistGradientBoosting uses
the same balanced sample-weight calculation as its baseline.

## Spatial OOF Results

| Model | Features | Pooled AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Top 5% recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Road only | 0.041273 | 0.051952 | 0.049384 | 0.973562 | 0.391304 | 0.847826 |
| Logistic Regression | Parking only | 0.026325 | 0.045658 | 0.030605 | 0.938299 | 0.543478 | 0.739130 |
| Logistic Regression | Road + parking | 0.031053 | 0.050466 | 0.024544 | 0.967786 | 0.543478 | 0.891304 |
| Random Forest | Road only | 0.022483 | 0.056862 | 0.065612 | 0.931780 | 0.456522 | 0.869565 |
| Random Forest | Parking only | 0.013861 | 0.025189 | 0.026428 | 0.857044 | 0.282609 | 0.652174 |
| Random Forest | Road + parking | 0.050246 | 0.062796 | 0.060769 | 0.960337 | 0.369565 | 0.934783 |
| HistGradientBoosting | Road only | 0.025755 | 0.052747 | 0.049359 | 0.964410 | 0.456522 | 0.847826 |
| HistGradientBoosting | Parking only | 0.018383 | 0.039653 | 0.040689 | 0.831209 | 0.304348 | 0.543478 |
| HistGradientBoosting | Road + parking | 0.063939 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | 0.826087 |

## Model-Level Feature-Group Comparison

- Logistic Regression: road-only AP 0.041273, parking-only AP 0.026325, combined AP 0.031053.
- Random Forest: road-only AP 0.022483, parking-only AP 0.013861, combined AP 0.050246.
- HistGradientBoosting: road-only AP 0.025755, parking-only AP 0.018383, combined AP 0.063939.

These comparisons are descriptive ablations, not causal estimates of the
real-world effect of roads or parking on charging-station placement.

A lower score after removing a feature family indicates that the model's
ranking performance depends on information in that family under the current
dataset and validation design. Correlated predictors and sparse OSM coverage
can affect the magnitude of the observed differences.

## Evaluation Policy

Primary emphasis remains on rare-class ranking quality:

- pooled average precision / PR-AUC
- mean and standard deviation of fold AP
- top-1-percent recall
- top-5-percent recall

ROC-AUC is reported as a secondary metric.

Accuracy is intentionally not used as a primary metric.

## Spatial Validation Limitation

The same predefined 5-km spatial block folds are reused for every model and
feature group, making the comparisons directly paired by validation fold.

Cells inside one block stay together, but neighboring blocks can still be
assigned to different folds. The procedure therefore reduces local spatial
dependence without claiming to eliminate all spatial autocorrelation.

## Outputs

- `data/processed/ankara_feature_group_ablation_metrics.csv`
- `data/processed/ankara_feature_group_ablation_fold_metrics.csv`
- `data/processed/ankara_feature_group_ablation_oof_predictions.csv`
- `docs/ankara_feature_group_ablation_ap.png`

## Generated At

2026-08-11T08:37:10.319667+00:00
