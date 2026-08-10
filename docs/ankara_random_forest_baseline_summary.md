# Ankara Random Forest Baseline

## Dataset

- Rows: 102,745
- Positive cells: 46
- Predictors: 14
- Spatial folds: 5
- Spatial block size: 5 km

## Model

- Random Forest
- 400 trees
- Maximum depth: 12
- Minimum leaf samples: 5
- `class_weight="balanced_subsample"`
- `max_features="sqrt"`

Charging-derived variables are excluded from training predictors.

## Spatial OOF Performance

- Pooled average precision: 0.050246
- Pooled ROC-AUC: 0.960337
- Mean fold average precision: 0.062796
- Fold AP standard deviation: 0.060769
- Mean fold ROC-AUC: 0.961486
- Fold ROC-AUC standard deviation: 0.029641

## Threshold 0.5 Diagnostic

- Precision: 0.031802
- Recall: 0.195652
- F1: 0.054711

The threshold is diagnostic only and is not interpreted as a calibrated
real-world probability threshold.

## Ranking Performance

### Top 1%

- Cells inspected: 1,028
- Positive cells recovered: 17
- Recall: 0.369565
- Lift: 36.94x

### Top 5%

- Cells inspected: 5,138
- Positive cells recovered: 43
- Recall: 0.934783
- Lift: 18.69x

## Descriptive Feature Importance

- `road_length_m`: 0.171202
- `road_density_km_per_km2`: 0.166426
- `road_segment_count`: 0.161017
- `distance_to_nearest_parking_m`: 0.151920
- `main_road_segment_count`: 0.130246
- `main_road_length_m`: 0.069133
- `distance_to_main_road_m`: 0.058160
- `parking_count_within_1000m`: 0.045363
- `parking_count_within_500m`: 0.019755
- `parking_area_ratio`: 0.011560

Impurity-based Random Forest importance is descriptive and should not
be interpreted as causal importance.

Correlated predictors can share or distort feature importance.

## Generated At

2026-08-10T08:58:09.550561+00:00
