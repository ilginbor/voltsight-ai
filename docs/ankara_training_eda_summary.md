# Ankara Training Dataset EDA

## Dataset

- Rows: 102,745
- Predictor features: 14
- Existing-station grid cells: 46
- Non-station grid cells: 102,699
- Positive prevalence: 0.0448%
- Negative-to-positive ratio: 2,232.59:1
- Missing predictor values: 0

## Interpretation

The existing-station learning problem is extremely imbalanced.

Accuracy is therefore not an appropriate primary evaluation metric.
A classifier predicting every cell as negative would achieve a very
high apparent accuracy while detecting no charging-station cells.

Future model evaluation should emphasize:

- precision
- recall
- F1
- average precision / PR-AUC
- ROC-AUC as a secondary metric
- spatial cross-validation stability

## Strongest Univariate Differences

- `parking_area_ratio`: positive median percentile 99.64, standardized mean difference 7.530
- `parking_area_m2`: positive median percentile 99.64, standardized mean difference 7.530
- `main_road_segment_count`: positive median percentile 98.86, standardized mean difference 4.916
- `parking_count_within_500m`: positive median percentile 99.17, standardized mean difference 4.288
- `main_road_length_m`: positive median percentile 98.72, standardized mean difference 4.275
- `road_length_m`: positive median percentile 98.39, standardized mean difference 4.164
- `road_density_km_per_km2`: positive median percentile 98.39, standardized mean difference 4.164
- `parking_count_within_1000m`: positive median percentile 99.28, standardized mean difference 4.074

The standardized mean difference is used only as a descriptive
screening statistic. It is not interpreted as causal importance.

The positive-median percentile reports where the median existing-
station cell falls inside the distribution of non-station cells.

Values far above 50 indicate that existing-station cells tend to have
higher feature values; values far below 50 indicate lower values.

## Generated Outputs

- `data/processed/ankara_training_feature_summary.csv`
- `docs/ankara_training_class_balance.png`
- `docs/ankara_training_positive_feature_profile.png`
- `docs/ankara_training_feature_correlation.png`

## Generated At

2026-08-10T08:29:57.825658+00:00
