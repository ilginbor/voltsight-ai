# Ankara Canonical Activity15 ML Evaluation

## Purpose

This evaluation establishes the forward-looking canonical Ankara ML reference
using the 15-feature predictor architecture selected after redundancy, population,
and OSM activity experiments.

Historical Full14 and Normalized12 experiments remain historical references.

## Dataset

- Rows: 102,745
- Positive existing-station cells: 46
- Predictors: 15
- Spatial folds: 5
- Spatial block size: 5 km

## Canonical Predictors

- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `parking_count`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`
- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

## Models

The existing untuned model configurations are reused unchanged:

- class-weighted standardized Logistic Regression
- Random Forest with 400 trees, depth 12, minimum leaf size 5,
  `max_features="sqrt"`, and `balanced_subsample`
- HistGradientBoosting with learning rate 0.05, 150 iterations, 15 leaf nodes,
  minimum leaf size 100, L2 regularization 1.0, balanced sample weights, and
  internal early stopping disabled

No hyperparameter search is performed.

## Spatial OOF Results

| Model | Pooled AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Top 5% recall | Top 1% lift | Top 5% lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.040671 | 0.064621 | 0.026692 | 0.955376 | 0.500000 | 0.847826 | 49.97x | 16.95x |
| Random Forest | 0.091954 | 0.107259 | 0.116055 | 0.963001 | 0.478261 | 0.913043 | 47.80x | 18.26x |
| HistGradientBoosting | 0.089014 | 0.130608 | 0.115073 | 0.899662 | 0.434783 | 0.826087 | 43.45x | 16.52x |

## Evaluation Policy

Average precision is primary because only a very small fraction of Ankara cells
contain known existing charging stations.

Top-1% and top-5% recall/lift remain decision-relevant because VoltSight is a
candidate-ranking system rather than a conventional balanced classifier.

The model scores are predictive ranking signals, not causal effects and not
calibrated probabilities.

The 5-km spatial block design reduces local train-validation dependence but
does not eliminate all spatial autocorrelation.

Only 46 positive cells are available, so fold-level variability remains an
important limitation.

OSM total-activity features are mapped urban-activity proxies. They do not
directly observe EV ownership, traffic, employment, trips, electricity-grid
capacity, or future charging demand.

## Historical Compatibility

Existing Full14 baseline outputs are not overwritten by this evaluation.

This script writes dedicated canonical-15 outputs so historical comparisons
remain reproducible.

## Outputs

- `data/processed/ankara_canonical_ml_model_metrics.csv`
- `data/processed/ankara_canonical_ml_model_fold_metrics.csv`
- `data/processed/ankara_canonical_ml_model_oof_predictions.csv`
- `docs/ankara_canonical_ml_model_comparison.png`

## Generated At

2026-08-11T13:54:06.379214+00:00
