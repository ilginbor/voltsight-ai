# Ankara OSM Activity Category Context Sensitivity

## Purpose

This experiment asks whether the predictive signal from OSM activity is better
represented by total POI density, by target-agnostic activity categories, or by
a parsimonious mixture of both.

It is a feature-family sensitivity analysis, not hyperparameter tuning.

## Dataset

- Rows: 102,745
- Positive existing-station cells: 46
- Spatial folds: 5
- Spatial block size: 5 km

## Feature Sets

### Normalized 12

The existing deduplicated road-and-parking baseline.

### Total Activity Context

Adds:

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

This reproduces the strongest target-agnostic total-activity context tested in
the preceding incremental-value experiment.

### Category 1-km Context

Adds all four predefined OSM activity families at a single 1-km scale:

- `retail_commercial_within_1000m`
- `education_within_1000m`
- `healthcare_within_1000m`
- `transport_activity_within_1000m`

All four categories are included by taxonomy design rather than by selecting
the categories with the strongest target association.

### Parsimonious Mixed Activity Context

Adds:

- `poi_count`
- `poi_count_within_2000m`
- all four category 1-km variables above

The earlier predictor-only audit found `poi_count_within_1000m` and
`transport_activity_within_1000m` at approximately 0.90 Spearman correlation.
The mixed set therefore excludes the redundant 1-km total count while retaining
the broader 2-km total context.

This redundancy decision uses predictor-to-predictor correlation only, not
target performance.

## Random Forest Comparability

Random Forest uses `max_features=3` for every feature set in this sensitivity.

For the historical 12-feature baseline and the 15-feature total-context model,
`max_features="sqrt"` already resolves to three predictors per split. Holding
the value at three prevents the 16- and 18-feature category models from gaining
a larger per-split candidate pool merely because more predictors were added.

All other model settings remain unchanged.

## Spatial OOF Results

| Model | Feature set | Features | Pooled AP | Delta AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Delta top 1% | Top 5% recall | Delta top 5% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Normalized 12 | 12 | 0.031057 | +0.000000 | 0.050472 | 0.024529 | 0.967787 | 0.543478 | +0.000000 | 0.891304 | +0.000000 |
| Logistic Regression | Normalized 12 + total activity context | 15 | 0.040671 | +0.009614 | 0.064621 | 0.026692 | 0.955376 | 0.500000 | -0.043478 | 0.847826 | -0.043478 |
| Logistic Regression | Normalized 12 + category 1 km context | 16 | 0.027426 | -0.003632 | 0.053381 | 0.032835 | 0.959244 | 0.500000 | -0.043478 | 0.826087 | -0.065217 |
| Logistic Regression | Normalized 12 + parsimonious mixed activity | 18 | 0.037940 | +0.006883 | 0.057735 | 0.024220 | 0.942016 | 0.521739 | -0.021739 | 0.760870 | -0.130435 |
| Random Forest | Normalized 12 | 12 | 0.083300 | +0.000000 | 0.110030 | 0.126112 | 0.948239 | 0.434783 | +0.000000 | 0.869565 | +0.000000 |
| Random Forest | Normalized 12 + total activity context | 15 | 0.091954 | +0.008655 | 0.107259 | 0.116055 | 0.963001 | 0.478261 | +0.043478 | 0.913043 | +0.043478 |
| Random Forest | Normalized 12 + category 1 km context | 16 | 0.049607 | -0.033693 | 0.060195 | 0.062371 | 0.963439 | 0.478261 | +0.043478 | 0.934783 | +0.065217 |
| Random Forest | Normalized 12 + parsimonious mixed activity | 18 | 0.078891 | -0.004408 | 0.090554 | 0.126881 | 0.984195 | 0.543478 | +0.108696 | 0.978261 | +0.108696 |
| HistGradientBoosting | Normalized 12 | 12 | 0.063939 | +0.000000 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | +0.000000 | 0.826087 | +0.000000 |
| HistGradientBoosting | Normalized 12 + total activity context | 15 | 0.089014 | +0.025075 | 0.130608 | 0.115073 | 0.899662 | 0.434783 | -0.043478 | 0.826087 | +0.000000 |
| HistGradientBoosting | Normalized 12 + category 1 km context | 16 | 0.046350 | -0.017590 | 0.090270 | 0.081557 | 0.945523 | 0.500000 | +0.021739 | 0.826087 | +0.000000 |
| HistGradientBoosting | Normalized 12 + parsimonious mixed activity | 18 | 0.064694 | +0.000755 | 0.111473 | 0.112383 | 0.901280 | 0.521739 | +0.043478 | 0.847826 | +0.021739 |

## Interpretation Policy

Average precision is primary because there are only 46 positive existing-
station cells. Top-1% and top-5% recall remain decision-relevant ranking
diagnostics.

A category feature set should not be preferred merely because one pooled metric
is highest. Fold-level AP, fold variability, top-ranked recall, and consistency
across Logistic Regression, Random Forest, and HistGradientBoosting must be
considered together.

The OSM taxonomy is a mapped urban-activity proxy. It does not directly observe
trips, employment intensity, retail turnover, traffic, EV ownership, or power-
grid capacity.

The predefined 5-km spatial blocks reduce local dependence but do not eliminate
all spatial autocorrelation.

No category is selected or removed based on its earlier SMD against the same
target labels. This avoids using the full target dataset as a feature-selection
oracle before spatial-CV evaluation.

## Outputs

- `data/processed/ankara_activity_category_context_metrics.csv`
- `data/processed/ankara_activity_category_context_fold_metrics.csv`
- `data/processed/ankara_activity_category_context_oof_predictions.csv`
- `docs/ankara_activity_category_context.png`

## Generated At

2026-08-11T12:55:55.055712+00:00
