# Ankara OSM Activity Incremental Value

## Purpose

This experiment tests whether total OpenStreetMap urban-activity counts add
predictive ranking information beyond the deduplicated road-and-parking
baseline.

The experiment is incremental rather than a new tuned model search.

## Dataset

- Rows: 102,745
- Positive existing-station cells: 46
- Spatial folds: 5
- Spatial block size: 5 km
- Activity source: OpenStreetMap mapped activity POIs

## Feature Sets

### Normalized 12

The existing deduplicated road-and-parking baseline. It excludes
`road_length_m` and `parking_area_m2` while retaining their normalized
counterparts.

### Normalized 12 + Local POI Activity

Adds:

- `poi_count`

This tests whether activity mapped inside the local 500-m cell adds information
beyond road and parking features.

### Normalized 12 + POI Activity Context

Adds:

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

The feature family is deliberately based on total POI counts and spatial scale,
not on the earlier target-association ranking of individual POI categories.
This avoids choosing activity categories merely because they looked strongest
against the same 46 positive labels later used for evaluation.

The 12-, 13-, and 15-feature configurations also keep Random Forest
`max_features="sqrt"` at three candidate predictors per split, reducing one
possible estimator-configuration confound.

## Models

The existing untuned Logistic Regression, Random Forest, and
HistGradientBoosting configurations are reused unchanged.

The same predefined 5-km spatial folds and the same class-imbalance treatments
are retained. No hyperparameter search is performed.

## Spatial OOF Results

| Model | Feature set | Features | Pooled AP | Delta AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Delta top 1% | Top 5% recall | Delta top 5% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Normalized 12 | 12 | 0.031057 | +0.000000 | 0.050472 | 0.024529 | 0.967787 | 0.543478 | +0.000000 | 0.891304 | +0.000000 |
| Logistic Regression | Normalized 12 + local POI activity | 13 | 0.039613 | +0.008556 | 0.067736 | 0.029362 | 0.964389 | 0.521739 | -0.021739 | 0.869565 | -0.021739 |
| Logistic Regression | Normalized 12 + local + 1 km + 2 km POI activity | 15 | 0.040671 | +0.009614 | 0.064621 | 0.026692 | 0.955376 | 0.500000 | -0.043478 | 0.847826 | -0.043478 |
| Random Forest | Normalized 12 | 12 | 0.083300 | +0.000000 | 0.110030 | 0.126112 | 0.948239 | 0.434783 | +0.000000 | 0.869565 | +0.000000 |
| Random Forest | Normalized 12 + local POI activity | 13 | 0.086774 | +0.003474 | 0.088981 | 0.099621 | 0.973490 | 0.456522 | +0.021739 | 0.934783 | +0.065217 |
| Random Forest | Normalized 12 + local + 1 km + 2 km POI activity | 15 | 0.091954 | +0.008655 | 0.107259 | 0.116055 | 0.963001 | 0.478261 | +0.043478 | 0.913043 | +0.043478 |
| HistGradientBoosting | Normalized 12 | 12 | 0.063939 | +0.000000 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | +0.000000 | 0.826087 | +0.000000 |
| HistGradientBoosting | Normalized 12 + local POI activity | 13 | 0.081490 | +0.017550 | 0.121086 | 0.100376 | 0.923951 | 0.500000 | +0.021739 | 0.847826 | +0.021739 |
| HistGradientBoosting | Normalized 12 + local + 1 km + 2 km POI activity | 15 | 0.089014 | +0.025075 | 0.130608 | 0.115073 | 0.899662 | 0.434783 | -0.043478 | 0.826087 | +0.000000 |

## Full Activity-Context Delta Against Normalized 12

- Logistic Regression: pooled AP delta +0.009614, top-1% recall delta -0.043478, top-5% recall delta -0.043478.
- Random Forest: pooled AP delta +0.008655, top-1% recall delta +0.043478, top-5% recall delta +0.043478.
- HistGradientBoosting: pooled AP delta +0.025075, top-1% recall delta -0.043478, top-5% recall delta +0.000000.

## Interpretation Policy

Average precision is primary because only a very small fraction of Ankara grid
cells contain known existing charging stations. Top-1% and top-5% recall are
also reported because VoltSight is a candidate-ranking system.

A positive delta means the activity feature set improved spatial OOF ranking
under this fixed experiment. It is predictive evidence, not a causal estimate
of the real-world effect of urban activity on station placement.

The activity audit showed strong descriptive separation between known station
cells and non-station cells, but that same target was used to calculate those
descriptive statistics. Those SMD values are therefore context only, not
independent validation evidence.

OSM activity is a mapped urban-activity proxy. Low counts can reflect either
low activity or incomplete OSM mapping.

Only 46 positive station cells are available, so fold-level variability must
be considered alongside pooled metrics. The existing spatial-block design
reduces local train-validation dependence but does not eliminate all spatial
autocorrelation.

The historical full-14 baselines remain historical references. This experiment
uses normalized-12 as the deduplicated baseline for future feature-family
evaluation.

Category-specific POI features are intentionally deferred. They should only be
tested after this target-agnostic total-activity experiment establishes whether
the feature family has robust incremental value.

## Outputs

- `data/processed/ankara_activity_incremental_value_metrics.csv`
- `data/processed/ankara_activity_incremental_value_fold_metrics.csv`
- `data/processed/ankara_activity_incremental_value_oof_predictions.csv`
- `docs/ankara_activity_incremental_value.png`

## Generated At

2026-08-11T12:46:33.908194+00:00
