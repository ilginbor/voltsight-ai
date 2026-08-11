# Ankara Random Forest Activity Seed Stability

## Purpose

This robustness diagnostic checks whether the Random Forest gain from total OSM
activity context is stable across several fixed Random Forest seeds.

This is not hyperparameter tuning.

## Configuration

- Spatial folds: 5
- Random Forest trees: 400
- Maximum depth: 12
- Minimum leaf samples: 5
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- Seeds: 42, 43, 44, 45, 46

The only deliberate change across repeated runs is `random_state`.

Both the 12-feature baseline and 15-feature activity-context model resolve
`sqrt(n_features)` to three candidate predictors per split.

## Feature Sets

### Normalized 12

The deduplicated road-and-parking baseline.

### Normalized 12 + Total Activity Context

Adds:

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

## Seed-Aggregated Results

| Feature set | Features | Mean pooled AP | Seed AP std | Min AP | Max AP | Mean fold AP | Mean fold-AP std | Mean ROC-AUC | Mean top 1% recall | Mean top 5% recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized 12 | 12 | 0.075522 | 0.012882 | 0.063088 | 0.093259 | 0.102483 | 0.112608 | 0.956926 | 0.443478 | 0.913043 |
| Normalized 12 + total activity context | 15 | 0.087936 | 0.005176 | 0.080194 | 0.092916 | 0.100255 | 0.119070 | 0.948888 | 0.473913 | 0.891304 |

## Paired Seed Results

- Seed 42: baseline AP 0.083300, activity AP 0.091954, delta +0.008655, mean-fold AP delta -0.002771, top-1 delta +0.043478, top-5 delta +0.043478
- Seed 43: baseline AP 0.073908, activity AP 0.092916, delta +0.019008, mean-fold AP delta +0.006207, top-1 delta +0.021739, top-5 delta -0.021739
- Seed 44: baseline AP 0.064056, activity AP 0.080194, delta +0.016139, mean-fold AP delta -0.000249, top-1 delta +0.000000, top-5 delta -0.086957
- Seed 45: baseline AP 0.063088, activity AP 0.085652, delta +0.022563, mean-fold AP delta -0.004849, top-1 delta +0.043478, top-5 delta +0.000000
- Seed 46: baseline AP 0.093259, activity AP 0.088963, delta -0.004296, mean-fold AP delta -0.009479, top-1 delta +0.043478, top-5 delta -0.043478

- Mean paired activity minus baseline pooled-AP delta:
  +0.012414
- Paired pooled-AP delta standard deviation:
  0.010650
- Activity pooled AP higher in:
  4/5 seeds
- Activity mean-fold AP higher in:
  1/5 seeds
- Activity top-1% recall higher in:
  4/5 seeds
- Activity top-5% recall higher in:
  1/5 seeds

## Interpretation Policy

The preceding activity experiment showed positive pooled-AP deltas for Logistic
Regression, Random Forest, and HistGradientBoosting. Random Forest is stochastic,
and its seed-42 mean-fold AP did not improve despite the pooled-AP gain.

This diagnostic therefore focuses on whether the Random Forest activity gain is
repeatable rather than an artifact of one favorable seed.

A stable positive paired pooled-AP delta would strengthen the case for adopting
the three total-activity features as the next canonical ML feature-family
extension. Mixed or negative seed results would favor keeping them as an
experimental/contextual layer.

Only 46 positive existing-station cells are available. Seed stability cannot
replace independent external validation, and the 5-km spatial-block design
reduces but does not eliminate spatial dependence.

OSM activity remains a mapped urban-activity proxy rather than direct observed
EV demand.

## Output

- `data/processed/ankara_rf_activity_seed_stability_metrics.csv`

## Generated At

2026-08-11T13:06:06.376990+00:00
