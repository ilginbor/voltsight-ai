# Ankara Random Forest Redundancy Seed Stability

## Purpose

This diagnostic checks whether the Random Forest difference between the
original 14-feature predictor set and the normalized 12-feature deduplicated
set is stable across Random Forest seeds.

This is a robustness diagnostic, not hyperparameter tuning.

## Configuration

- Spatial folds: 5
- Random Forest trees: 400
- Maximum depth: 12
- Minimum leaf samples: 5
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- Seeds: 42, 43, 44, 45, 46

The only deliberate change across repeated runs is the Random Forest
`random_state`.

## Feature Sets

- `full_14`: original 14 leakage-safe road and parking predictors
- `normalized_12`: drops `road_length_m` and `parking_area_m2`, retaining
  `road_density_km_per_km2` and `parking_area_ratio`

## Seed-Aggregated Results

| Feature set | Features | Mean pooled AP | Seed AP std | Min AP | Max AP | Mean fold AP | Mean fold-AP std | Mean ROC-AUC | Mean top 1% recall | Mean top 5% recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_14 | 14 | 0.051378 | 0.004367 | 0.044925 | 0.055470 | 0.068074 | 0.069510 | 0.959976 | 0.404348 | 0.921739 |
| normalized_12 | 12 | 0.075522 | 0.012882 | 0.063088 | 0.093259 | 0.102483 | 0.112608 | 0.956926 | 0.443478 | 0.913043 |

## Paired Pooled-AP Differences

- Seed 42: full-14 AP 0.050246, normalized-12 AP 0.083300, delta +0.033054
- Seed 43: full-14 AP 0.055418, normalized-12 AP 0.073908, delta +0.018490
- Seed 44: full-14 AP 0.044925, normalized-12 AP 0.064056, delta +0.019130
- Seed 45: full-14 AP 0.055470, normalized-12 AP 0.063088, delta +0.007618
- Seed 46: full-14 AP 0.050831, normalized-12 AP 0.093259, delta +0.042428

- Mean paired normalized-12 minus full-14 AP delta:
  +0.024144
- Paired delta standard deviation:
  0.013637
- Normalized-12 higher in:
  5/5 seeds

## Interpretation Policy

A consistent positive paired delta would indicate that the normalized
deduplicated representation is not merely benefiting from one favorable
Random Forest seed.

Because `max_features="sqrt"` randomly samples candidate predictors at each
split, duplicate or near-duplicate columns can change which latent information
is available to individual trees. This diagnostic therefore measures the
stability of that algorithm-feature-set interaction.

The result should not be interpreted as evidence that removing a variable has
a causal effect on charging-station placement.

## Output

- `data/processed/ankara_rf_redundancy_seed_stability_metrics.csv`

## Generated At

2026-08-11T09:05:55.782665+00:00
