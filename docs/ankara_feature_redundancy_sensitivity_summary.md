# Ankara Feature Redundancy Sensitivity

## Purpose

This experiment tests whether two near-deterministic feature pairs can be
removed without materially changing the spatially validated Ankara baseline
rankings.

The existing model hyperparameters, class-imbalance handling, and predefined
5-km spatial folds are kept unchanged.

No hyperparameter search is performed.

## Dataset

- Rows: 102,745
- Positive station cells: 46
- Spatial folds: 5
- Full predictors: 14
- Deduplicated predictors: 12

## Redundancy Audit

- `road_length_m` vs `road_density_km_per_km2`: Pearson 0.999999999989, median ratio 0.004, ratio std 2.9782952825e-06
- `parking_area_m2` vs `parking_area_ratio`: Pearson 0.999999999970, median ratio 4e-06, ratio std 4.30464969107e-08

The relationships are treated as near-deterministic transforms for this fixed
500-m grid. Small deviations from an exact constant ratio can arise from
stored precision or upstream rounding.

## Feature Sets

### Full 14

The original leakage-safe road and parking predictor set.

### Normalized 12

Drops:

- `road_length_m`
- `parking_area_m2`

Retains their normalized counterparts:

- `road_density_km_per_km2`
- `parking_area_ratio`

This is the preferred deduplicated representation if predictive performance is
not materially worse, because the retained variables express road intensity
and parking coverage independently of raw square-metre / metre scale.

### Raw 12

Drops the normalized counterparts instead:

- `road_density_km_per_km2`
- `parking_area_ratio`

and keeps:

- `road_length_m`
- `parking_area_m2`

The raw-12 branch is a sensitivity check rather than a proposed canonical
feature set.

## Spatial OOF Results

| Model | Feature set | Features | Pooled AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Top 5% recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Full 14 | 14 | 0.031053 | 0.050466 | 0.024544 | 0.967786 | 0.543478 | 0.891304 |
| Logistic Regression | Normalized 12 | 12 | 0.031057 | 0.050472 | 0.024529 | 0.967787 | 0.543478 | 0.891304 |
| Logistic Regression | Raw 12 | 12 | 0.031057 | 0.050472 | 0.024529 | 0.967787 | 0.543478 | 0.891304 |
| Random Forest | Full 14 | 14 | 0.050246 | 0.062796 | 0.060769 | 0.960337 | 0.369565 | 0.934783 |
| Random Forest | Normalized 12 | 12 | 0.083300 | 0.110030 | 0.126112 | 0.948239 | 0.434783 | 0.869565 |
| Random Forest | Raw 12 | 12 | 0.056635 | 0.100597 | 0.112510 | 0.978507 | 0.391304 | 0.891304 |
| HistGradientBoosting | Full 14 | 14 | 0.063939 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | 0.826087 |
| HistGradientBoosting | Normalized 12 | 12 | 0.063939 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | 0.826087 |
| HistGradientBoosting | Raw 12 | 12 | 0.063939 | 0.091578 | 0.078839 | 0.938261 | 0.478261 | 0.826087 |

## Normalized-12 Comparison

- Logistic Regression: full AP 0.031053, normalized-12 AP 0.031057, delta +0.000004.
- Random Forest: full AP 0.050246, normalized-12 AP 0.083300, delta +0.033054.
- HistGradientBoosting: full AP 0.063939, normalized-12 AP 0.063939, delta +0.000000.

## Interpretation Policy

This is a predictive redundancy sensitivity analysis, not a causal feature
selection procedure.

A deduplicated feature set should only replace the original 14-feature baseline
if performance remains comparable while interpretation becomes cleaner.

The `road_segment_count` relationship with road length/density is intentionally
not removed here. Although it can be highly correlated, segment count encodes a
different network characteristic and is not a deterministic scale conversion.

Likewise, main-road segment count and main-road length remain separate.

## Outputs

- `data/processed/ankara_feature_redundancy_sensitivity_metrics.csv`
- `data/processed/ankara_feature_redundancy_sensitivity_fold_metrics.csv`
- `data/processed/ankara_feature_redundancy_relations.csv`
- `docs/ankara_feature_redundancy_sensitivity.png`

## Generated At

2026-08-11T08:55:42.259690+00:00
