# Ankara Spatial Permutation Importance

## Purpose

This analysis measures validation-set dependence on individual road and
parking predictors for the two nonlinear Ankara baselines.

The Random Forest and HistGradientBoosting configurations are reused without
hyperparameter tuning. The same predefined 5-km spatial cross-validation
folds are used throughout.

## Dataset

- Rows: 102,745
- Positive station cells: 46
- Predictor features: 14
- Spatial folds: 5
- Permutation repeats per feature per fold: 5

## Method

For each model and spatial fold:

1. Fit the baseline model on the other four folds.
2. Score the untouched validation fold and calculate baseline AP / ROC-AUC.
3. Permute one predictor only inside that validation fold.
4. Re-score the already-fitted model without refitting it.
5. Record the decrease in AP and ROC-AUC.
6. Repeat each feature permutation 5 times with deterministic seeds.

The pooled metric also reconstructs a complete permuted OOF score vector for
each feature and repeat before calculating the pooled degradation.

A positive drop means that shuffling the feature reduced validation ranking
performance. A value near zero indicates little measurable dependence under
this experiment. Negative values are retained rather than clipped because
permutation noise or correlated predictors can occasionally improve the
metric by chance.

## Baseline Spatial OOF Metrics

- Random Forest: pooled AP 0.050246, pooled ROC-AUC 0.960337
- HistGradientBoosting: pooled AP 0.063939, pooled ROC-AUC 0.938261

## Highest Pooled-AP Degradation

- Random Forest: `parking_area_m2` (+0.036251), `parking_area_ratio` (+0.034802), `distance_to_nearest_parking_m` (+0.032107), `road_segment_count` (+0.029943), `parking_count_within_1000m` (+0.029175)
- HistGradientBoosting: `distance_to_nearest_parking_m` (+0.055669), `parking_area_m2` (+0.049499), `road_segment_count` (+0.047564), `distance_to_main_road_m` (+0.022725), `parking_count_within_500m` (+0.019866)

## Full Results

| Model | Feature | Group | Pooled AP drop | Fold AP drop | Fold AP std | ROC-AUC drop |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | `parking_area_m2` | parking | 0.036251 | 0.044738 | 0.058842 | 0.001951 |
| Random Forest | `parking_area_ratio` | parking | 0.034802 | 0.040643 | 0.056840 | 0.001689 |
| Random Forest | `distance_to_nearest_parking_m` | parking | 0.032107 | 0.034995 | 0.059538 | 0.005934 |
| Random Forest | `road_segment_count` | road | 0.029943 | 0.033285 | 0.049218 | 0.000351 |
| Random Forest | `parking_count_within_1000m` | parking | 0.029175 | 0.034325 | 0.048022 | -0.000708 |
| Random Forest | `parking_count_within_500m` | parking | 0.029121 | 0.026718 | 0.053241 | 0.000602 |
| Random Forest | `road_density_km_per_km2` | road | 0.022284 | 0.016371 | 0.019624 | 0.003661 |
| Random Forest | `main_road_length_m` | road | 0.021644 | 0.017479 | 0.019387 | 0.009120 |
| Random Forest | `road_length_m` | road | 0.018798 | 0.016501 | 0.032640 | 0.004099 |
| Random Forest | `distance_to_main_road_m` | road | 0.016285 | 0.014392 | 0.039720 | -0.008990 |
| Random Forest | `main_road_segment_count` | road | 0.015602 | 0.004901 | 0.012074 | 0.000931 |
| Random Forest | `known_parking_capacity` | parking | 0.000019 | 0.000019 | 0.000042 | 0.000066 |
| Random Forest | `parking_capacity_record_count` | parking | -0.000011 | -0.000016 | 0.000035 | -0.000016 |
| Random Forest | `parking_count` | parking | -0.002732 | 0.000982 | 0.035107 | 0.000179 |
| HistGradientBoosting | `distance_to_nearest_parking_m` | parking | 0.055669 | 0.073393 | 0.067188 | 0.164382 |
| HistGradientBoosting | `parking_area_m2` | parking | 0.049499 | 0.062057 | 0.057368 | 0.036939 |
| HistGradientBoosting | `road_segment_count` | road | 0.047564 | 0.066644 | 0.077434 | 0.040384 |
| HistGradientBoosting | `distance_to_main_road_m` | road | 0.022725 | 0.031229 | 0.038114 | 0.028443 |
| HistGradientBoosting | `parking_count_within_500m` | parking | 0.019866 | 0.016388 | 0.035824 | -0.000690 |
| HistGradientBoosting | `road_length_m` | road | 0.018895 | 0.009512 | 0.017280 | 0.084249 |
| HistGradientBoosting | `main_road_segment_count` | road | 0.016785 | 0.027788 | 0.034801 | 0.067064 |
| HistGradientBoosting | `parking_count_within_1000m` | parking | 0.011628 | 0.004147 | 0.038267 | 0.020793 |
| HistGradientBoosting | `parking_count` | parking | 0.008331 | -0.011615 | 0.031078 | -0.000149 |
| HistGradientBoosting | `road_density_km_per_km2` | road | 0.001384 | -0.009716 | 0.018229 | 0.003083 |
| HistGradientBoosting | `known_parking_capacity` | parking | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| HistGradientBoosting | `parking_area_ratio` | parking | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| HistGradientBoosting | `parking_capacity_record_count` | parking | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| HistGradientBoosting | `main_road_length_m` | road | -0.005543 | -0.012786 | 0.037279 | 0.036830 |

## Interpretation Policy

Permutation importance is model- and dataset-dependent. It is not a causal
estimate of the real-world effect of a road or parking variable on charging
station placement.

Correlated predictors can substitute for one another, reducing the apparent
importance of each individual variable. Sparse or incomplete OSM parking
coverage can also affect the measured parking importance.

The procedure uses validation-fold permutation rather than training-set
impurity importance, so the reported degradation is evaluated on held-out
spatial folds. However, neighboring spatial blocks can still occur in
different folds; the existing spatial CV design reduces local dependence but
does not eliminate all possible spatial autocorrelation.

## Outputs

- `data/processed/ankara_spatial_permutation_importance.csv`
- `data/processed/ankara_spatial_permutation_importance_fold_drops.csv`
- `docs/ankara_spatial_permutation_importance.png`

## Generated At

2026-08-11T08:46:06.898478+00:00
