# Ankara Canonical Activity15 Spatial Permutation Importance

## Purpose

This diagnostic measures how much predictive ranking quality changes when one
canonical Activity15 feature is shuffled inside each validation fold.

The analysis uses the same fixed 5-km spatial folds and the same untuned Random
Forest and HistGradientBoosting configurations as the canonical ML evaluation.

Each model is fitted once per fold. A feature is then permuted only in the
validation data. The model is not retrained for each permutation.

## Configuration

- Canonical predictors: 15
- Models: Random Forest and HistGradientBoosting
- Permutation repeats per feature/fold: 5
- Primary importance statistic: pooled spatial OOF average-precision drop
- Supporting statistic: fold-level average-precision drop

## Random Forest

| Feature | Group | Pooled AP drop | Mean fold AP drop | Fold-drop std | Positive fold/repeat fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `poi_count` | activity | +0.077060 | +0.080743 | 0.095115 | 96.00% |
| `parking_area_ratio` | parking | +0.070434 | +0.075979 | 0.093602 | 84.00% |
| `poi_count_within_2000m` | activity | +0.067270 | +0.065336 | 0.096327 | 88.00% |
| `poi_count_within_1000m` | activity | +0.063087 | +0.065972 | 0.102139 | 72.00% |
| `distance_to_nearest_parking_m` | parking | +0.059631 | +0.064986 | 0.083330 | 92.00% |
| `main_road_length_m` | road | +0.059445 | +0.063950 | 0.078966 | 80.00% |
| `road_segment_count` | road | +0.059442 | +0.056789 | 0.092223 | 84.00% |
| `parking_count_within_1000m` | parking | +0.058739 | +0.046091 | 0.065486 | 80.00% |
| `road_density_km_per_km2` | road | +0.058114 | +0.052771 | 0.075026 | 88.00% |
| `distance_to_main_road_m` | road | +0.054539 | +0.059622 | 0.077905 | 76.00% |
| `main_road_segment_count` | road | +0.052722 | +0.044331 | 0.065835 | 72.00% |
| `parking_count_within_500m` | parking | +0.043928 | +0.010946 | 0.022482 | 80.00% |
| `parking_capacity_record_count` | parking | +0.000015 | +0.000021 | 0.000040 | 56.00% |
| `known_parking_capacity` | parking | -0.000001 | +0.000007 | 0.000015 | 20.00% |
| `parking_count` | parking | -0.004219 | +0.001246 | 0.012665 | 60.00% |
## HistGradientBoosting

| Feature | Group | Pooled AP drop | Mean fold AP drop | Fold-drop std | Positive fold/repeat fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `poi_count` | activity | +0.079130 | +0.106851 | 0.082406 | 100.00% |
| `parking_area_ratio` | parking | +0.068785 | +0.096125 | 0.087128 | 100.00% |
| `poi_count_within_1000m` | activity | +0.056650 | +0.077904 | 0.092525 | 76.00% |
| `distance_to_main_road_m` | road | +0.054312 | +0.076870 | 0.094395 | 88.00% |
| `road_segment_count` | road | +0.051567 | +0.040207 | 0.057962 | 68.00% |
| `distance_to_nearest_parking_m` | parking | +0.039423 | +0.047908 | 0.067083 | 56.00% |
| `poi_count_within_2000m` | activity | +0.014845 | +0.032862 | 0.061538 | 72.00% |
| `parking_count` | parking | +0.013661 | +0.017075 | 0.021165 | 60.00% |
| `main_road_segment_count` | road | +0.010596 | -0.006672 | 0.024509 | 12.00% |
| `road_density_km_per_km2` | road | +0.008822 | +0.013561 | 0.037617 | 68.00% |
| `known_parking_capacity` | parking | +0.000000 | +0.000000 | 0.000000 | 0.00% |
| `parking_capacity_record_count` | parking | +0.000000 | +0.000000 | 0.000000 | 0.00% |
| `main_road_length_m` | road | -0.005605 | +0.000480 | 0.004786 | 56.00% |
| `parking_count_within_500m` | parking | -0.009414 | +0.014689 | 0.023210 | 60.00% |
| `parking_count_within_1000m` | parking | -0.010241 | -0.028414 | 0.036186 | 40.00% |

## Interpretation Policy

A positive AP drop means prediction quality deteriorated after the feature was
shuffled, which is evidence that the fitted model depended on information carried
by that feature.

A zero or negative drop does not prove that a feature is useless. Correlated or
substitutable predictors can mask one another, and the target contains only 46
positive cells.

Permutation importance is model-specific predictive dependence, not causal
importance.

The activity variables are mapped OSM urban-activity proxies. Importance should
not be interpreted as direct evidence of EV demand, trips, employment, traffic,
or commercial turnover.

This diagnostic should be read together with the earlier feature-family
incremental-value and seed-stability experiments.

## Outputs

- `data/processed/ankara_canonical_spatial_permutation_importance.csv`
- `data/processed/ankara_canonical_spatial_permutation_importance_fold_drops.csv`
- `docs/ankara_canonical_spatial_permutation_importance.png`

## Generated At

2026-08-11T14:40:23.738613+00:00
