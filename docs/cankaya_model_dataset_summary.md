# Çankaya Model Dataset Summary

## Source

- Source file: `cankaya_grid_charging_features.csv`
- Generated at: 2026-08-01T12:19:14.293190+00:00
- Source grid rows: 7,227
- Source columns: 31
- Missing source values: 0

## Supervised Training Dataset

- Output: `data/processed/cankaya_existing_station_training_dataset.csv`
- Rows: 7,227
- Columns: 23
- Target: `has_existing_charging_station`
- Positive rows: 10
- Negative rows: 7,217
- Positive-class rate: 0.1384%

The supervised dataset excludes charging-derived columns that would
directly reveal the target. It retains road and parking characteristics
for future experiments.

### Training Features

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `nearest_main_road_type`
- `parking_count`
- `known_parking_capacity`
- `parking_capacity_record_count`
- `parking_area_m2`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `parking_area_ratio`

## Candidate-Site Dataset

- Output: `data/processed/cankaya_candidate_site_dataset.csv`
- Candidate rows: 7,217
- Columns: 27
- Existing-station cells excluded: 10

The candidate dataset contains only grid cells without an existing
charging station. Charging proximity and neighborhood-count features
are retained because they can help identify underserved areas and
measure existing infrastructure coverage.

### Charging Context Features

- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

## Leakage Controls

The following charging-derived fields are not included in the
supervised existing-station training matrix:

- `charging_station_count`
- `known_charging_capacity`
- `charging_capacity_record_count`
- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

## Modeling Limitation

Only 10 of 7,227 grid cells contain an
existing charging station. The positive-class rate is
0.1384%. This is an extremely imbalanced target and is not
sufficient by itself for a reliable production classifier.

The candidate-site dataset should first be used for explainable
suitability scoring, ranking, clustering or weak-supervision
experiments. Additional verified stations, utilization data or expert
labels are required before treating the supervised target as strong
ground truth.
