# Ankara Model Dataset Summary

## Source Dataset

- Source rows: 102,745
- Source columns: 31
- Existing-station grid cells: 46
- Grid cells without an existing station: 102,699
- Existing-station prevalence: 0.0448%
- Missing values in required model columns: 0

## Leakage-Safe Training Dataset

- Training rows: 102,745
- Training columns: 16
- Positive target rows: 46
- Negative target rows: 102,699
- Target: `has_existing_charging_station`
- Charging-derived predictor columns included: 0

### Predictor Columns

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `parking_count`
- `parking_area_m2`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`

## Candidate Dataset

- Candidate rows: 102,699
- Candidate columns: 20
- Existing-station cells excluded: 46

### Charging Context Retained for Suitability Analysis

- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

These variables describe the current infrastructure gap and are
retained for candidate suitability scoring. They are intentionally
excluded from the existing-station training predictors.

## Direct Charging Leakage Columns

- `charging_station_count`
- `known_charging_capacity`
- `charging_capacity_record_count`

These columns directly describe charging infrastructure inside the
grid cell and are excluded from both model predictors and candidate
scoring inputs.

## Leakage Policy

The existing-station classification dataset uses only road and parking
variables as predictors.

Current charging-station distance and neighborhood counts are not used
as predictors because they are functions of the same existing station
distribution represented by the target.

Candidate-site suitability is a separate decision-support task.
Charging context is allowed there because infrastructure scarcity is
an explicit component of site need rather than a predictor used to
reproduce the existing-station target.

## Outputs

- `data/processed/ankara_existing_station_training_dataset.csv`
- `data/processed/ankara_candidate_site_dataset.csv`

## Generated At

2026-08-10T06:26:21.183383+00:00
