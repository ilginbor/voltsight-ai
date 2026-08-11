# Ankara Canonical ML Dataset

## Purpose

This dataset defines the forward-looking canonical predictor architecture for
Ankara ML experiments after the road/parking redundancy and OSM activity
feature-family evaluations.

Historical baseline datasets and scripts remain valid historical references.

## Dataset

- Training rows: 102,745
- Candidate rows: 102,699
- Positive existing-station cells: 46
- Predictors: 15
- Target: `has_existing_charging_station`

## Canonical Feature Set

The canonical set contains the deduplicated normalized-12 road/parking
predictors plus three target-agnostic total OSM activity-context features.

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

## Removed Near-Deterministic Scale Duplicates

- `road_length_m`
- `parking_area_m2`

Their normalized counterparts remain in the canonical road/parking set.

## Added Activity Context

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

Category-specific OSM activity variables are not included in the canonical
feature set because the category-context sensitivity experiment did not show
the same model-general pooled-AP improvement as the total activity context.

Population features are not included in the canonical ML predictor set.

## Leakage Policy

Charging-derived context and direct charging-count variables are excluded from
the canonical ML predictors.

The target describes the existing charging-station distribution, so current
charging infrastructure cannot be used as a predictor in this classification
task.

Candidate-site suitability remains a separate decision-support layer where
charging scarcity is allowed as an explicit need component.

## Historical Compatibility

The existing `ankara_existing_station_training_dataset.csv` remains the
historical full-14 road/parking dataset.

This pipeline creates new canonical outputs instead of overwriting that
historical dataset.

## Outputs

- `data/processed/ankara_canonical_ml_training_dataset.csv`
- `data/processed/ankara_canonical_ml_candidate_dataset.csv`

## Generated At

2026-08-11T13:17:34.342528+00:00
