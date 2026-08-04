# Ankara Grid Parking Feature Summary

## Inputs

- Ankara grid cells: 102,745
- Unique parking features: 2,959
- Parking features with known capacity: 113
- Analysis CRS: EPSG:32636

## Processing

- Grid batch size: 5,000
- Batch count: 21
- Checkpoint directory: `data/interim/ankara_parking_feature_batches_500m`

## Accessibility Results

- Cells containing parking: 984
- Cells with parking within 500 metres: 1,906
- Cells with parking within 1,000 metres: 3,960
- Mean distance to nearest parking: 14,545.95 m
- Median distance to nearest parking: 11,823.59 m
- Maximum distance to nearest parking: 64,185.46 m
- Mean parking count within 500 metres: 0.09
- Mean parking count within 1,000 metres: 0.36

## Generated Features

- `parking_count`
- `parking_area_m2`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`

## Generated Outputs

- `data/processed/ankara_grid_parking_features.gpkg`
- `data/processed/ankara_grid_parking_features.csv`
- `docs/ankara_grid_parking_features_summary.md`

## Method

Unique OpenStreetMap parking features were connected to the Ankara
500 x 500 metre grid in resumable batches.

Representative points were used for cell assignment and radius-based
counts. Original parking geometries were used for nearest-distance
calculations.

Parking polygons were unioned before grid intersection, preventing
overlapping mapped parking areas from being counted twice.

## Data Limitation

OpenStreetMap parking coverage and capacity attributes can be
incomplete. These variables represent mapped parking accessibility,
not a complete official inventory.

## Generated At

2026-08-04T12:01:38.737698+00:00
