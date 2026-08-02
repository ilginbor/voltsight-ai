# Ankara Road Feature Summary

## Inputs

- Ankara grid cells: 102,745
- Ankara road pieces: 177,714
- Ankara main-road pieces: 54,180
- Total road-network length: 29,274.14 km
- Total main-road length: 13,941.81 km
- Analysis CRS: EPSG:32636

## Processing

- Grid batch size: 5,000
- Batch count: 21
- Checkpoint directory: `data/interim/ankara_road_feature_batches_500m`

## Grid Features

- Cells with road data: 28,676
- Cells without road data: 74,069
- Mean road density: 1.14 km/km²
- Median road density: 0.00 km/km²
- Maximum road density: 38.39 km/km²
- Mean distance to a main road: 1,438.11 m
- Median distance to a main road: 958.35 m
- Maximum distance to a main road: 23,080.08 m

## Generated Features

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `nearest_main_road_type`

## Generated Outputs

- `data/processed/ankara_grid_road_features.gpkg`
- `data/processed/ankara_grid_road_features.csv`
- `docs/ankara_road_features_summary.md`

## Method

The merged Ankara road network was intersected with the
500 x 500 metre study grid in resumable batches.

Only the road geometry inside each grid cell contributed to road
length and density. Distance to the nearest main road was calculated
from each grid-cell centroid in the projected metre-based coordinate
system.

Batch CSV checkpoints allow an interrupted run to continue without
recalculating completed grid sections.

## Generated At

2026-08-02T14:41:28.195097+00:00
