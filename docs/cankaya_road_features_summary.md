# Çankaya Road Feature Summary

## Source

- Road data source: OpenStreetMap
- Download method: OSMnx polygon road-network query
- Network type: `drive`
- Download buffer: 1,000 meters
- Successful Overpass endpoint: https://overpass.private.coffee/api
- Generated at: 2026-07-31T09:47:45.553276+00:00

## Road Network

- Physical road edge count: 26,865
- Main-road edge count: 8,116
- Total downloaded road length: 3,190.27 km
- Total downloaded main-road length: 1,217.49 km

## Grid Features

- Grid cell count: 7,227
- Cells without a road segment: 3,881
- Mean road density: 5.48 km/km²
- Median road density: 0.00 km/km²
- Maximum road density: 46.25 km/km²
- Mean distance to a main road: 671.29 m
- Median distance to a main road: 375.49 m
- Maximum distance to a main road: 4,037.23 m

## Main-Road Classification

- `motorway`
- `motorway_link`
- `primary`
- `primary_link`
- `secondary`
- `secondary_link`
- `tertiary`
- `tertiary_link`
- `trunk`
- `trunk_link`

## Generated Features

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `nearest_main_road_type`

## Generated Outputs

- `data/interim/cankaya_drive_roads.gpkg`
- `data/processed/cankaya_grid_road_features.gpkg`
- `data/processed/cankaya_grid_road_features.geojson`
- `data/processed/cankaya_grid_road_features.csv`
- `docs/cankaya_road_features_preview.png`

## Method

The OpenStreetMap drive network was downloaded for the Çankaya
administrative boundary with an additional one-kilometer buffer.

The directed graph was converted to an undirected physical road
graph to avoid counting reciprocal travel directions as separate
physical roads.

Road geometries were intersected with each 250 x 250 meter grid
cell. Only the road length inside the corresponding grid cell was
included in that cell's feature value.

Distance to a main road was calculated from every grid-cell
centroid in the projected meter-based coordinate system.
