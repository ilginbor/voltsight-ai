# Cankaya Charging Station Feature Summary

## Source

- Charging-station data source: OpenStreetMap
- OSM tag query: `amenity=charging_station`
- Download method: OSMnx polygon feature query
- Download buffer: 2,500 meters
- Successful Overpass endpoint: https://maps.mail.ru/osm/tools/overpass/api
- Generated at: 2026-08-01T09:22:16.106775+00:00

## Downloaded Charging Stations

- Unique mapped charging stations: 18
- Stations with known numeric capacity: 8
- Total known capacity: 296
- Stations with a mapped AC connector: 6
- Stations with a mapped DC connector: 3

## Grid Results

- Grid cell count: 7,227
- Cells containing a mapped charging station: 9
- Cells with a mapped station within 1,000 meters: 322
- Cells with a mapped station within 2,000 meters: 1,156
- Mean distance to nearest mapped station: 9,085.28 m
- Median distance to nearest mapped station: 5,417.67 m
- Maximum distance to nearest mapped station: 26,524.34 m

## Generated Columns

- `charging_station_count`
- `has_existing_charging_station`
- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `known_charging_capacity`
- `charging_capacity_record_count`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

## Generated Outputs

- `data/interim/cankaya_charging_stations.gpkg`
- `data/processed/cankaya_grid_charging_features.gpkg`
- `data/processed/cankaya_grid_charging_features.geojson`
- `data/processed/cankaya_grid_charging_features.csv`
- `docs/cankaya_charging_features_preview.png`

## Method

Charging stations were downloaded from OpenStreetMap for Cankaya and
an additional 2.5-kilometer buffer around the district.

Each mapped station was represented by one internal point for local
cell assignment and radius-based accessibility counts. Distances were
calculated from grid-cell centroids in the projected meter-based
coordinate system.

## Scientific Use Warning

`has_existing_charging_station` and `charging_station_count` are target
or descriptive columns. They must not be included as predictor inputs
when training a model to reproduce the current station distribution.

Distance and neighborhood-count columns also require leakage-aware
feature design before model training.

## Data Limitation

OpenStreetMap coverage, station capacity and connector tags may be
incomplete. These values describe mapped infrastructure rather than a
complete official charging-station inventory.
