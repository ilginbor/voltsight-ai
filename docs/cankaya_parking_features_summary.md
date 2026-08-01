# Cankaya Parking Feature Summary

## Source

- Parking data source: OpenStreetMap
- OSM tag query: `amenity=parking`
- Download method: OSMnx polygon feature query
- Download buffer: 1,000 meters
- Successful Overpass endpoint: https://maps.mail.ru/osm/tools/overpass/api
- Generated at: 2026-08-01T09:02:43.731178+00:00

## Downloaded Parking Data

- Unique parking feature count: 1,156
- Point or multipoint count: 106
- Polygon or multipolygon count: 1,050
- Features with known numeric capacity: 71
- Total known capacity: 2,890
- Total mapped polygon area: 1,867,326.86 m2

## Grid Accessibility Results

- Grid cell count: 7,227
- Cells containing a parking representative point: 453
- Cells with parking within 500 meters: 1,952
- Cells with parking within 1,000 meters: 2,957
- Mean distance to nearest parking: 6,112.69 m
- Median distance to nearest parking: 1,945.42 m
- Maximum distance to nearest parking: 22,063.93 m
- Mean parking count within 500 meters: 1.75
- Mean parking count within 1,000 meters: 6.94

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

- `data/interim/cankaya_parking_features.gpkg`
- `data/processed/cankaya_grid_parking_features.gpkg`
- `data/processed/cankaya_grid_parking_features.geojson`
- `data/processed/cankaya_grid_parking_features.csv`
- `docs/cankaya_parking_features_preview.png`

## Method

Parking features were downloaded from OpenStreetMap for Cankaya and
an additional one-kilometer buffer around the district.

Each parking feature was represented by one internal point for local
cell assignment and radius-based counts. Polygon parking geometries
were unioned before grid intersection so overlapping mapped areas
would not be double-counted.

Distance to the nearest parking feature was calculated from every
grid-cell centroid in the projected meter-based coordinate system.

## Data Limitation

OpenStreetMap parking coverage and capacity attributes can be
incomplete. These variables represent mapped parking accessibility,
not a complete official parking inventory.
