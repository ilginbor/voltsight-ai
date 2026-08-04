# Ankara Parking Dataset Summary

## Chunk Results

- Total chunks: 488
- Empty-success chunks: 401
- Non-empty chunks: 87
- Raw downloaded parking records: 4,623
- Records after Ankara buffer filter: 4,623
- Duplicate chunk occurrences removed: 1,664
- Parking IDs with multiple geometry variants: 0
- Final unique parking features: 2,959

## Geometry Statistics

- Point or multipoint features: 191
- Line or multiline features: 0
- Polygon or multipolygon features: 2,768
- Features with known numeric capacity: 113
- Total known capacity: 8,381
- Total mapped polygon area: 5,432,708.42 m²
- Analysis CRS: EPSG:32636

## Generated Outputs

- `data/interim/ankara_parking_features.gpkg`
- `data/interim/ankara_parking_merge_manifest.csv`
- `docs/ankara_parking_features_preview.png`

## Method

OpenStreetMap parking records downloaded from overlapping Ankara
chunks were merged using their stable OSM-derived `parking_id`.

When one parking feature appeared in multiple chunks, the record with
the most complete geometry was retained. Ties were resolved using the
lowest source chunk order.

Only parking features intersecting Ankara and its one-kilometre
download buffer were retained. This preserves nearby parking needed
for the 500-metre and 1,000-metre accessibility calculations.

## Data Limitation

OpenStreetMap parking coverage and capacity fields can be incomplete.
The dataset represents mapped parking availability rather than a
complete official parking inventory.

## Generated At

2026-08-04T12:01:01.648432+00:00
