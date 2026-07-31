# Çankaya Study Grid Summary

## Data Source

- Boundary source: OpenStreetMap / Nominatim
- Selected result: Çankaya, Ankara, Central Anatolia Region, Turkey
- Study area: Çankaya, Ankara, Türkiye

## Grid Configuration

- Grid type: Fixed square grid
- Cell width: 250 meters
- Cell height: 250 meters
- Individual cell area: 62,500 square meters
- Generated cell count: 7,227

## Coordinate Systems

- Download and web CRS: EPSG:4326
- Analysis CRS: EPSG:32636

## Area Information

- Approximate district area: 450.93 square kilometers
- Total retained grid area: 451.69 square kilometers

## Generated Outputs

- `data/raw/cankaya_boundary_osm.geojson`
- `data/processed/cankaya_grid_250m.gpkg`
- `data/processed/cankaya_grid_250m.geojson`
- `docs/cankaya_grid_preview.png`

## Method

A square grid was generated over the district bounding box. A grid
cell was retained when its center point fell inside the Çankaya
administrative boundary. This approach preserves a consistent
250 x 250 meter shape for every analysis cell.

## Generated At

2026-07-31T07:39:28.747064+00:00
