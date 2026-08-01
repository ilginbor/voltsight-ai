# Çankaya Study Grid Summary

## Data Source

- Boundary source: OpenStreetMap / Nominatim
- Selected result: Çankaya, Ankara, Central Anatolia Region, Turkey
- Study-area key: `cankaya`
- Boundary scope: district
- Study area: Çankaya, Ankara, Türkiye

## Grid Configuration

- Grid type: Fixed square grid
- Grid prefix: `CKY`
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

A square grid was generated over the administrative boundary bounding
box. A cell was retained when its centroid fell inside or touched the
selected boundary. This preserves a consistent
250 x 250 meter square for every retained analysis
cell.

The boundary area was validated against the expected range configured
for `cankaya`. This prevents an Ankara city boundary
from being mistaken for the complete Ankara province boundary.

## Generated At

2026-08-01T13:03:05.145484+00:00
