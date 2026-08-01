# Ankara Study Grid Summary

## Data Source

- Boundary source: OpenStreetMap / Nominatim
- Selected result: Ankara, Central Anatolia Region, Turkey
- Study-area key: `ankara`
- Boundary scope: province
- Study area: Ankara Province, Türkiye

## Grid Configuration

- Grid type: Fixed square grid
- Grid prefix: `ANK`
- Cell width: 500 meters
- Cell height: 500 meters
- Individual cell area: 250,000 square meters
- Generated cell count: 102,745

## Coordinate Systems

- Download and web CRS: EPSG:4326
- Analysis CRS: EPSG:32636

## Area Information

- Approximate province area: 25,680.96 square kilometers
- Total retained grid area: 25,686.25 square kilometers

## Generated Outputs

- `data/raw/ankara_boundary_osm.geojson`
- `data/processed/ankara_grid_500m.gpkg`
- `data/processed/ankara_grid_500m.geojson`
- `docs/ankara_grid_500m_preview.png`

## Method

A square grid was generated over the administrative boundary bounding
box. A cell was retained when its centroid fell inside or touched the
selected boundary. This preserves a consistent
500 x 500 meter square for every retained analysis
cell.

The boundary area was validated against the expected range configured
for `ankara`. This prevents an Ankara city boundary
from being mistaken for the complete Ankara province boundary.

## Generated At

2026-08-01T13:04:22.389559+00:00
