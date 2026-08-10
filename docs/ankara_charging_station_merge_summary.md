# Ankara Charging Station Merge Summary

## Analysis Inventory

- Ankara-wide OSM charging stations: 68
- Supplemental verified-coordinate EPDK records: 1
- EPDK records matched to existing OSM stations: 0
- EPDK-only records added: 1
- Final analysis station count: 69
- Duplicate-distance threshold: 100 m

## Source Counts

- OSM: 68
- EPDK: 1

## EPDK / OSM Comparison

- ŞRJ/2622: nearest OSM 1,844.18 m; duplicate=False

## Important Scope Note

The OpenStreetMap inventory covers the Ankara study area.

The EPDK component is only the previously reviewed supplemental
coordinate dataset used in the Çankaya pilot. It must not be
interpreted as a complete province-wide spatial EPDK inventory.

The accepted EPDK coordinate is retained with its provenance and
coordinate-confidence fields.

## Outputs

- `data/interim/ankara_charging_stations_merged.gpkg`
- `data/interim/ankara_charging_stations_merged.csv`

## Generated At

2026-08-10T05:55:34.519471+00:00
