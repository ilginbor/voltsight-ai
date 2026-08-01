# Çankaya Charging-Station Source Merge

## Sources

- OpenStreetMap station records: 18
- Verified EPDK station records: 1
- Duplicate-distance threshold: 100 m
- Generated at: 2026-08-01T12:01:39.395323+00:00

## Result

- EPDK records treated as existing OSM stations: 0
- New EPDK-only stations added: 1
- Final merged station count: 19

## EPDK and OSM Distance Review

- `ŞRJ/2622` Çankaya 365 AVM: nearest OSM distance 1,844.18 m; duplicate=False

## Generated Files

- `data/interim/cankaya_charging_stations_merged.gpkg`
- `data/interim/cankaya_charging_stations_merged.csv`

## Provenance

Every merged record contains `data_source`, `source_osm` and
`source_epdk` fields.

The EPDK coordinate currently used is not an official coordinate
published in the downloaded EPDK report. Its confidence and source
are retained in the merged dataset.
