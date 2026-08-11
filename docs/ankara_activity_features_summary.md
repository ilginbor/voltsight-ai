# Ankara OSM Activity Features

## Purpose

These features provide an OpenStreetMap-based urban-activity proxy for the
Ankara 500-m analysis grid.

They are not direct observations of EV demand, traffic volume, trips, or
economic activity.

## Source Inventory

- Unique buffered OSM activity POIs: 21,513
- Grid rows: 102,745
- Grid cells with at least one local activity POI: 3,026
- Grid cells with zero local activity POIs: 99,719

Category source counts:

- retail_commercial: 10,182
- education: 1,504
- healthcare: 2,279
- transport_activity: 7,551

One OSM element may carry tags from more than one activity family. Total POI
counts deduplicate by OSM element identity; category counts are separate
descriptive flags and therefore are not required to sum to the total.

## Activity Taxonomy

### Retail / Commercial

The category includes:

- `shop=*` except explicit vacant/closed values
- `office=*` except explicit vacant/closed values
- selected commercial amenities such as marketplace, restaurants, cafes,
  fast-food/food-court venues, bars/pubs, banks and ATMs

### Education

Selected `amenity=*` values include school, college, university, kindergarten,
childcare, language school, music school and driving school.

### Healthcare

The category includes any non-empty `healthcare=*` plus selected healthcare
amenities: hospital, clinic, doctors, dentist and pharmacy.

### Transport Activity

The category includes:

- `public_transport=*`
- `highway=bus_stop`
- railway station/halt/tram-stop/subway-entrance features
- bus station, ferry terminal and taxi amenities
- aerodrome/terminal features

Charging stations and parking facilities are intentionally not part of this
activity taxonomy because VoltSight already models those feature families
separately.

## Spatial Method

Local counts assign every OSM POI to at most one Ankara 500-m grid cell.

The downloader keeps a 2.5-km buffer around Ankara. This allows 1-km and 2-km
neighborhood features for boundary cells to include nearby mapped activity just
outside the administrative boundary.

Neighborhood counts use exact Euclidean distance from each 500-m grid cell's
representative point to POI representative points in `EPSG:32636`.

Ways and relations are represented by the center returned by Overpass rather
than their full footprint. The resulting counts should therefore be interpreted
as mapped activity-presence proxies, not precise floor-area or capacity
measures.

## Output Features

- `poi_count`
- `retail_commercial_count`
- `education_count`
- `healthcare_count`
- `transport_activity_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`
- `retail_commercial_within_1000m`
- `education_within_1000m`
- `healthcare_within_1000m`
- `transport_activity_within_1000m`

## Feature Distribution

- `poi_count`: mean 0.21, median 0.00, p95 0.00, p99 5.00, max 303
- `retail_commercial_count`: mean 0.10, median 0.00, p95 0.00, p99 1.00, max 275
- `education_count`: mean 0.01, median 0.00, p95 0.00, p99 0.00, max 12
- `healthcare_count`: mean 0.02, median 0.00, p95 0.00, p99 0.00, max 15
- `transport_activity_count`: mean 0.07, median 0.00, p95 0.00, p99 2.00, max 49
- `poi_count_within_1000m`: mean 2.63, median 0.00, p95 4.00, p99 79.00, max 715
- `poi_count_within_2000m`: mean 10.51, median 0.00, p95 19.00, p99 324.00, max 1,565
- `retail_commercial_within_1000m`: mean 1.24, median 0.00, p95 0.00, p99 34.00, max 508
- `education_within_1000m`: mean 0.18, median 0.00, p95 0.00, p99 7.00, max 31
- `healthcare_within_1000m`: mean 0.28, median 0.00, p95 0.00, p99 9.00, max 79
- `transport_activity_within_1000m`: mean 0.92, median 0.00, p95 3.00, p99 28.00, max 210

## Interpretation Policy

OpenStreetMap completeness is spatially heterogeneous. A low POI count can mean
low mapped activity, incomplete mapping, or both.

These features must therefore be treated as urban-activity proxies rather than
ground-truth demand.

Before any canonical scoring change, the feature family should be audited for
coverage/redundancy and evaluated incrementally under the existing 5-km spatial
cross-validation design.

## Outputs

- `data/processed/ankara_grid_activity_features.gpkg`
- `data/processed/ankara_grid_activity_features.csv`

## Generated At

2026-08-11T12:34:08.290546+00:00
