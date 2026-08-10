# Ankara Charging Feature Summary

## Inventory

- Final analysis charging stations: 69
- OSM-source stations: 68
- Supplemental EPDK-source stations: 1
- Stations with mapped AC connector: 13
- Stations with mapped DC connector: 7
- Stations with known capacity: 23

## Grid Results

- Grid cell count: 102,745
- Cells containing a charging station: 46
- Cells with a station within 1,000 m: 463
- Cells with a station within 2,000 m: 1,399
- Mean distance to nearest station: 35,980.89 m
- Median distance to nearest station: 31,631.14 m
- Maximum distance to nearest station: 125,676.81 m

## Processing

- Batch size: 5,000
- Batch count: 21
- Analysis CRS: EPSG:32636

## Generated Features

- `charging_station_count`
- `has_existing_charging_station`
- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `known_charging_capacity`
- `charging_capacity_record_count`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

## Scientific Use Warning

`charging_station_count` and
`has_existing_charging_station` describe the current station
distribution and must not be used as predictor variables when
training a model whose target is existing-station presence.

Distance and neighborhood charging variables also require
leakage-aware treatment in predictive modeling.

## EPDK Scope Note

The EPDK component is the previously reviewed supplemental
coordinate record from the Çankaya pilot. It is not a complete
province-wide spatial EPDK inventory.

## Generated At

2026-08-10T06:03:23.754159+00:00
