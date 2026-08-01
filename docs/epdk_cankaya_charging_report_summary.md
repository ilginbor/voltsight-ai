# EPDK Çankaya Charging Report Summary

## Source

- Source organization: EPDK
- Source file: `sarjIstasyonlari.xls`
- File format: legacy Microsoft Excel (`.xls`)
- Generated at: 2026-08-01T10:04:52.900710+00:00

## Normalized Records

- Charging station count: 10
- Socket count: 18
- AC socket count: 17
- DC socket count: 1
- Unknown current-type socket count: 0
- Stations containing an AC socket: 9
- Stations containing a DC socket: 1
- Stations containing an address: 10
- Sockets with known power: 18
- Total reported socket power: 368.00 kW
- Maximum reported socket power: 60.00 kW

## Service Types

- HALKA_ACIK: 10

## Generated Outputs

- `data/interim/epdk_cankaya_charging_stations.csv`
- `data/interim/epdk_cankaya_charging_sockets.csv`
- `data/interim/epdk_cankaya_charging_report.json`

## Parsing Method

The downloaded EPDK report contains hierarchical rows. A station
metadata row is followed by one or more socket rows. Station metadata
was forward-filled only for the purpose of associating socket records
with their parent station.

Each station is represented once in the normalized station output.
Each physical EPDK socket is represented once in the normalized socket
output.

## Spatial Limitation

The downloaded report does not contain latitude or longitude columns.
Station addresses must therefore be geocoded and spatially validated
before these records can be merged with OpenStreetMap charging
stations.
