# Ankara Road Network Summary

## Chunk Results

- Total road chunks: 488
- Empty-success chunks: 34
- Raw downloaded road edges: 278,285
- Road pieces after core clipping: 177,714
- Exact boundary duplicates removed: 0
- Final merged road pieces: 177,714

## Network Statistics

- Main-road pieces: 54,180
- Total road length: 29,274.14 km
- Total main-road length: 13,941.81 km
- Analysis CRS: EPSG:32636

## Generated Outputs

- `data/interim/ankara_drive_roads.gpkg`
- `data/interim/ankara_road_merge_manifest.csv`
- `docs/ankara_road_network_preview.png`

## Method

Each buffered OpenStreetMap road download was clipped to its
corresponding non-overlapping eight-kilometre core chunk.

This removed the one-kilometre download-buffer overlap between
neighboring chunks. Exact road pieces remaining on shared chunk
boundaries were deduplicated using OpenStreetMap identifiers,
road classification and normalized geometry.

The resulting network is ready for intersection with the
500 x 500 metre Ankara study grid.

## Generated At

2026-08-02T13:50:13.681245+00:00
