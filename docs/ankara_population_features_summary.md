# Ankara Population Features

## Source

- Dataset: WorldPop 2025 constrained population, R2024B
- Country raster: Turkey
- Raster file: `tur_pop_2025_CN_100m_R2024B_v1.tif`
- Source CRS: EPSG:4326
- Source raster size: 22,983 x 7,552
- Source resolution: 0.00083333333 x 0.00083333333 degrees
- Source NoData: -99999.0
- Ankara source window: 2,525 x 3,699 pixels

WorldPop values are treated as population counts per raster cell.

## Method

1. Load the existing Ankara 500-m grid in `EPSG:32636`.
2. Read only the WorldPop window covering the Ankara grid extent.
3. Mask source pixels to the Ankara administrative boundary.
4. Replace source NoData and invalid values with zero.
5. Reproject population counts onto a grid-aligned 500-m raster using
   `Resampling.sum`.
6. Attach one 500-m population count to every Ankara grid cell.
7. Calculate population density using the fixed 0.25-km2 grid-cell area.
8. Calculate 1-km and 2-km neighborhood population using circular
   center-to-center neighborhoods on the 500-m lattice.

`Resampling.sum` is used because population is an extensive quantity. Bilinear
interpolation is intentionally not used for population counts.

The administrative boundary mask uses the center of each approximately 100-m
WorldPop pixel. Boundary-edge totals are therefore an approximation rather
than a cadastral or official population accounting.

## Rows

- Grid rows: 102,745
- Cells with population greater than zero: 42,625
- Cells with zero population: 60,120

## Population-Mass Diagnostic

- WorldPop population inside the Ankara boundary before reprojection: 6,164,255.64
- Population represented by Ankara 500-m grid cells after reprojection: 6,164,020.56
- Difference after reprojection/grid selection: -235.09
- Relative difference: -0.003814%

This diagnostic checks numerical population-mass preservation inside this
pipeline. It is not a validation against official TÜİK population statistics.

## Feature Summary

### Local 500-m Cell Population

- Total: 6,164,020.56
- Mean: 59.99
- Median: 0.00
- Maximum: 2,509.99

### Population Density

- Mean people/km2: 239.97
- Median people/km2: 0.00
- Maximum people/km2: 10,039.97

Because every analysis cell is exactly 0.25 km2,
`population_density_per_km2 = population_count / 0.25`.
The two variables are deterministic scale transforms and should not both be
used as predictors in the same ML model.

### Population Within 1 km

- Mean: 779.74
- Median: 4.41
- Maximum: 32,027.29

### Population Within 2 km

- Mean: 2,938.11
- Median: 78.52
- Maximum: 117,036.48

## ML Interpretation Policy

These variables represent modeled residential population demand context.

For future ML experiments, use either `population_count` or
`population_density_per_km2`, not both. The neighborhood totals can then test
whether surrounding demand adds information beyond the local 500-m cell.

Population features should first be audited descriptively and then evaluated
incrementally under the existing 5-km spatial cross-validation design.

## Outputs

- `data/processed/ankara_grid_population_features.gpkg`
- `data/processed/ankara_grid_population_features.csv`

## Generated At

2026-08-11T09:29:02.051793+00:00
