from __future__ import annotations

import os

# PostgreSQL/PostGIS can install machine-level PROJ/GDAL variables that point
# Rasterio/PyProj at an incompatible proj.db. Ignore only those external
# PostgreSQL paths so the Python geospatial wheels can use their bundled data.
for _environment_name in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    _environment_value = os.environ.get(_environment_name, "")
    if "PostgreSQL" in _environment_value:
        os.environ.pop(_environment_name, None)

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin, rowcol
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]

WORLDPOP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "worldpop"
    / "tur_pop_2025_CN_100m_R2024B_v1.tif"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_charging_features.gpkg"
)

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ankara_boundary_osm.geojson"
)

OUTPUT_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_population_features.gpkg"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_population_features.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_features_summary.md"
)

GRID_LAYER_NAME = "grid_charging_features"
OUTPUT_LAYER_NAME = "grid_population_features"

EXPECTED_GRID_CRS = "EPSG:32636"
EXPECTED_RASTER_CRS = "EPSG:4326"

GRID_CELL_SIZE_M = 500.0
GRID_CELL_AREA_KM2 = 0.25

NEIGHBORHOOD_RADII_M = (
    1_000,
    2_000,
)


def load_grid() -> gpd.GeoDataFrame:
    """Load and validate the Ankara 500-m analysis grid."""

    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Ankara grid not found: {GRID_PATH}"
        )

    grid = gpd.read_file(
        GRID_PATH,
        layer=GRID_LAYER_NAME,
    )[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    if grid.empty:
        raise ValueError(
            "Ankara grid is empty."
        )

    if grid.crs is None:
        raise ValueError(
            "Ankara grid CRS is missing."
        )

    if grid.crs.to_string().upper() != EXPECTED_GRID_CRS:
        raise ValueError(
            "Unexpected Ankara grid CRS: "
            f"{grid.crs}"
        )

    grid["grid_id"] = (
        grid["grid_id"]
        .astype(str)
    )

    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate Ankara grid IDs were found."
        )

    if grid.geometry.isna().any():
        raise ValueError(
            "Missing Ankara grid geometry was found."
        )

    if (
        ~grid.geometry.is_valid
    ).any():
        raise ValueError(
            "Invalid Ankara grid geometry was found."
        )

    bounds = grid.geometry.bounds

    widths = (
        bounds["maxx"]
        - bounds["minx"]
    ).to_numpy(
        dtype=float
    )

    heights = (
        bounds["maxy"]
        - bounds["miny"]
    ).to_numpy(
        dtype=float
    )

    if not np.allclose(
        widths,
        GRID_CELL_SIZE_M,
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError(
            "Grid cells are not consistently 500 m wide."
        )

    if not np.allclose(
        heights,
        GRID_CELL_SIZE_M,
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError(
            "Grid cells are not consistently 500 m high."
        )

    return gpd.GeoDataFrame(
        grid,
        geometry="geometry",
        crs=grid.crs,
    )


def load_boundary(
    target_crs: object,
) -> gpd.GeoDataFrame:
    """Load the Ankara administrative boundary in the requested CRS."""

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Ankara boundary not found: {BOUNDARY_PATH}"
        )

    boundary = gpd.read_file(
        BOUNDARY_PATH
    )[
        [
            "geometry",
        ]
    ].copy()

    if boundary.empty:
        raise ValueError(
            "Ankara boundary is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "Ankara boundary CRS is missing."
        )

    if boundary.geometry.isna().any():
        raise ValueError(
            "Missing Ankara boundary geometry was found."
        )

    if (
        ~boundary.geometry.is_valid
    ).any():
        raise ValueError(
            "Invalid Ankara boundary geometry was found."
        )

    return boundary.to_crs(
        target_crs
    )


def derive_target_grid(
    grid: gpd.GeoDataFrame,
) -> tuple[
    rasterio.Affine,
    int,
    int,
]:
    """Create a 500-m raster lattice aligned with the vector grid."""

    bounds = grid.geometry.bounds

    left = float(
        bounds["minx"].min()
    )

    bottom = float(
        bounds["miny"].min()
    )

    right = float(
        bounds["maxx"].max()
    )

    top = float(
        bounds["maxy"].max()
    )

    width_float = (
        right - left
    ) / GRID_CELL_SIZE_M

    height_float = (
        top - bottom
    ) / GRID_CELL_SIZE_M

    width = int(
        round(
            width_float
        )
    )

    height = int(
        round(
            height_float
        )
    )

    if not np.isclose(
        width_float,
        width,
        atol=1e-6,
    ):
        raise ValueError(
            "Grid horizontal extent is not aligned "
            "to 500-m cells."
        )

    if not np.isclose(
        height_float,
        height,
        atol=1e-6,
    ):
        raise ValueError(
            "Grid vertical extent is not aligned "
            "to 500-m cells."
        )

    transform = from_origin(
        left,
        top,
        GRID_CELL_SIZE_M,
        GRID_CELL_SIZE_M,
    )

    return (
        transform,
        height,
        width,
    )


def create_grid_cell_indices(
    grid: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    *,
    height: int,
    width: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Map every vector grid cell to one target raster row and column."""

    points = (
        grid.geometry
        .representative_point()
    )

    rows, columns = rowcol(
        transform,
        points.x.to_numpy(
            dtype=float
        ),
        points.y.to_numpy(
            dtype=float
        ),
    )

    rows = np.asarray(
        rows,
        dtype=int,
    )

    columns = np.asarray(
        columns,
        dtype=int,
    )

    if (
        (rows < 0).any()
        or (rows >= height).any()
        or (columns < 0).any()
        or (columns >= width).any()
    ):
        raise ValueError(
            "At least one grid cell maps outside "
            "the target raster lattice."
        )

    index_pairs = np.column_stack(
        [
            rows,
            columns,
        ]
    )

    if len(
        np.unique(
            index_pairs,
            axis=0,
        )
    ) != len(
        grid
    ):
        raise ValueError(
            "Multiple vector grid cells map to "
            "the same raster lattice position."
        )

    return (
        rows,
        columns,
    )


def create_study_grid_mask(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Mark target-raster positions that correspond to Ankara grid cells."""

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    mask[
        rows,
        columns,
    ] = True

    return mask


def _intersect_window(
    window: Window,
    *,
    raster_width: int,
    raster_height: int,
) -> Window:
    """Clip a floating Rasterio window to the available source raster."""

    full_window = Window(
        col_off=0,
        row_off=0,
        width=raster_width,
        height=raster_height,
    )

    return (
        window
        .round_offsets()
        .round_lengths()
        .intersection(
            full_window
        )
    )


def warp_worldpop_to_grid(
    grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    target_transform: rasterio.Affine,
    *,
    target_height: int,
    target_width: int,
) -> tuple[
    np.ndarray,
    dict[str, float | int | str],
]:
    """
    Aggregate WorldPop counts onto the 500-m Ankara lattice.

    Source pixels are first masked to the Ankara administrative boundary.
    Population values are extensive quantities, so GDAL's sum resampling is
    used instead of bilinear interpolation when reprojecting.
    """

    if not WORLDPOP_PATH.exists():
        raise FileNotFoundError(
            f"WorldPop raster not found: {WORLDPOP_PATH}"
        )

    grid_bounds = grid.total_bounds

    with rasterio.open(
        WORLDPOP_PATH
    ) as source_dataset:
        if source_dataset.crs is None:
            raise ValueError(
                "WorldPop raster CRS is missing."
            )

        if (
            source_dataset.crs.to_string().upper()
            != EXPECTED_RASTER_CRS
        ):
            raise ValueError(
                "Unexpected WorldPop raster CRS: "
                f"{source_dataset.crs}"
            )

        if source_dataset.count != 1:
            raise ValueError(
                "Expected a single-band WorldPop raster."
            )

        source_bounds = transform_bounds(
            grid.crs,
            source_dataset.crs,
            float(
                grid_bounds[0]
            ),
            float(
                grid_bounds[1]
            ),
            float(
                grid_bounds[2]
            ),
            float(
                grid_bounds[3]
            ),
            densify_pts=21,
        )

        pad_x = abs(
            source_dataset.res[0]
        ) * 2

        pad_y = abs(
            source_dataset.res[1]
        ) * 2

        source_window = from_bounds(
            source_bounds[0] - pad_x,
            source_bounds[1] - pad_y,
            source_bounds[2] + pad_x,
            source_bounds[3] + pad_y,
            transform=source_dataset.transform,
        )

        source_window = _intersect_window(
            source_window,
            raster_width=source_dataset.width,
            raster_height=source_dataset.height,
        )

        source_array = source_dataset.read(
            1,
            window=source_window,
            masked=False,
        ).astype(
            np.float64
        )

        source_transform = (
            source_dataset.window_transform(
                source_window
            )
        )

        nodata = source_dataset.nodata

        valid = np.isfinite(
            source_array
        )

        if nodata is not None:
            valid &= (
                source_array
                != float(nodata)
            )

        valid &= (
            source_array >= 0
        )

        source_mass = np.where(
            valid,
            source_array,
            0.0,
        )

        boundary_in_source_crs = (
            boundary.to_crs(
                source_dataset.crs
            )
        )

        boundary_geometry = (
            boundary_in_source_crs
            .geometry
            .union_all()
        )

        inside_boundary = geometry_mask(
            [
                mapping(
                    boundary_geometry
                )
            ],
            out_shape=source_mass.shape,
            transform=source_transform,
            invert=True,
            all_touched=False,
        )

        source_mass = np.where(
            inside_boundary,
            source_mass,
            0.0,
        )

        source_boundary_population = float(
            source_mass.sum(
                dtype=np.float64
            )
        )

        destination = np.zeros(
            (
                target_height,
                target_width,
            ),
            dtype=np.float64,
        )

        reproject(
            source=source_mass,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_dataset.crs,
            src_nodata=None,
            dst_transform=target_transform,
            dst_crs=grid.crs,
            dst_nodata=0.0,
            resampling=Resampling.sum,
            init_dest_nodata=True,
            num_threads=2,
        )

        metadata: dict[
            str,
            float | int | str,
        ] = {
            "source_crs": (
                source_dataset.crs.to_string()
            ),
            "source_width": (
                source_dataset.width
            ),
            "source_height": (
                source_dataset.height
            ),
            "source_resolution_x": float(
                source_dataset.res[0]
            ),
            "source_resolution_y": float(
                source_dataset.res[1]
            ),
            "source_nodata": (
                float(nodata)
                if nodata is not None
                else "None"
            ),
            "source_window_rows": int(
                source_mass.shape[0]
            ),
            "source_window_columns": int(
                source_mass.shape[1]
            ),
            "source_boundary_population": (
                source_boundary_population
            ),
        }

    if not np.isfinite(
        destination
    ).all():
        raise ValueError(
            "Warped population raster contains "
            "non-finite values."
        )

    minimum_population = float(
        destination.min()
    )

    if minimum_population < -1e-8:
        raise ValueError(
            "Warped population raster contains "
            "negative population values."
        )

    destination[
        destination < 0
    ] = 0.0

    return (
        destination,
        metadata,
    )


def create_circular_offsets(
    radius_m: float,
    *,
    cell_size_m: float = GRID_CELL_SIZE_M,
) -> tuple[
    tuple[int, int],
    ...,
]:
    """Return row/column offsets whose cell centers fall inside a radius."""

    if radius_m < 0:
        raise ValueError(
            "Neighborhood radius cannot be negative."
        )

    if cell_size_m <= 0:
        raise ValueError(
            "Cell size must be positive."
        )

    maximum_offset = int(
        np.floor(
            radius_m
            / cell_size_m
        )
    )

    offsets: list[
        tuple[int, int]
    ] = []

    tolerance = 1e-9

    for row_offset in range(
        -maximum_offset,
        maximum_offset + 1,
    ):
        for column_offset in range(
            -maximum_offset,
            maximum_offset + 1,
        ):
            distance = (
                np.hypot(
                    row_offset,
                    column_offset,
                )
                * cell_size_m
            )

            if distance <= (
                radius_m
                + tolerance
            ):
                offsets.append(
                    (
                        row_offset,
                        column_offset,
                    )
                )

    return tuple(
        offsets
    )


def shift_and_add(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    row_offset: int,
    column_offset: int,
) -> None:
    """Add one integer-cell shift of source into destination."""

    height, width = source.shape

    if row_offset >= 0:
        source_row_start = 0
        source_row_end = (
            height - row_offset
        )
        destination_row_start = (
            row_offset
        )
        destination_row_end = height
    else:
        source_row_start = (
            -row_offset
        )
        source_row_end = height
        destination_row_start = 0
        destination_row_end = (
            height + row_offset
        )

    if column_offset >= 0:
        source_column_start = 0
        source_column_end = (
            width - column_offset
        )
        destination_column_start = (
            column_offset
        )
        destination_column_end = width
    else:
        source_column_start = (
            -column_offset
        )
        source_column_end = width
        destination_column_start = 0
        destination_column_end = (
            width + column_offset
        )

    if (
        source_row_start
        >= source_row_end
        or source_column_start
        >= source_column_end
    ):
        return

    destination[
        destination_row_start:
        destination_row_end,
        destination_column_start:
        destination_column_end,
    ] += source[
        source_row_start:
        source_row_end,
        source_column_start:
        source_column_end,
    ]


def calculate_neighborhood_sum(
    population_surface: np.ndarray,
    *,
    radius_m: float,
    cell_size_m: float = GRID_CELL_SIZE_M,
) -> np.ndarray:
    """
    Sum population in cells whose centers are within a Euclidean radius.

    The 500-m grid itself defines the neighborhood lattice. This avoids
    introducing another interpolation step after population aggregation.
    """

    surface = np.asarray(
        population_surface,
        dtype=np.float64,
    )

    if surface.ndim != 2:
        raise ValueError(
            "Population surface must be two-dimensional."
        )

    if not np.isfinite(
        surface
    ).all():
        raise ValueError(
            "Population surface contains non-finite values."
        )

    if (
        surface < 0
    ).any():
        raise ValueError(
            "Population surface contains negative values."
        )

    result = np.zeros_like(
        surface,
        dtype=np.float64,
    )

    for row_offset, column_offset in (
        create_circular_offsets(
            radius_m,
            cell_size_m=cell_size_m,
        )
    ):
        shift_and_add(
            surface,
            result,
            row_offset=row_offset,
            column_offset=column_offset,
        )

    return result


def build_population_features(
    grid: gpd.GeoDataFrame,
    population_surface: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    study_grid_mask: np.ndarray,
) -> gpd.GeoDataFrame:
    """Attach local and neighborhood population features to Ankara cells."""

    if population_surface.shape != (
        study_grid_mask.shape
    ):
        raise ValueError(
            "Population surface and study mask "
            "shapes do not match."
        )

    study_population_surface = (
        np.where(
            study_grid_mask,
            population_surface,
            0.0,
        )
    )

    population_count = (
        study_population_surface[
            rows,
            columns,
        ]
    )

    population_density = (
        population_count
        / GRID_CELL_AREA_KM2
    )

    neighborhood_values: dict[
        int,
        np.ndarray,
    ] = {}

    for radius_m in NEIGHBORHOOD_RADII_M:
        neighborhood_surface = (
            calculate_neighborhood_sum(
                study_population_surface,
                radius_m=radius_m,
            )
        )

        neighborhood_values[
            radius_m
        ] = neighborhood_surface[
            rows,
            columns,
        ]

    result = grid[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    result[
        "population_count"
    ] = population_count

    result[
        "population_density_per_km2"
    ] = population_density

    result[
        "population_within_1000m"
    ] = neighborhood_values[
        1_000
    ]

    result[
        "population_within_2000m"
    ] = neighborhood_values[
        2_000
    ]

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=grid.crs,
    )


def validate_population_features(
    result: gpd.GeoDataFrame,
    *,
    expected_rows: int,
) -> None:
    """Validate generated population features before writing outputs."""

    if len(
        result
    ) != expected_rows:
        raise ValueError(
            "Population feature row count does not "
            "match the Ankara grid."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in population features."
        )

    feature_columns = [
        "population_count",
        "population_density_per_km2",
        "population_within_1000m",
        "population_within_2000m",
    ]

    for column in feature_columns:
        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (
            values < -1e-8
        ).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    expected_density = (
        result[
            "population_count"
        ].to_numpy(
            dtype=float
        )
        / GRID_CELL_AREA_KM2
    )

    if not np.allclose(
        result[
            "population_density_per_km2"
        ].to_numpy(
            dtype=float
        ),
        expected_density,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError(
            "Population density is not consistent "
            "with the fixed 0.25-km2 cell area."
        )

    local_population = result[
        "population_count"
    ].to_numpy(
        dtype=float
    )

    within_1km = result[
        "population_within_1000m"
    ].to_numpy(
        dtype=float
    )

    within_2km = result[
        "population_within_2000m"
    ].to_numpy(
        dtype=float
    )

    if (
        within_1km
        + 1e-8
        < local_population
    ).any():
        raise ValueError(
            "1-km population cannot be below local population."
        )

    if (
        within_2km
        + 1e-8
        < within_1km
    ).any():
        raise ValueError(
            "2-km population cannot be below 1-km population."
        )


def save_outputs(
    result: gpd.GeoDataFrame,
) -> None:
    """Save spatial and tabular population feature outputs."""

    OUTPUT_GPKG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = (
        result.sort_values(
            "grid_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    result.to_file(
        OUTPUT_GPKG_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    pd.DataFrame(
        result.drop(
            columns="geometry"
        )
    ).to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    result: gpd.GeoDataFrame,
    metadata: dict[str, float | int | str],
    *,
    target_population_total: float,
) -> None:
    """Document source, methodology, diagnostics, and feature distributions."""

    local = result[
        "population_count"
    ]

    density = result[
        "population_density_per_km2"
    ]

    within_1km = result[
        "population_within_1000m"
    ]

    within_2km = result[
        "population_within_2000m"
    ]

    source_boundary_population = float(
        metadata[
            "source_boundary_population"
        ]
    )

    mass_difference = (
        target_population_total
        - source_boundary_population
    )

    mass_relative_difference = (
        mass_difference
        / source_boundary_population
        if source_boundary_population > 0
        else float("nan")
    )

    summary = f"""# Ankara Population Features

## Source

- Dataset: WorldPop 2025 constrained population, R2024B
- Country raster: Turkey
- Raster file: `{WORLDPOP_PATH.name}`
- Source CRS: {metadata["source_crs"]}
- Source raster size: {int(metadata["source_width"]):,} x {int(metadata["source_height"]):,}
- Source resolution: {metadata["source_resolution_x"]} x {metadata["source_resolution_y"]} degrees
- Source NoData: {metadata["source_nodata"]}
- Ankara source window: {int(metadata["source_window_rows"]):,} x {int(metadata["source_window_columns"]):,} pixels

WorldPop values are treated as population counts per raster cell.

## Method

1. Load the existing Ankara 500-m grid in `{EXPECTED_GRID_CRS}`.
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

- Grid rows: {len(result):,}
- Cells with population greater than zero: {int((local > 0).sum()):,}
- Cells with zero population: {int((local <= 0).sum()):,}

## Population-Mass Diagnostic

- WorldPop population inside the Ankara boundary before reprojection: {source_boundary_population:,.2f}
- Population represented by Ankara 500-m grid cells after reprojection: {target_population_total:,.2f}
- Difference after reprojection/grid selection: {mass_difference:+,.2f}
- Relative difference: {mass_relative_difference:+.6%}

This diagnostic checks numerical population-mass preservation inside this
pipeline. It is not a validation against official TÜİK population statistics.

## Feature Summary

### Local 500-m Cell Population

- Total: {local.sum():,.2f}
- Mean: {local.mean():,.2f}
- Median: {local.median():,.2f}
- Maximum: {local.max():,.2f}

### Population Density

- Mean people/km2: {density.mean():,.2f}
- Median people/km2: {density.median():,.2f}
- Maximum people/km2: {density.max():,.2f}

Because every analysis cell is exactly 0.25 km2,
`population_density_per_km2 = population_count / 0.25`.
The two variables are deterministic scale transforms and should not both be
used as predictors in the same ML model.

### Population Within 1 km

- Mean: {within_1km.mean():,.2f}
- Median: {within_1km.median():,.2f}
- Maximum: {within_1km.max():,.2f}

### Population Within 2 km

- Mean: {within_2km.mean():,.2f}
- Median: {within_2km.median():,.2f}
- Maximum: {within_2km.max():,.2f}

## ML Interpretation Policy

These variables represent modeled residential population demand context.

For future ML experiments, use either `population_count` or
`population_density_per_km2`, not both. The neighborhood totals can then test
whether surrounding demand adds information beyond the local 500-m cell.

Population features should first be audited descriptively and then evaluated
incrementally under the existing 5-km spatial cross-validation design.

## Outputs

- `data/processed/{OUTPUT_GPKG_PATH.name}`
- `data/processed/{OUTPUT_CSV_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    result: gpd.GeoDataFrame,
    metadata: dict[str, float | int | str],
    *,
    target_population_total: float,
) -> None:
    """Print key population feature diagnostics."""

    print("-" * 70)

    print(
        "Grid rows:",
        f"{len(result):,}",
    )

    print(
        "Cells with population:",
        f"{int((result['population_count'] > 0).sum()):,}",
    )

    print()

    print(
        "WorldPop boundary total before reprojection:",
        f"{float(metadata['source_boundary_population']):,.2f}",
    )

    print(
        "500-m grid population total:",
        f"{target_population_total:,.2f}",
    )

    print()

    columns = [
        "population_count",
        "population_density_per_km2",
        "population_within_1000m",
        "population_within_2000m",
    ]

    summary = result[
        columns
    ].describe().T[
        [
            "mean",
            "50%",
            "max",
        ]
    ]

    print(
        summary.to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )


def main() -> None:
    """Create WorldPop-derived Ankara population features."""

    print("=" * 70)
    print(
        "VoltSight - Ankara Population Features"
    )
    print("=" * 70)

    grid = load_grid()

    boundary = load_boundary(
        grid.crs
    )

    (
        target_transform,
        target_height,
        target_width,
    ) = derive_target_grid(
        grid
    )

    rows, columns = (
        create_grid_cell_indices(
            grid,
            target_transform,
            height=target_height,
            width=target_width,
        )
    )

    study_grid_mask = (
        create_study_grid_mask(
            rows,
            columns,
            height=target_height,
            width=target_width,
        )
    )

    (
        population_surface,
        metadata,
    ) = warp_worldpop_to_grid(
        grid,
        boundary,
        target_transform,
        target_height=target_height,
        target_width=target_width,
    )

    result = build_population_features(
        grid,
        population_surface,
        rows,
        columns,
        study_grid_mask,
    )

    validate_population_features(
        result,
        expected_rows=len(grid),
    )

    target_population_total = float(
        result[
            "population_count"
        ].sum()
    )

    save_outputs(
        result
    )

    create_summary(
        result,
        metadata,
        target_population_total=target_population_total,
    )

    print_results(
        result,
        metadata,
        target_population_total=target_population_total,
    )

    print("=" * 70)
    print(
        "Ankara population features completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
