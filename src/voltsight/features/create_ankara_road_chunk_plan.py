from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from voltsight.core.study_areas import (  # noqa: E402
    SUPPORTED_GRID_SIZES_METERS,
    get_study_area,
)


ANKARA_CONFIG = get_study_area(
    "ankara"
)

DEFAULT_GRID_SIZE_METERS = 500
DEFAULT_CHUNK_SIZE_METERS = 20_000
DEFAULT_DOWNLOAD_BUFFER_METERS = 1_000

CHUNK_LAYER_NAME = "road_download_chunks"


@dataclass(
    frozen=True,
    slots=True,
)
class RoadChunkPlanPaths:
    """Resolved Ankara road chunk-plan paths."""

    boundary_geojson: Path
    grid_gpkg: Path
    grid_layer_name: str
    chunk_gpkg: Path
    preview_png: Path
    summary_md: Path
    chunk_cache_directory: Path


def distance_token(
    distance_meters: int,
) -> str:
    """Return a deterministic filename token."""

    if distance_meters % 1_000 == 0:
        return (
            f"{distance_meters // 1_000}km"
        )

    return f"{distance_meters}m"


def resolve_paths(
    grid_size_m: int,
    chunk_size_m: int,
) -> RoadChunkPlanPaths:
    """Resolve Ankara road chunk-plan paths."""

    chunk_token = distance_token(
        chunk_size_m
    )

    stem = (
        "ankara_road_download_chunks_"
        f"{chunk_token}"
    )

    return RoadChunkPlanPaths(
        boundary_geojson=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "ankara_boundary_osm.geojson"
        ),
        grid_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"ankara_grid_{grid_size_m}m.gpkg"
        ),
        grid_layer_name=(
            f"ankara_grid_{grid_size_m}m"
        ),
        chunk_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / f"{stem}.gpkg"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / f"{stem}_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / f"{stem}_summary.md"
        ),
        chunk_cache_directory=(
            PROJECT_ROOT
            / "cache"
            / "ankara"
            / "road_chunks"
            / chunk_token
        ),
    )


def create_output_directories(
    paths: RoadChunkPlanPaths,
) -> None:
    """Create required output directories."""

    directories = {
        paths.chunk_gpkg.parent,
        paths.preview_png.parent,
        paths.summary_md.parent,
        paths.chunk_cache_directory,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_positive_integer(
    value: int,
    name: str,
) -> None:
    """Validate a positive integer argument."""

    if value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )


def validate_parameters(
    grid_size_m: int,
    chunk_size_m: int,
    download_buffer_m: int,
) -> None:
    """Validate chunk-plan parameters."""

    validate_positive_integer(
        grid_size_m,
        "Grid size",
    )

    validate_positive_integer(
        chunk_size_m,
        "Chunk size",
    )

    if grid_size_m not in (
        SUPPORTED_GRID_SIZES_METERS
    ):
        raise ValueError(
            "Unsupported grid size: "
            f"{grid_size_m}"
        )

    if download_buffer_m < 0:
        raise ValueError(
            "Download buffer cannot be negative."
        )

    if download_buffer_m >= chunk_size_m:
        raise ValueError(
            "Download buffer must be smaller "
            "than the chunk size."
        )


def validate_input_files(
    paths: RoadChunkPlanPaths,
) -> None:
    """Ensure that the Ankara boundary and grid exist."""

    required_paths = (
        paths.boundary_geojson,
        paths.grid_gpkg,
    )

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if not missing:
        return

    missing_text = "\n".join(
        f"- {path}"
        for path in missing
    )

    raise FileNotFoundError(
        "Required Ankara study files are missing.\n"
        "Run create_study_grid.py first.\n"
        f"{missing_text}"
    )


def load_study_data(
    paths: RoadChunkPlanPaths,
    expected_grid_size_m: int,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load and validate the Ankara boundary and grid."""

    boundary = gpd.read_file(
        paths.boundary_geojson
    )

    grid = gpd.read_file(
        paths.grid_gpkg,
        layer=paths.grid_layer_name,
    )

    if boundary.empty:
        raise ValueError(
            "The Ankara boundary is empty."
        )

    if grid.empty:
        raise ValueError(
            "The Ankara study grid is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "The Ankara boundary has no CRS."
        )

    if grid.crs is None:
        raise ValueError(
            "The Ankara grid has no CRS."
        )

    if not grid.crs.is_projected:
        raise ValueError(
            "The Ankara grid must use a "
            "projected meter-based CRS."
        )

    required_grid_columns = {
        "grid_id",
        "grid_size_m",
        "cell_area_m2",
        "geometry",
    }

    missing_columns = (
        required_grid_columns
        - set(grid.columns)
    )

    if missing_columns:
        raise ValueError(
            "The Ankara grid is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "The Ankara grid contains duplicate IDs."
        )

    if not grid[
        "grid_id"
    ].astype(str).str.startswith(
        "ANK_"
    ).all():
        raise ValueError(
            "The Ankara grid contains invalid ID prefixes."
        )

    actual_grid_sizes = set(
        pd.to_numeric(
            grid["grid_size_m"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    if actual_grid_sizes != {
        expected_grid_size_m
    }:
        raise ValueError(
            "Unexpected grid-size values. "
            f"Expected {expected_grid_size_m}, "
            f"found {sorted(actual_grid_sizes)}."
        )

    if grid.geometry.isna().any():
        raise ValueError(
            "The Ankara grid contains missing geometry."
        )

    if not grid.geometry.is_valid.all():
        raise ValueError(
            "The Ankara grid contains invalid geometry."
        )

    boundary = boundary.to_crs(
        grid.crs
    )

    boundary_geometry = (
        boundary.geometry.union_all()
    )

    boundary_area_km2 = (
        boundary_geometry.area
        / 1_000_000.0
    )

    if not (
        ANKARA_CONFIG.minimum_expected_area_km2
        <= boundary_area_km2
        <= ANKARA_CONFIG.maximum_expected_area_km2
    ):
        raise ValueError(
            "Unexpected Ankara boundary area: "
            f"{boundary_area_km2:,.2f} km²."
        )

    print(
        f"Loaded Ankara grid cells: {len(grid):,}"
    )

    print(
        "Validated Ankara boundary area: "
        f"{boundary_area_km2:,.2f} km²"
    )

    print(
        f"Analysis CRS: {grid.crs}"
    )

    return boundary, grid


def create_core_chunks(
    boundary_projected: gpd.GeoDataFrame,
    chunk_size_m: int,
) -> gpd.GeoDataFrame:
    """
    Create non-overlapping square chunks clipped to Ankara.

    The stored geometries represent non-overlapping core areas.
    A download buffer will be applied later when each road chunk is
    downloaded.
    """

    boundary_geometry = (
        boundary_projected.geometry.union_all()
    )

    if boundary_geometry.is_empty:
        raise ValueError(
            "The Ankara boundary union is empty."
        )

    min_x, min_y, max_x, max_y = (
        boundary_geometry.bounds
    )

    start_x = int(
        math.floor(
            min_x / chunk_size_m
        )
        * chunk_size_m
    )

    start_y = int(
        math.floor(
            min_y / chunk_size_m
        )
        * chunk_size_m
    )

    end_x = int(
        math.ceil(
            max_x / chunk_size_m
        )
        * chunk_size_m
    )

    end_y = int(
        math.ceil(
            max_y / chunk_size_m
        )
        * chunk_size_m
    )

    records: list[dict[str, object]] = []

    for x_coordinate in range(
        start_x,
        end_x,
        chunk_size_m,
    ):
        for y_coordinate in range(
            start_y,
            end_y,
            chunk_size_m,
        ):
            tile = box(
                x_coordinate,
                y_coordinate,
                x_coordinate + chunk_size_m,
                y_coordinate + chunk_size_m,
            )

            clipped = tile.intersection(
                boundary_geometry
            )

            if clipped.is_empty:
                continue

            if clipped.area <= 0:
                continue

            records.append(
                {
                    "tile_min_x": x_coordinate,
                    "tile_min_y": y_coordinate,
                    "tile_max_x": (
                        x_coordinate
                        + chunk_size_m
                    ),
                    "tile_max_y": (
                        y_coordinate
                        + chunk_size_m
                    ),
                    "core_area_km2": (
                        clipped.area
                        / 1_000_000.0
                    ),
                    "geometry": clipped,
                }
            )

    if not records:
        raise RuntimeError(
            "No Ankara road chunks were generated."
        )

    chunks = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=boundary_projected.crs,
    )

    chunks = chunks.sort_values(
        by=[
            "tile_min_x",
            "tile_min_y",
        ]
    ).reset_index(
        drop=True
    )

    id_width = max(
        4,
        len(str(len(chunks))),
    )

    chunks.insert(
        0,
        "chunk_id",
        [
            (
                "ANK_ROAD_"
                f"{index:0{id_width}d}"
            )
            for index in range(
                1,
                len(chunks) + 1,
            )
        ],
    )

    chunks.insert(
        1,
        "chunk_order",
        range(
            1,
            len(chunks) + 1,
        ),
    )

    chunks["chunk_size_m"] = (
        chunk_size_m
    )

    chunks["core_area_km2"] = (
        chunks["core_area_km2"]
        .round(4)
    )

    return chunks


def assign_grid_cells_to_chunks(
    chunks: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign every Ankara grid centroid to one core chunk."""

    centers = grid[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    centers["geometry"] = (
        centers.geometry.centroid
    )

    matches = gpd.sjoin(
        centers,
        chunks[
            [
                "chunk_id",
                "chunk_order",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    matches = matches.sort_values(
        by=[
            "grid_id",
            "chunk_order",
        ],
        na_position="last",
    )

    matches = matches.drop_duplicates(
        subset="grid_id",
        keep="first",
    )

    missing_count = int(
        matches["chunk_id"].isna().sum()
    )

    if missing_count:
        raise RuntimeError(
            "Some grid cells could not be assigned "
            "to a road chunk. "
            f"Missing count: {missing_count:,}"
        )

    counts = (
        matches.groupby(
            "chunk_id",
            as_index=False,
        )
        .agg(
            grid_cell_count=(
                "grid_id",
                "nunique",
            )
        )
    )

    result = chunks.merge(
        counts,
        on="chunk_id",
        how="left",
        validate="one_to_one",
    )

    result["grid_cell_count"] = (
        result["grid_cell_count"]
        .fillna(0)
        .astype(int)
    )

    assigned_total = int(
        result["grid_cell_count"].sum()
    )

    if assigned_total != len(grid):
        raise RuntimeError(
            "Chunk grid-cell counts do not match "
            "the Ankara grid. "
            f"Expected {len(grid):,}, "
            f"found {assigned_total:,}."
        )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=chunks.crs,
    )


def add_download_area_estimates(
    chunks: gpd.GeoDataFrame,
    download_buffer_m: int,
) -> gpd.GeoDataFrame:
    """Estimate each buffered road-download area."""

    result = chunks.copy()

    buffered_areas = (
        result.geometry.buffer(
            download_buffer_m
        ).area
        / 1_000_000.0
    )

    result[
        "download_buffer_m"
    ] = download_buffer_m

    result[
        "estimated_download_area_km2"
    ] = buffered_areas.round(4)

    return result


def validate_chunks(
    chunks: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> None:
    """Validate chunk IDs, geometries and workload counts."""

    if chunks.empty:
        raise ValueError(
            "The road chunk plan is empty."
        )

    if chunks["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate road chunk IDs were found."
        )

    if chunks.geometry.isna().any():
        raise ValueError(
            "Road chunks contain missing geometry."
        )

    if not chunks.geometry.is_valid.all():
        raise ValueError(
            "Road chunks contain invalid geometry."
        )

    if (
        chunks["grid_cell_count"] < 0
    ).any():
        raise ValueError(
            "Road chunks contain negative grid counts."
        )

    if int(
        chunks["grid_cell_count"].sum()
    ) != len(grid):
        raise ValueError(
            "Road chunk workload totals are incorrect."
        )

    print(
        "Ankara road chunk validation "
        "completed successfully."
    )


def save_chunks(
    chunks: gpd.GeoDataFrame,
    paths: RoadChunkPlanPaths,
) -> None:
    """Save the road chunk plan as a GeoPackage."""

    if paths.chunk_gpkg.exists():
        paths.chunk_gpkg.unlink()

    chunks.to_file(
        paths.chunk_gpkg,
        layer=CHUNK_LAYER_NAME,
        driver="GPKG",
    )

    print(
        "Road chunk plan saved: "
        f"{paths.chunk_gpkg}"
    )


def create_preview(
    boundary: gpd.GeoDataFrame,
    chunks: gpd.GeoDataFrame,
    paths: RoadChunkPlanPaths,
) -> None:
    """Create a preview of the Ankara road chunks."""

    figure, axis = plt.subplots(
        figsize=(11, 11)
    )

    chunks.plot(
        ax=axis,
        column="grid_cell_count",
        legend=True,
        linewidth=0.7,
        edgecolor="black",
        alpha=0.75,
        legend_kwds={
            "label": (
                "Assigned Ankara grid cells"
            ),
            "shrink": 0.65,
        },
    )

    boundary.boundary.plot(
        ax=axis,
        linewidth=1.5,
    )

    axis.set_title(
        "VoltSight - Ankara Road Download Chunk Plan"
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        paths.preview_png,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Road chunk preview saved: "
        f"{paths.preview_png}"
    )


def project_relative_path(
    path: Path,
) -> str:
    """Return a project-relative or absolute POSIX path."""

    try:
        relative_path = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative_path = path

    return relative_path.as_posix()


def create_summary(
    chunks: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    paths: RoadChunkPlanPaths,
    grid_size_m: int,
    chunk_size_m: int,
    download_buffer_m: int,
) -> None:
    """Create a Markdown summary of the chunk plan."""

    grid_counts = (
        chunks["grid_cell_count"]
    )

    core_areas = (
        chunks["core_area_km2"]
    )

    download_areas = (
        chunks[
            "estimated_download_area_km2"
        ]
    )

    table = (
        chunks.sort_values(
            "chunk_order"
        )[
            [
                "chunk_id",
                "grid_cell_count",
                "core_area_km2",
                "estimated_download_area_km2",
            ]
        ]
        .to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    summary = f"""# Ankara Road Download Chunk Plan

## Inputs

- Ankara grid: `{project_relative_path(paths.grid_gpkg)}`
- Grid layer: `{paths.grid_layer_name}`
- Ankara grid cells: {len(grid):,}
- Analysis CRS: {grid.crs}

## Chunk Configuration

- Core chunk size: {chunk_size_m:,} x {chunk_size_m:,} metres
- Road-download buffer: {download_buffer_m:,} metres
- Generated chunks: {len(chunks):,}
- Chunk layer: `{CHUNK_LAYER_NAME}`

## Workload Statistics

- Minimum grid cells per chunk: {int(grid_counts.min()):,}
- Median grid cells per chunk: {grid_counts.median():,.0f}
- Maximum grid cells per chunk: {int(grid_counts.max()):,}
- Mean core area: {core_areas.mean():,.2f} km²
- Maximum core area: {core_areas.max():,.2f} km²
- Maximum estimated buffered download area: {download_areas.max():,.2f} km²

## Generated Outputs

- `{project_relative_path(paths.chunk_gpkg)}`
- `{project_relative_path(paths.preview_png)}`

## Method

The Ankara province boundary was divided into deterministic,
non-overlapping {chunk_size_m:,}-metre square tiles. Each tile was
clipped to the province boundary.

Every {grid_size_m:,}-metre Ankara grid-cell centroid was assigned to
exactly one core road chunk. A {download_buffer_m:,}-metre buffer will
be applied around each chunk during the road-download stage so that
roads close to chunk boundaries are not lost.

Each chunk will be downloaded and cached independently. A failed or
interrupted run can therefore resume without downloading completed
chunks again.

## Chunk Workload

{table}

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    paths.summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "Road chunk summary saved: "
        f"{paths.summary_md}"
    )


def print_statistics(
    chunks: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> None:
    """Print chunk-plan statistics."""

    print("-" * 70)

    print(
        f"Ankara grid cells: {len(grid):,}"
    )

    print(
        f"Road download chunks: {len(chunks):,}"
    )

    print(
        "Minimum grid cells per chunk: "
        f"{int(chunks['grid_cell_count'].min()):,}"
    )

    print(
        "Median grid cells per chunk: "
        f"{chunks['grid_cell_count'].median():,.0f}"
    )

    print(
        "Maximum grid cells per chunk: "
        f"{int(chunks['grid_cell_count'].max()):,}"
    )

    print(
        "Maximum estimated download area: "
        f"{chunks['estimated_download_area_km2'].max():,.2f} km²"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Ankara road chunk-plan CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Create resumable Ankara road-download chunks."
        )
    )

    parser.add_argument(
        "--grid-size-m",
        type=int,
        choices=SUPPORTED_GRID_SIZES_METERS,
        default=DEFAULT_GRID_SIZE_METERS,
    )

    parser.add_argument(
        "--chunk-size-m",
        type=int,
        default=DEFAULT_CHUNK_SIZE_METERS,
    )

    parser.add_argument(
        "--download-buffer-m",
        type=int,
        default=DEFAULT_DOWNLOAD_BUFFER_METERS,
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    try:
        validate_parameters(
            grid_size_m=arguments.grid_size_m,
            chunk_size_m=arguments.chunk_size_m,
            download_buffer_m=(
                arguments.download_buffer_m
            ),
        )
    except ValueError as error:
        parser.error(
            str(error)
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Create the Ankara road chunk plan."""

    paths = resolve_paths(
        grid_size_m=arguments.grid_size_m,
        chunk_size_m=arguments.chunk_size_m,
    )

    print("=" * 70)

    print(
        "VoltSight - Ankara Road Download Chunk Plan"
    )

    print("=" * 70)

    print(
        f"Grid size: {arguments.grid_size_m:,} m"
    )

    print(
        f"Chunk size: {arguments.chunk_size_m:,} m"
    )

    print(
        "Download buffer: "
        f"{arguments.download_buffer_m:,} m"
    )

    create_output_directories(
        paths
    )

    validate_input_files(
        paths
    )

    boundary, grid = load_study_data(
        paths,
        expected_grid_size_m=(
            arguments.grid_size_m
        ),
    )

    chunks = create_core_chunks(
        boundary_projected=boundary,
        chunk_size_m=(
            arguments.chunk_size_m
        ),
    )

    chunks = assign_grid_cells_to_chunks(
        chunks,
        grid,
    )

    chunks = add_download_area_estimates(
        chunks,
        download_buffer_m=(
            arguments.download_buffer_m
        ),
    )

    validate_chunks(
        chunks,
        grid,
    )

    save_chunks(
        chunks,
        paths,
    )

    create_preview(
        boundary,
        chunks,
        paths,
    )

    create_summary(
        chunks=chunks,
        grid=grid,
        paths=paths,
        grid_size_m=arguments.grid_size_m,
        chunk_size_m=arguments.chunk_size_m,
        download_buffer_m=(
            arguments.download_buffer_m
        ),
    )

    print_statistics(
        chunks,
        grid,
    )

    print("=" * 70)

    print(
        "Ankara road chunk plan "
        "completed successfully."
    )

    print("=" * 70)

    return chunks


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the Ankara road chunk-plan pipeline."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
