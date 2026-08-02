from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from voltsight.features.create_road_features import (  # noqa: E402
    aggregate_road_features,
    intersect_roads_with_grid,
    validate_features,
)


DEFAULT_GRID_SIZE_METERS = 500
DEFAULT_BATCH_SIZE = 5_000

GRID_FEATURE_LAYER_NAME = "grid_road_features"
GRID_LAYER_NAME_TEMPLATE = "ankara_grid_{grid_size_m}m"
ROADS_LAYER_NAME = "drive_roads"


ROAD_METRIC_COLUMNS = [
    "grid_id",
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
]


@dataclass(
    frozen=True,
    slots=True,
)
class AnkaraRoadFeaturePaths:
    """Paths used by the Ankara road-feature pipeline."""

    grid_gpkg: Path
    grid_layer_name: str
    roads_gpkg: Path
    batch_directory: Path
    output_gpkg: Path
    output_geojson: Path
    output_csv: Path
    preview_png: Path
    summary_md: Path


def resolve_paths(
    grid_size_m: int = DEFAULT_GRID_SIZE_METERS,
) -> AnkaraRoadFeaturePaths:
    """Resolve deterministic Ankara road-feature paths."""

    if grid_size_m <= 0:
        raise ValueError(
            "Grid size must be positive."
        )

    grid_stem = (
        f"ankara_grid_{grid_size_m}m"
    )

    return AnkaraRoadFeaturePaths(
        grid_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"{grid_stem}.gpkg"
        ),
        grid_layer_name=(
            GRID_LAYER_NAME_TEMPLATE.format(
                grid_size_m=grid_size_m
            )
        ),
        roads_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_drive_roads.gpkg"
        ),
        batch_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / (
                "ankara_road_feature_batches_"
                f"{grid_size_m}m"
            )
        ),
        output_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_road_features.gpkg"
        ),
        output_geojson=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_road_features.geojson"
        ),
        output_csv=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_road_features.csv"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / "ankara_road_features_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / "ankara_road_features_summary.md"
        ),
    )


def create_output_directories(
    paths: AnkaraRoadFeaturePaths,
) -> None:
    """Create all required output directories."""

    for directory in {
        paths.batch_directory,
        paths.output_gpkg.parent,
        paths.preview_png.parent,
        paths.summary_md.parent,
    }:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_input_files(
    paths: AnkaraRoadFeaturePaths,
) -> None:
    """Ensure that the Ankara grid and road network exist."""

    required_paths = (
        paths.grid_gpkg,
        paths.roads_gpkg,
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
        "Required Ankara input files are missing.\n"
        f"{missing_text}"
    )


def normalize_main_road_column(
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Normalize the main-road indicator to real Boolean values."""

    result = roads.copy()

    values = result[
        "is_main_road"
    ]

    if pd.api.types.is_bool_dtype(
        values
    ):
        return result

    if pd.api.types.is_numeric_dtype(
        values
    ):
        result["is_main_road"] = (
            pd.to_numeric(
                values,
                errors="coerce",
            )
            .fillna(0)
            .ne(0)
        )

        return result

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    result["is_main_road"] = (
        normalized.isin(
            {
                "1",
                "true",
                "yes",
                "y",
            }
        )
    )

    return result


def load_inputs(
    paths: AnkaraRoadFeaturePaths,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load and validate the Ankara grid and merged road network."""

    grid = gpd.read_file(
        paths.grid_gpkg,
        layer=paths.grid_layer_name,
    )

    roads = gpd.read_file(
        paths.roads_gpkg,
        layer=ROADS_LAYER_NAME,
    )

    if grid.empty:
        raise ValueError(
            "The Ankara grid is empty."
        )

    if roads.empty:
        raise ValueError(
            "The Ankara road network is empty."
        )

    if grid.crs is None:
        raise ValueError(
            "The Ankara grid has no CRS."
        )

    if roads.crs is None:
        raise ValueError(
            "The Ankara road network has no CRS."
        )

    if not grid.crs.is_projected:
        raise ValueError(
            "The Ankara grid must use a projected CRS."
        )

    required_grid_columns = {
        "grid_id",
        "cell_area_m2",
        "geometry",
    }

    missing_grid_columns = (
        required_grid_columns
        - set(grid.columns)
    )

    if missing_grid_columns:
        raise ValueError(
            "The Ankara grid is missing columns: "
            f"{sorted(missing_grid_columns)}"
        )

    required_road_columns = {
        "road_id",
        "highway",
        "is_main_road",
        "geometry",
    }

    missing_road_columns = (
        required_road_columns
        - set(roads.columns)
    )

    if missing_road_columns:
        raise ValueError(
            "The Ankara road network is missing columns: "
            f"{sorted(missing_road_columns)}"
        )

    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate Ankara grid IDs were found."
        )

    if roads["road_id"].duplicated().any():
        raise ValueError(
            "Duplicate Ankara road IDs were found."
        )

    if roads.crs != grid.crs:
        roads = roads.to_crs(
            grid.crs
        )

    roads = normalize_main_road_column(
        roads
    )

    roads = roads.loc[
        roads.geometry.notna()
    ].copy()

    roads = roads.loc[
        ~roads.geometry.is_empty
    ].copy()

    if not roads.geometry.is_valid.all():
        raise ValueError(
            "The Ankara road network contains invalid geometry."
        )

    print(
        f"Loaded Ankara grid cells: {len(grid):,}"
    )

    print(
        f"Loaded Ankara road pieces: {len(roads):,}"
    )

    print(
        "Loaded Ankara main-road pieces: "
        f"{int(roads['is_main_road'].sum()):,}"
    )

    print(
        f"Analysis CRS: {grid.crs}"
    )

    return grid, roads


def select_candidate_roads(
    roads: gpd.GeoDataFrame,
    grid_batch: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Select only roads whose bounds intersect one grid batch."""

    min_x, min_y, max_x, max_y = (
        grid_batch.total_bounds
    )

    batch_bounds = box(
        min_x,
        min_y,
        max_x,
        max_y,
    )

    candidate_positions = (
        roads.sindex.query(
            batch_bounds,
            predicate="intersects",
        )
    )

    candidate_positions = np.unique(
        np.asarray(
            candidate_positions,
            dtype=int,
        )
    )

    if len(candidate_positions) == 0:
        return roads.iloc[
            0:0
        ].copy()

    return roads.iloc[
        candidate_positions
    ].copy()


def add_zero_road_metrics(
    grid_batch: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add zero-valued road metrics to a road-free batch."""

    result = grid_batch.copy()

    result["road_length_m"] = 0.0
    result["road_segment_count"] = 0
    result["main_road_length_m"] = 0.0
    result["main_road_segment_count"] = 0
    result["road_density_km_per_km2"] = 0.0

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=grid_batch.crs,
    )


def calculate_intersection_features(
    grid_batch: gpd.GeoDataFrame,
    candidate_roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Calculate road length and density for one grid batch."""

    if candidate_roads.empty:
        return add_zero_road_metrics(
            grid_batch
        )

    intersections = (
        intersect_roads_with_grid(
            grid_batch,
            candidate_roads,
        )
    )

    if intersections.empty:
        return add_zero_road_metrics(
            grid_batch
        )

    return aggregate_road_features(
        grid_batch,
        intersections,
    )


def add_nearest_main_road_features(
    features: gpd.GeoDataFrame,
    main_roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Calculate nearest-main-road features for one batch."""

    if main_roads.empty:
        raise RuntimeError(
            "The Ankara network contains no main roads."
        )

    grid_centers = features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_centers["geometry"] = (
        grid_centers.geometry.centroid
    )

    nearest_matches = gpd.sjoin_nearest(
        grid_centers,
        main_roads[
            [
                "road_id",
                "highway",
                "geometry",
            ]
        ],
        how="left",
        distance_col=(
            "distance_to_main_road_m"
        ),
    )

    nearest_matches = (
        nearest_matches.sort_values(
            by=[
                "grid_id",
                "distance_to_main_road_m",
                "road_id",
            ],
            na_position="last",
            kind="stable",
        )
        .drop_duplicates(
            subset="grid_id",
            keep="first",
        )
    )

    nearest_features = (
        nearest_matches[
            [
                "grid_id",
                "distance_to_main_road_m",
                "highway",
            ]
        ]
        .rename(
            columns={
                "highway": (
                    "nearest_main_road_type"
                ),
            }
        )
    )

    result = features.merge(
        nearest_features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    result[
        "distance_to_main_road_m"
    ] = pd.to_numeric(
        result[
            "distance_to_main_road_m"
        ],
        errors="coerce",
    ).round(2)

    missing_distance_count = int(
        result[
            "distance_to_main_road_m"
        ].isna().sum()
    )

    if missing_distance_count:
        raise RuntimeError(
            "Some Ankara grid cells could not be matched "
            "to a main road. "
            f"Missing count: {missing_distance_count:,}"
        )

    if result[
        "nearest_main_road_type"
    ].isna().any():
        raise RuntimeError(
            "Some Ankara grid cells have no nearest "
            "main-road classification."
        )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=features.crs,
    )


def compute_batch_features(
    grid_batch: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    main_roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Calculate all road features for one Ankara grid batch."""

    candidate_roads = select_candidate_roads(
        roads,
        grid_batch,
    )

    print(
        "Candidate roads for batch: "
        f"{len(candidate_roads):,}"
    )

    features = (
        calculate_intersection_features(
            grid_batch,
            candidate_roads,
        )
    )

    features = (
        add_nearest_main_road_features(
            features,
            main_roads,
        )
    )

    validate_features(
        features
    )

    return features


def batch_output_path(
    paths: AnkaraRoadFeaturePaths,
    batch_number: int,
) -> Path:
    """Return the CSV checkpoint path for one batch."""

    return (
        paths.batch_directory
        / f"batch_{batch_number:04d}.csv"
    )


def save_batch_metrics(
    features: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save one resumable feature batch without geometry."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".csv.tmp"
    )

    pd.DataFrame(
        features[
            ROAD_METRIC_COLUMNS
        ]
    ).to_csv(
        temporary_path,
        index=False,
        encoding="utf-8",
    )

    temporary_path.replace(
        output_path
    )


def load_cached_batch(
    output_path: Path,
    expected_grid_ids: Sequence[str],
) -> pd.DataFrame:
    """Load and validate one completed feature batch."""

    cached = pd.read_csv(
        output_path
    )

    missing_columns = (
        set(ROAD_METRIC_COLUMNS)
        - set(cached.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Cached batch {output_path.name} "
            "is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if cached["grid_id"].duplicated().any():
        raise ValueError(
            f"Cached batch {output_path.name} "
            "contains duplicate grid IDs."
        )

    expected_ids = [
        str(value)
        for value in expected_grid_ids
    ]

    actual_ids = set(
        cached["grid_id"].astype(str)
    )

    if actual_ids != set(
        expected_ids
    ):
        raise ValueError(
            f"Cached batch {output_path.name} "
            "does not match the expected grid cells."
        )

    cached["grid_id"] = (
        cached["grid_id"].astype(str)
    )

    cached = (
        cached.set_index(
            "grid_id"
        )
        .loc[
            expected_ids
        ]
        .reset_index()
    )

    return cached


def process_batches(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    paths: AnkaraRoadFeaturePaths,
    *,
    batch_size: int,
    force: bool,
) -> pd.DataFrame:
    """Process all Ankara grid batches with resumable checkpoints."""

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    main_roads = roads.loc[
        roads["is_main_road"]
    ].copy()

    if main_roads.empty:
        raise RuntimeError(
            "No Ankara main roads were identified."
        )

    # Build and cache the spatial index before batch processing.
    _ = roads.sindex
    _ = main_roads.sindex

    batch_count = math.ceil(
        len(grid)
        / batch_size
    )

    completed_frames: list[
        pd.DataFrame
    ] = []

    for batch_number in range(
        1,
        batch_count + 1,
    ):
        start_index = (
            batch_number - 1
        ) * batch_size

        end_index = min(
            start_index + batch_size,
            len(grid),
        )

        grid_batch = grid.iloc[
            start_index:end_index
        ].copy()

        expected_ids = (
            grid_batch[
                "grid_id"
            ]
            .astype(str)
            .tolist()
        )

        output_path = batch_output_path(
            paths,
            batch_number,
        )

        print("=" * 70)

        print(
            f"Batch {batch_number:,}/{batch_count:,}"
        )

        print(
            "Grid row range: "
            f"{start_index + 1:,}-{end_index:,}"
        )

        if (
            output_path.exists()
            and not force
        ):
            try:
                cached = load_cached_batch(
                    output_path,
                    expected_ids,
                )

                completed_frames.append(
                    cached
                )

                print(
                    "[SKIP] Valid cached batch loaded: "
                    f"{output_path.name}"
                )

                continue

            except Exception as error:
                print(
                    "Cached batch is invalid and will "
                    f"be recalculated: {error}"
                )

        features = compute_batch_features(
            grid_batch,
            roads,
            main_roads,
        )

        save_batch_metrics(
            features,
            output_path,
        )

        completed_frames.append(
            pd.DataFrame(
                features[
                    ROAD_METRIC_COLUMNS
                ]
            )
        )

        print(
            "[SUCCESS] Batch saved: "
            f"{output_path.name}"
        )

    metrics = pd.concat(
        completed_frames,
        ignore_index=True,
    )

    return metrics


def assemble_features(
    grid: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Merge batch metrics back onto the complete Ankara grid."""

    if metrics["grid_id"].duplicated().any():
        raise ValueError(
            "Batch metrics contain duplicate grid IDs."
        )

    if len(metrics) != len(grid):
        raise ValueError(
            "Batch metric row count does not match "
            "the Ankara grid. "
            f"Expected {len(grid):,}, "
            f"found {len(metrics):,}."
        )

    grid_ids = set(
        grid["grid_id"].astype(str)
    )

    metric_ids = set(
        metrics["grid_id"].astype(str)
    )

    if grid_ids != metric_ids:
        raise ValueError(
            "Batch metric grid IDs do not match "
            "the complete Ankara grid."
        )

    existing_metric_columns = [
        column
        for column in ROAD_METRIC_COLUMNS
        if (
            column != "grid_id"
            and column in grid.columns
        )
    ]

    base_grid = grid.drop(
        columns=existing_metric_columns,
        errors="ignore",
    )

    features = base_grid.merge(
        metrics,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    result = gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=grid.crs,
    )

    validate_features(
        result
    )

    return result


def save_outputs(
    features: gpd.GeoDataFrame,
    paths: AnkaraRoadFeaturePaths,
    *,
    write_geojson: bool,
) -> None:
    """Save GIS and machine-learning outputs."""

    output_paths = [
        paths.output_gpkg,
        paths.output_csv,
    ]

    if write_geojson:
        output_paths.append(
            paths.output_geojson
        )

    for output_path in output_paths:
        if output_path.exists():
            output_path.unlink()

    features.to_file(
        paths.output_gpkg,
        layer=GRID_FEATURE_LAYER_NAME,
        driver="GPKG",
    )

    csv_features = pd.DataFrame(
        features.drop(
            columns="geometry"
        )
    )

    csv_features.to_csv(
        paths.output_csv,
        index=False,
        encoding="utf-8",
    )

    if write_geojson:
        features.to_crs(
            epsg=4326
        ).to_file(
            paths.output_geojson,
            driver="GeoJSON",
        )

    print(
        f"Feature GeoPackage saved: {paths.output_gpkg}"
    )

    print(
        f"Machine-learning CSV saved: {paths.output_csv}"
    )

    if write_geojson:
        print(
            f"Feature GeoJSON saved: {paths.output_geojson}"
        )


def create_preview(
    features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    paths: AnkaraRoadFeaturePaths,
) -> None:
    """Create an Ankara road-density preview."""

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column=(
            "road_density_km_per_km2"
        ),
        legend=True,
        linewidth=0,
        alpha=0.90,
        legend_kwds={
            "label": (
                "Road density (km / km²)"
            ),
            "shrink": 0.65,
        },
    )

    main_roads = roads.loc[
        roads["is_main_road"]
    ]

    main_roads.plot(
        ax=axis,
        linewidth=0.04,
        alpha=0.40,
    )

    axis.set_title(
        "VoltSight - Ankara Grid Road Density"
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
        f"Road feature preview saved: {paths.preview_png}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    paths: AnkaraRoadFeaturePaths,
    *,
    batch_size: int,
) -> None:
    """Create a Markdown Ankara road-feature summary."""

    road_density = features[
        "road_density_km_per_km2"
    ]

    main_road_distance = features[
        "distance_to_main_road_m"
    ]

    zero_road_count = int(
        (
            features["road_length_m"]
            == 0
        ).sum()
    )

    cells_with_roads = (
        len(features)
        - zero_road_count
    )

    total_road_length_km = (
        roads["edge_length_m"].sum()
        / 1_000
    )

    main_road_length_km = (
        roads.loc[
            roads["is_main_road"],
            "edge_length_m",
        ].sum()
        / 1_000
    )

    summary = f"""# Ankara Road Feature Summary

## Inputs

- Ankara grid cells: {len(features):,}
- Ankara road pieces: {len(roads):,}
- Ankara main-road pieces: {int(roads["is_main_road"].sum()):,}
- Total road-network length: {total_road_length_km:,.2f} km
- Total main-road length: {main_road_length_km:,.2f} km
- Analysis CRS: {features.crs}

## Processing

- Grid batch size: {batch_size:,}
- Batch count: {math.ceil(len(features) / batch_size):,}
- Checkpoint directory: `data/interim/{paths.batch_directory.name}`

## Grid Features

- Cells with road data: {cells_with_roads:,}
- Cells without road data: {zero_road_count:,}
- Mean road density: {road_density.mean():,.2f} km/km²
- Median road density: {road_density.median():,.2f} km/km²
- Maximum road density: {road_density.max():,.2f} km/km²
- Mean distance to a main road: {main_road_distance.mean():,.2f} m
- Median distance to a main road: {main_road_distance.median():,.2f} m
- Maximum distance to a main road: {main_road_distance.max():,.2f} m

## Generated Features

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `nearest_main_road_type`

## Generated Outputs

- `data/processed/ankara_grid_road_features.gpkg`
- `data/processed/ankara_grid_road_features.csv`
- `docs/ankara_road_features_summary.md`

## Method

The merged Ankara road network was intersected with the
500 x 500 metre study grid in resumable batches.

Only the road geometry inside each grid cell contributed to road
length and density. Distance to the nearest main road was calculated
from each grid-cell centroid in the projected metre-based coordinate
system.

Batch CSV checkpoints allow an interrupted run to continue without
recalculating completed grid sections.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    paths.summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Road feature summary saved: {paths.summary_md}"
    )


def print_statistics(
    features: gpd.GeoDataFrame,
) -> None:
    """Print key Ankara road-feature statistics."""

    print("-" * 70)

    print(
        "Feature row count: "
        f"{len(features):,}"
    )

    print(
        "Cells with road data: "
        f"{int((features['road_length_m'] > 0).sum()):,}"
    )

    print(
        "Cells without road data: "
        f"{int((features['road_length_m'] == 0).sum()):,}"
    )

    print(
        "Mean road density: "
        f"{features['road_density_km_per_km2'].mean():,.2f} "
        "km/km²"
    )

    print(
        "Median distance to main road: "
        f"{features['distance_to_main_road_m'].median():,.2f} m"
    )

    print(
        "Maximum distance to main road: "
        f"{features['distance_to_main_road_m'].max():,.2f} m"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Ankara road-feature CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Create resumable Ankara grid road features."
        )
    )

    parser.add_argument(
        "--grid-size-m",
        type=int,
        default=DEFAULT_GRID_SIZE_METERS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recalculate all batches even when "
            "valid checkpoints exist."
        ),
    )

    parser.add_argument(
        "--write-geojson",
        action="store_true",
        help=(
            "Also save the large web-compatible GeoJSON."
        ),
    )

    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help=(
            "Skip the documentation preview image."
        ),
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    if arguments.grid_size_m <= 0:
        parser.error(
            "--grid-size-m must be positive."
        )

    if arguments.batch_size <= 0:
        parser.error(
            "--batch-size must be positive."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Run the Ankara road-feature pipeline."""

    paths = resolve_paths(
        arguments.grid_size_m
    )

    create_output_directories(
        paths
    )

    validate_input_files(
        paths
    )

    print("=" * 70)

    print(
        "VoltSight - Ankara Road Feature Pipeline"
    )

    print("=" * 70)

    print(
        f"Grid size: {arguments.grid_size_m:,} m"
    )

    print(
        f"Batch size: {arguments.batch_size:,}"
    )

    grid, roads = load_inputs(
        paths
    )

    metrics = process_batches(
        grid=grid,
        roads=roads,
        paths=paths,
        batch_size=arguments.batch_size,
        force=arguments.force,
    )

    features = assemble_features(
        grid,
        metrics,
    )

    save_outputs(
        features,
        paths,
        write_geojson=(
            arguments.write_geojson
        ),
    )

    if not arguments.skip_preview:
        create_preview(
            features,
            roads,
            paths,
        )

    create_summary(
        features,
        roads,
        paths,
        batch_size=arguments.batch_size,
    )

    print_statistics(
        features
    )

    print("=" * 70)

    print(
        "Ankara road feature pipeline "
        "completed successfully."
    )

    print("=" * 70)

    return features


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the command-line pipeline."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
