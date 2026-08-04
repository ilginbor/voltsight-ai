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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from voltsight.features.create_parking_features import (  # noqa: E402
    PARKING_COUNT_RADII_METERS,
    PARKING_FEATURE_COLUMNS,
    POLYGON_GEOMETRY_TYPES,
    calculate_local_parking_features,
    calculate_nearest_parking_features,
    calculate_radius_count_features,
    create_parking_points,
    validate_features,
)


DEFAULT_GRID_SIZE_METERS = 500
DEFAULT_BATCH_SIZE = 5_000

BASE_FEATURE_LAYER_NAME = "grid_road_features"
PARKING_LAYER_NAME = "parking_features"
FEATURE_LAYER_NAME = "grid_parking_features"

CHECKPOINT_COLUMNS = [
    "grid_id",
    *PARKING_FEATURE_COLUMNS,
]


@dataclass(frozen=True, slots=True)
class AnkaraParkingFeaturePaths:
    """Paths used by the Ankara parking-feature pipeline."""

    base_features_gpkg: Path
    parking_gpkg: Path
    batch_directory: Path
    output_gpkg: Path
    output_geojson: Path
    output_csv: Path
    preview_png: Path
    summary_md: Path


def resolve_paths(
    grid_size_m: int = DEFAULT_GRID_SIZE_METERS,
) -> AnkaraParkingFeaturePaths:
    """Resolve deterministic Ankara parking-feature paths."""

    if grid_size_m <= 0:
        raise ValueError(
            "Grid size must be positive."
        )

    return AnkaraParkingFeaturePaths(
        base_features_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_road_features.gpkg"
        ),
        parking_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_parking_features.gpkg"
        ),
        batch_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / (
                "ankara_parking_feature_batches_"
                f"{grid_size_m}m"
            )
        ),
        output_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_parking_features.gpkg"
        ),
        output_geojson=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_parking_features.geojson"
        ),
        output_csv=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_parking_features.csv"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / "ankara_parking_accessibility_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / "ankara_grid_parking_features_summary.md"
        ),
    )


def create_output_directories(
    paths: AnkaraParkingFeaturePaths,
) -> None:
    """Create output directories."""

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
    paths: AnkaraParkingFeaturePaths,
) -> None:
    """Ensure required Ankara inputs exist."""

    missing = [
        path
        for path in (
            paths.base_features_gpkg,
            paths.parking_gpkg,
        )
        if not path.exists()
    ]

    if not missing:
        return

    missing_text = "\n".join(
        f"- {path}"
        for path in missing
    )

    raise FileNotFoundError(
        "Required Ankara inputs are missing:\n"
        f"{missing_text}"
    )


def load_inputs(
    paths: AnkaraParkingFeaturePaths,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load and validate Ankara grid and parking data."""

    base_features = gpd.read_file(
        paths.base_features_gpkg,
        layer=BASE_FEATURE_LAYER_NAME,
    )

    parking = gpd.read_file(
        paths.parking_gpkg,
        layer=PARKING_LAYER_NAME,
    )

    if base_features.empty:
        raise ValueError(
            "The Ankara road-feature grid is empty."
        )

    if parking.empty:
        raise ValueError(
            "The merged Ankara parking dataset is empty."
        )

    if base_features.crs is None:
        raise ValueError(
            "The Ankara grid has no CRS."
        )

    if parking.crs is None:
        raise ValueError(
            "The Ankara parking dataset has no CRS."
        )

    if not base_features.crs.is_projected:
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
        - set(base_features.columns)
    )

    if missing_grid_columns:
        raise ValueError(
            "The Ankara grid is missing columns: "
            f"{sorted(missing_grid_columns)}"
        )

    required_parking_columns = {
        "parking_id",
        "capacity_numeric",
        "geometry",
    }

    missing_parking_columns = (
        required_parking_columns
        - set(parking.columns)
    )

    if missing_parking_columns:
        raise ValueError(
            "The parking dataset is missing columns: "
            f"{sorted(missing_parking_columns)}"
        )

    if base_features["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs were found."
        )

    if parking["parking_id"].duplicated().any():
        raise ValueError(
            "Duplicate parking IDs were found."
        )

    if parking.crs != base_features.crs:
        parking = parking.to_crs(
            base_features.crs
        )

    parking = parking.loc[
        parking.geometry.notna()
    ].copy()

    parking = parking.loc[
        ~parking.geometry.is_empty
    ].copy()

    if not parking.geometry.is_valid.all():
        raise ValueError(
            "Invalid parking geometries were found."
        )

    parking["capacity_numeric"] = pd.to_numeric(
        parking["capacity_numeric"],
        errors="coerce",
    )

    print(
        "Loaded Ankara grid cells: "
        f"{len(base_features):,}"
    )

    print(
        "Loaded unique parking features: "
        f"{len(parking):,}"
    )

    print(
        "Parking features with known capacity: "
        f"{int(parking['capacity_numeric'].notna().sum()):,}"
    )

    print(
        f"Analysis CRS: {base_features.crs}"
    )

    return base_features, parking


def create_parking_union(
    parking: gpd.GeoDataFrame,
) -> Any | None:
    """Create a non-overlapping union of parking polygons."""

    polygon_parking = parking.loc[
        parking.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
        )
    ].copy()

    if polygon_parking.empty:
        return None

    parking_union = (
        polygon_parking.geometry.union_all()
    )

    if parking_union.is_empty:
        return None

    print(
        "Parking polygon count: "
        f"{len(polygon_parking):,}"
    )

    return parking_union


def calculate_parking_area_batch(
    grid_batch: gpd.GeoDataFrame,
    parking_union: Any | None,
) -> pd.DataFrame:
    """Calculate non-overlapping parking area per grid cell."""

    if parking_union is None:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "parking_area_m2",
            ]
        )

    intersects_mask = (
        grid_batch.geometry.intersects(
            parking_union
        )
    )

    selected = grid_batch.loc[
        intersects_mask,
        [
            "grid_id",
            "geometry",
        ],
    ].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "parking_area_m2",
            ]
        )

    intersections = (
        selected.geometry.intersection(
            parking_union
        )
    )

    selected["parking_area_m2"] = (
        intersections.area
    )

    selected = selected.loc[
        selected["parking_area_m2"] > 0
    ].copy()

    return pd.DataFrame(
        selected[
            [
                "grid_id",
                "parking_area_m2",
            ]
        ]
    )


def build_batch_metrics(
    grid_batch: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
    parking_points: gpd.GeoDataFrame,
    parking_union: Any | None,
    local_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate every parking feature for one grid batch."""

    grid_ids = (
        grid_batch["grid_id"]
        .astype(str)
        .tolist()
    )

    grid_id_set = set(grid_ids)

    metrics = pd.DataFrame(
        {
            "grid_id": grid_ids,
        }
    )

    local_batch = local_features.loc[
        local_features["grid_id"]
        .astype(str)
        .isin(grid_id_set)
    ].copy()

    area_features = (
        calculate_parking_area_batch(
            grid_batch,
            parking_union,
        )
    )

    nearest_features = (
        calculate_nearest_parking_features(
            grid_batch,
            parking,
        )
    )

    metrics = metrics.merge(
        local_batch,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    metrics = metrics.merge(
        area_features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    metrics = metrics.merge(
        nearest_features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    for radius_meters in (
        PARKING_COUNT_RADII_METERS
    ):
        radius_features = (
            calculate_radius_count_features(
                grid_batch,
                parking_points,
                radius_meters,
            )
        )

        metrics = metrics.merge(
            radius_features,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

    zero_fill_columns = [
        "parking_count",
        "parking_area_m2",
        "known_parking_capacity",
        "parking_capacity_record_count",
        "parking_count_within_500m",
        "parking_count_within_1000m",
    ]

    metrics[zero_fill_columns] = (
        metrics[zero_fill_columns]
        .fillna(0)
    )

    integer_columns = [
        "parking_count",
        "parking_capacity_record_count",
        "parking_count_within_500m",
        "parking_count_within_1000m",
    ]

    for column in integer_columns:
        metrics[column] = (
            metrics[column].astype(int)
        )

    metrics["parking_area_m2"] = (
        metrics["parking_area_m2"]
        .round(2)
    )

    metrics["known_parking_capacity"] = (
        metrics["known_parking_capacity"]
        .round(2)
    )

    cell_area_by_id = (
        grid_batch.set_index(
            "grid_id"
        )["cell_area_m2"]
    )

    metrics["parking_area_ratio"] = (
        metrics["parking_area_m2"]
        / metrics["grid_id"].map(
            cell_area_by_id
        )
    ).round(6)

    metrics[
        "distance_to_nearest_parking_m"
    ] = pd.to_numeric(
        metrics[
            "distance_to_nearest_parking_m"
        ],
        errors="coerce",
    ).round(2)

    if metrics[
        "distance_to_nearest_parking_m"
    ].isna().any():
        raise RuntimeError(
            "Some grid cells could not be matched "
            "to a nearest parking feature."
        )

    if (
        metrics[
            "parking_count_within_500m"
        ]
        > metrics[
            "parking_count_within_1000m"
        ]
    ).any():
        raise RuntimeError(
            "A 500-metre parking count exceeds "
            "its 1,000-metre count."
        )

    return metrics[
        CHECKPOINT_COLUMNS
    ].copy()


def batch_output_path(
    paths: AnkaraParkingFeaturePaths,
    batch_number: int,
) -> Path:
    """Return one checkpoint CSV path."""

    return (
        paths.batch_directory
        / f"batch_{batch_number:04d}.csv"
    )


def save_batch_metrics(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Atomically save one batch checkpoint."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".csv.tmp"
    )

    metrics.to_csv(
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
    """Load and validate a completed batch checkpoint."""

    cached = pd.read_csv(
        output_path,
        dtype={
            "grid_id": str,
        },
    )

    missing_columns = (
        set(CHECKPOINT_COLUMNS)
        - set(cached.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{output_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if cached["grid_id"].duplicated().any():
        raise ValueError(
            f"{output_path.name} contains duplicate grid IDs."
        )

    expected_ids = [
        str(grid_id)
        for grid_id in expected_grid_ids
    ]

    actual_ids = set(
        cached["grid_id"].astype(str)
    )

    if actual_ids != set(expected_ids):
        raise ValueError(
            f"{output_path.name} does not match "
            "the expected grid cells."
        )

    return (
        cached.set_index("grid_id")
        .loc[expected_ids]
        .reset_index()
    )


def process_batches(
    base_features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
    paths: AnkaraParkingFeaturePaths,
    *,
    batch_size: int,
    force: bool,
) -> pd.DataFrame:
    """Process all Ankara parking-feature batches."""

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    parking_points = create_parking_points(
        parking
    )

    parking_union = create_parking_union(
        parking
    )

    local_features = (
        calculate_local_parking_features(
            base_features,
            parking_points,
        )
    )

    local_features["grid_id"] = (
        local_features["grid_id"]
        .astype(str)
    )

    _ = parking.sindex
    _ = parking_points.sindex
    _ = base_features.sindex

    batch_count = math.ceil(
        len(base_features) / batch_size
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
            len(base_features),
        )

        grid_batch = base_features.iloc[
            start_index:end_index
        ].copy()

        expected_ids = (
            grid_batch["grid_id"]
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

        if output_path.exists() and not force:
            try:
                cached = load_cached_batch(
                    output_path,
                    expected_ids,
                )

                completed_frames.append(
                    cached
                )

                print(
                    "[SKIP] Valid checkpoint loaded: "
                    f"{output_path.name}"
                )

                continue

            except Exception as error:
                print(
                    "Checkpoint is invalid and will "
                    f"be recalculated: {error}"
                )

        metrics = build_batch_metrics(
            grid_batch=grid_batch,
            parking=parking,
            parking_points=parking_points,
            parking_union=parking_union,
            local_features=local_features,
        )

        save_batch_metrics(
            metrics,
            output_path,
        )

        completed_frames.append(
            metrics
        )

        print(
            "[SUCCESS] Batch saved: "
            f"{output_path.name}"
        )

    return pd.concat(
        completed_frames,
        ignore_index=True,
    )


def assemble_features(
    base_features: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Merge parking metrics onto the Ankara road grid."""

    if metrics["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate metric grid IDs were found."
        )

    if len(metrics) != len(base_features):
        raise ValueError(
            "Metric row count does not match the grid. "
            f"Expected {len(base_features):,}, "
            f"found {len(metrics):,}."
        )

    base_ids = set(
        base_features["grid_id"].astype(str)
    )

    metric_ids = set(
        metrics["grid_id"].astype(str)
    )

    if base_ids != metric_ids:
        raise ValueError(
            "Metric IDs do not match the Ankara grid."
        )

    old_columns = [
        column
        for column in PARKING_FEATURE_COLUMNS
        if column in base_features.columns
    ]

    base_grid = base_features.drop(
        columns=old_columns,
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
        crs=base_features.crs,
    )

    validate_features(
        result,
        base_features,
    )

    return result


def save_outputs(
    features: gpd.GeoDataFrame,
    paths: AnkaraParkingFeaturePaths,
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
        layer=FEATURE_LAYER_NAME,
        driver="GPKG",
    )

    pd.DataFrame(
        features.drop(
            columns="geometry"
        )
    ).to_csv(
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
    parking: gpd.GeoDataFrame,
    paths: AnkaraParkingFeaturePaths,
) -> None:
    """Create Ankara parking-accessibility preview."""

    parking_points = create_parking_points(
        parking
    )

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column="parking_count_within_1000m",
        legend=True,
        linewidth=0,
        alpha=0.90,
        legend_kwds={
            "label": (
                "Parking features within 1,000 metres"
            ),
            "shrink": 0.65,
        },
    )

    parking_points.plot(
        ax=axis,
        markersize=1.5,
        alpha=0.55,
    )

    axis.set_title(
        "VoltSight - Ankara Parking Accessibility"
    )

    axis.set_aspect("equal")
    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        paths.preview_png,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Parking preview saved: {paths.preview_png}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
    paths: AnkaraParkingFeaturePaths,
    *,
    batch_size: int,
) -> None:
    """Create Ankara parking-feature summary."""

    cells_with_local_parking = int(
        (
            features["parking_count"] > 0
        ).sum()
    )

    cells_with_500m_parking = int(
        (
            features[
                "parking_count_within_500m"
            ] > 0
        ).sum()
    )

    cells_with_1000m_parking = int(
        (
            features[
                "parking_count_within_1000m"
            ] > 0
        ).sum()
    )

    summary = f"""# Ankara Grid Parking Feature Summary

## Inputs

- Ankara grid cells: {len(features):,}
- Unique parking features: {len(parking):,}
- Parking features with known capacity: {int(parking["capacity_numeric"].notna().sum()):,}
- Analysis CRS: {features.crs}

## Processing

- Grid batch size: {batch_size:,}
- Batch count: {math.ceil(len(features) / batch_size):,}
- Checkpoint directory: `data/interim/{paths.batch_directory.name}`

## Accessibility Results

- Cells containing parking: {cells_with_local_parking:,}
- Cells with parking within 500 metres: {cells_with_500m_parking:,}
- Cells with parking within 1,000 metres: {cells_with_1000m_parking:,}
- Mean distance to nearest parking: {features["distance_to_nearest_parking_m"].mean():,.2f} m
- Median distance to nearest parking: {features["distance_to_nearest_parking_m"].median():,.2f} m
- Maximum distance to nearest parking: {features["distance_to_nearest_parking_m"].max():,.2f} m
- Mean parking count within 500 metres: {features["parking_count_within_500m"].mean():,.2f}
- Mean parking count within 1,000 metres: {features["parking_count_within_1000m"].mean():,.2f}

## Generated Features

- `parking_count`
- `parking_area_m2`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`

## Generated Outputs

- `data/processed/ankara_grid_parking_features.gpkg`
- `data/processed/ankara_grid_parking_features.csv`
- `docs/ankara_grid_parking_features_summary.md`

## Method

Unique OpenStreetMap parking features were connected to the Ankara
500 x 500 metre grid in resumable batches.

Representative points were used for cell assignment and radius-based
counts. Original parking geometries were used for nearest-distance
calculations.

Parking polygons were unioned before grid intersection, preventing
overlapping mapped parking areas from being counted twice.

## Data Limitation

OpenStreetMap parking coverage and capacity attributes can be
incomplete. These variables represent mapped parking accessibility,
not a complete official inventory.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    paths.summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Parking summary saved: {paths.summary_md}"
    )


def print_statistics(
    features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> None:
    """Print key parking statistics."""

    print("-" * 70)

    print(
        "Prepared parking feature count: "
        f"{len(parking):,}"
    )

    print(
        "Feature row count: "
        f"{len(features):,}"
    )

    print(
        "Grid cells containing parking: "
        f"{int((features['parking_count'] > 0).sum()):,}"
    )

    print(
        "Grid cells with parking within 500 m: "
        f"{int((features['parking_count_within_500m'] > 0).sum()):,}"
    )

    print(
        "Grid cells with parking within 1,000 m: "
        f"{int((features['parking_count_within_1000m'] > 0).sum()):,}"
    )

    print(
        "Median distance to nearest parking: "
        f"{features['distance_to_nearest_parking_m'].median():,.2f} m"
    )

    print(
        "Maximum distance to nearest parking: "
        f"{features['distance_to_nearest_parking_m'].max():,.2f} m"
    )

    print(
        "Mean parking count within 1,000 m: "
        f"{features['parking_count_within_1000m'].mean():,.2f}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Create resumable Ankara grid parking features."
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
    )

    parser.add_argument(
        "--write-geojson",
        action="store_true",
    )

    parser.add_argument(
        "--skip-preview",
        action="store_true",
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and validate CLI arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(argv)

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
    """Run the Ankara parking-feature pipeline."""

    paths = resolve_paths(
        arguments.grid_size_m
    )

    create_output_directories(paths)
    validate_input_files(paths)

    print("=" * 70)

    print(
        "VoltSight - Ankara Parking Feature Pipeline"
    )

    print("=" * 70)

    print(
        f"Grid size: {arguments.grid_size_m:,} m"
    )

    print(
        f"Batch size: {arguments.batch_size:,}"
    )

    base_features, parking = load_inputs(
        paths
    )

    metrics = process_batches(
        base_features=base_features,
        parking=parking,
        paths=paths,
        batch_size=arguments.batch_size,
        force=arguments.force,
    )

    features = assemble_features(
        base_features,
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
            parking,
            paths,
        )

    create_summary(
        features,
        parking,
        paths,
        batch_size=arguments.batch_size,
    )

    print_statistics(
        features,
        parking,
    )

    print("=" * 70)

    print(
        "Ankara parking feature pipeline "
        "completed successfully."
    )

    print("=" * 70)

    return features


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run command-line pipeline."""

    arguments = parse_arguments(argv)

    run_pipeline(arguments)


if __name__ == "__main__":
    main()
