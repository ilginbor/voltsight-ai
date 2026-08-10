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
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from voltsight.features.create_charging_features import (  # noqa: E402
    CHARGING_COUNT_RADII_METERS,
    CHARGING_FEATURE_COLUMNS,
    calculate_local_station_features,
    calculate_nearest_station_features,
    calculate_radius_count_features,
    create_station_points,
    validate_features,
)


DEFAULT_BATCH_SIZE = 5_000

BASE_FEATURE_LAYER_NAME = "grid_parking_features"
CHARGING_STATION_LAYER_NAME = "charging_stations_merged"
FEATURE_LAYER_NAME = "grid_charging_features"

CHECKPOINT_COLUMNS = [
    "grid_id",
    *CHARGING_FEATURE_COLUMNS,
]


@dataclass(
    frozen=True,
    slots=True,
)
class AnkaraChargingFeaturePaths:
    """Paths used by the Ankara charging-feature pipeline."""

    base_features_gpkg: Path
    charging_stations_gpkg: Path
    batch_directory: Path
    output_gpkg: Path
    output_csv: Path
    output_geojson: Path
    preview_png: Path
    summary_md: Path


def resolve_paths() -> AnkaraChargingFeaturePaths:
    """Resolve deterministic Ankara charging-feature paths."""

    return AnkaraChargingFeaturePaths(
        base_features_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_parking_features.gpkg"
        ),
        charging_stations_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_charging_stations_merged.gpkg"
        ),
        batch_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_charging_feature_batches_500m"
        ),
        output_gpkg=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_charging_features.gpkg"
        ),
        output_csv=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_charging_features.csv"
        ),
        output_geojson=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ankara_grid_charging_features.geojson"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / "ankara_charging_features_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / "ankara_charging_features_summary.md"
        ),
    )


def create_output_directories(
    paths: AnkaraChargingFeaturePaths,
) -> None:
    """Create all output directories."""

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


def normalize_station_flags(
    stations: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Normalize connector and provenance flags."""

    result = stations.copy()

    boolean_columns = (
        "has_ac_connector",
        "has_dc_connector",
    )

    for column in boolean_columns:
        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
        )

    integer_columns = (
        "source_osm",
        "source_epdk",
    )

    for column in integer_columns:
        if column not in result.columns:
            result[column] = 0

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    return result


def load_inputs(
    paths: AnkaraChargingFeaturePaths,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load and validate Ankara grid and charging stations."""

    for path in (
        paths.base_features_gpkg,
        paths.charging_stations_gpkg,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input was not found: {path}"
            )

    base_features = gpd.read_file(
        paths.base_features_gpkg,
        layer=BASE_FEATURE_LAYER_NAME,
    )

    stations = gpd.read_file(
        paths.charging_stations_gpkg,
        layer=CHARGING_STATION_LAYER_NAME,
    )

    if base_features.empty:
        raise ValueError(
            "The Ankara parking-feature grid is empty."
        )

    if stations.empty:
        raise ValueError(
            "The Ankara charging inventory is empty."
        )

    if base_features.crs is None:
        raise ValueError(
            "The Ankara grid has no CRS."
        )

    if stations.crs is None:
        raise ValueError(
            "The Ankara charging inventory has no CRS."
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
            "Missing grid columns: "
            f"{sorted(missing_grid_columns)}"
        )

    required_station_columns = {
        "station_id",
        "capacity_numeric",
        "has_ac_connector",
        "has_dc_connector",
        "geometry",
    }

    missing_station_columns = (
        required_station_columns
        - set(stations.columns)
    )

    if missing_station_columns:
        raise ValueError(
            "Missing charging-station columns: "
            f"{sorted(missing_station_columns)}"
        )

    if base_features["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate Ankara grid IDs were found."
        )

    if stations["station_id"].duplicated().any():
        raise ValueError(
            "Duplicate charging station IDs were found."
        )

    if stations.crs != base_features.crs:
        stations = stations.to_crs(
            base_features.crs
        )

    stations = normalize_station_flags(
        stations
    )

    stations["capacity_numeric"] = (
        pd.to_numeric(
            stations["capacity_numeric"],
            errors="coerce",
        )
    )

    stations.loc[
        stations["capacity_numeric"] < 0,
        "capacity_numeric",
    ] = np.nan

    stations = stations.loc[
        stations.geometry.notna()
    ].copy()

    stations = stations.loc[
        ~stations.geometry.is_empty
    ].copy()

    if not stations.geometry.is_valid.all():
        raise ValueError(
            "Invalid charging-station geometries were found."
        )

    print(
        "Loaded Ankara grid cells: "
        f"{len(base_features):,}"
    )

    print(
        "Loaded charging stations: "
        f"{len(stations):,}"
    )

    print(
        "OSM source stations: "
        f"{int(stations['source_osm'].sum()):,}"
    )

    print(
        "EPDK source stations: "
        f"{int(stations['source_epdk'].sum()):,}"
    )

    print(
        "AC stations: "
        f"{int(stations['has_ac_connector'].sum()):,}"
    )

    print(
        "DC stations: "
        f"{int(stations['has_dc_connector'].sum()):,}"
    )

    print(
        f"Analysis CRS: {base_features.crs}"
    )

    return (
        base_features,
        stations,
    )


def build_batch_metrics(
    grid_batch: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    station_points: gpd.GeoDataFrame,
    ac_station_points: gpd.GeoDataFrame,
    dc_station_points: gpd.GeoDataFrame,
    local_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate charging features for one grid batch."""

    grid_ids = (
        grid_batch["grid_id"]
        .astype(str)
        .tolist()
    )

    grid_id_set = set(
        grid_ids
    )

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

    nearest_features = (
        calculate_nearest_station_features(
            grid_batch,
            stations,
        )
    )

    metrics = metrics.merge(
        local_batch,
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
        CHARGING_COUNT_RADII_METERS
    ):
        output_column = (
            "charging_station_count_within_"
            f"{radius_meters}m"
        )

        radius_features = (
            calculate_radius_count_features(
                grid_batch,
                station_points,
                radius_meters,
                output_column,
            )
        )

        metrics = metrics.merge(
            radius_features,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

    ac_features = (
        calculate_radius_count_features(
            grid_batch,
            ac_station_points,
            1_000,
            "ac_station_count_within_1000m",
        )
    )

    dc_features = (
        calculate_radius_count_features(
            grid_batch,
            dc_station_points,
            1_000,
            "dc_station_count_within_1000m",
        )
    )

    metrics = metrics.merge(
        ac_features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    metrics = metrics.merge(
        dc_features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    zero_fill_columns = [
        "charging_station_count",
        "known_charging_capacity",
        "charging_capacity_record_count",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    ]

    for column in zero_fill_columns:
        metrics[column] = (
            pd.to_numeric(
                metrics[column],
                errors="coerce",
            )
            .fillna(0)
        )

    integer_columns = [
        "charging_station_count",
        "charging_capacity_record_count",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    ]

    for column in integer_columns:
        metrics[column] = (
            metrics[column]
            .astype(int)
        )

    metrics[
        "has_existing_charging_station"
    ] = (
        metrics[
            "charging_station_count"
        ] > 0
    ).astype(int)

    metrics[
        "known_charging_capacity"
    ] = metrics[
        "known_charging_capacity"
    ].round(2)

    metrics[
        "distance_to_nearest_charging_station_m"
    ] = pd.to_numeric(
        metrics[
            "distance_to_nearest_charging_station_m"
        ],
        errors="coerce",
    ).round(2)

    if metrics[
        "distance_to_nearest_charging_station_m"
    ].isna().any():
        raise RuntimeError(
            "Some grid cells could not be matched "
            "to a charging station."
        )

    if (
        metrics[
            "charging_station_count_within_1000m"
        ]
        > metrics[
            "charging_station_count_within_2000m"
        ]
    ).any():
        raise RuntimeError(
            "A 1-km station count exceeds "
            "its 2-km station count."
        )

    return metrics[
        CHECKPOINT_COLUMNS
    ].copy()


def batch_output_path(
    paths: AnkaraChargingFeaturePaths,
    batch_number: int,
) -> Path:
    """Return one checkpoint CSV path."""

    return (
        paths.batch_directory
        / f"batch_{batch_number:04d}.csv"
    )


def save_batch(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Atomically save one charging checkpoint."""

    temporary_path = (
        output_path.with_suffix(
            ".csv.tmp"
        )
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
    expected_ids: Sequence[str],
) -> pd.DataFrame:
    """Load and validate one checkpoint."""

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
            "Cached batch is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if cached["grid_id"].duplicated().any():
        raise ValueError(
            "Cached batch contains duplicate grid IDs."
        )

    expected_ids = [
        str(value)
        for value in expected_ids
    ]

    if (
        set(cached["grid_id"])
        != set(expected_ids)
    ):
        raise ValueError(
            "Cached batch does not match "
            "the expected grid cells."
        )

    return (
        cached
        .set_index("grid_id")
        .loc[expected_ids]
        .reset_index()
    )


def process_batches(
    base_features: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    paths: AnkaraChargingFeaturePaths,
    *,
    batch_size: int,
    force: bool,
) -> pd.DataFrame:
    """Calculate all Ankara charging features."""

    station_points = create_station_points(
        stations
    )

    ac_station_points = (
        station_points.loc[
            station_points[
                "has_ac_connector"
            ]
        ].copy()
    )

    dc_station_points = (
        station_points.loc[
            station_points[
                "has_dc_connector"
            ]
        ].copy()
    )

    local_features = (
        calculate_local_station_features(
            base_features,
            station_points,
        )
    )

    local_features["grid_id"] = (
        local_features["grid_id"]
        .astype(str)
    )

    _ = stations.sindex
    _ = station_points.sindex

    if not ac_station_points.empty:
        _ = ac_station_points.sindex

    if not dc_station_points.empty:
        _ = dc_station_points.sindex

    batch_count = math.ceil(
        len(base_features)
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
            f"Batch {batch_number}/{batch_count}"
        )

        print(
            "Grid rows: "
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
                    "[SKIP] Valid checkpoint: "
                    f"{output_path.name}"
                )

                continue

            except Exception as error:
                print(
                    "Invalid checkpoint; recalculating: "
                    f"{error}"
                )

        metrics = build_batch_metrics(
            grid_batch=grid_batch,
            stations=stations,
            station_points=station_points,
            ac_station_points=ac_station_points,
            dc_station_points=dc_station_points,
            local_features=local_features,
        )

        save_batch(
            metrics,
            output_path,
        )

        completed_frames.append(
            metrics
        )

        print(
            "[SUCCESS] "
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
    """Attach charging metrics to the complete Ankara grid."""

    if metrics["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate charging metric IDs were found."
        )

    if len(metrics) != len(base_features):
        raise ValueError(
            "Charging metric row count does not "
            "match the Ankara grid."
        )

    old_columns = [
        column
        for column in CHARGING_FEATURE_COLUMNS
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
    paths: AnkaraChargingFeaturePaths,
    *,
    write_geojson: bool,
) -> None:
    """Save GIS and machine-learning outputs."""

    for path in (
        paths.output_gpkg,
        paths.output_csv,
    ):
        if path.exists():
            path.unlink()

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
        if paths.output_geojson.exists():
            paths.output_geojson.unlink()

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


def create_preview(
    features: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    paths: AnkaraChargingFeaturePaths,
) -> None:
    """Create Ankara charging accessibility preview."""

    station_points = create_station_points(
        stations
    )

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column=(
            "distance_to_nearest_charging_station_m"
        ),
        legend=True,
        linewidth=0,
        alpha=0.9,
        legend_kwds={
            "label": (
                "Distance to nearest charging station (m)"
            ),
            "shrink": 0.65,
        },
    )

    station_points.plot(
        ax=axis,
        markersize=5,
        alpha=0.8,
    )

    axis.set_title(
        "VoltSight - Ankara EV Charging Accessibility"
    )

    axis.set_aspect("equal")
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
        f"Preview saved: {paths.preview_png}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    paths: AnkaraChargingFeaturePaths,
    *,
    batch_size: int,
) -> None:
    """Create Ankara charging-feature summary."""

    cells_with_station = int(
        features[
            "has_existing_charging_station"
        ].sum()
    )

    cells_within_1000 = int(
        (
            features[
                "charging_station_count_within_1000m"
            ] > 0
        ).sum()
    )

    cells_within_2000 = int(
        (
            features[
                "charging_station_count_within_2000m"
            ] > 0
        ).sum()
    )

    summary = f"""# Ankara Charging Feature Summary

## Inventory

- Final analysis charging stations: {len(stations):,}
- OSM-source stations: {int(stations["source_osm"].sum()):,}
- Supplemental EPDK-source stations: {int(stations["source_epdk"].sum()):,}
- Stations with mapped AC connector: {int(stations["has_ac_connector"].sum()):,}
- Stations with mapped DC connector: {int(stations["has_dc_connector"].sum()):,}
- Stations with known capacity: {int(stations["capacity_numeric"].notna().sum()):,}

## Grid Results

- Grid cell count: {len(features):,}
- Cells containing a charging station: {cells_with_station:,}
- Cells with a station within 1,000 m: {cells_within_1000:,}
- Cells with a station within 2,000 m: {cells_within_2000:,}
- Mean distance to nearest station: {features["distance_to_nearest_charging_station_m"].mean():,.2f} m
- Median distance to nearest station: {features["distance_to_nearest_charging_station_m"].median():,.2f} m
- Maximum distance to nearest station: {features["distance_to_nearest_charging_station_m"].max():,.2f} m

## Processing

- Batch size: {batch_size:,}
- Batch count: {math.ceil(len(features) / batch_size):,}
- Analysis CRS: {features.crs}

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

{datetime.now(timezone.utc).isoformat()}
"""

    paths.summary_md.write_text(
        summary,
        encoding="utf-8",
    )


def print_statistics(
    features: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
) -> None:
    """Print key feature statistics."""

    print("-" * 70)

    print(
        "Charging station count: "
        f"{len(stations):,}"
    )

    print(
        "Feature row count: "
        f"{len(features):,}"
    )

    print(
        "Grid cells containing station: "
        f"{int(features['has_existing_charging_station'].sum()):,}"
    )

    print(
        "Grid cells with station within 1,000 m: "
        f"{int((features['charging_station_count_within_1000m'] > 0).sum()):,}"
    )

    print(
        "Grid cells with station within 2,000 m: "
        f"{int((features['charging_station_count_within_2000m'] > 0).sum()):,}"
    )

    print(
        "Median nearest station distance: "
        f"{features['distance_to_nearest_charging_station_m'].median():,.2f} m"
    )

    print(
        "Maximum nearest station distance: "
        f"{features['distance_to_nearest_charging_station_m'].max():,.2f} m"
    )

    print(
        "AC stations: "
        f"{int(stations['has_ac_connector'].sum()):,}"
    )

    print(
        "DC stations: "
        f"{int(stations['has_dc_connector'].sum()):,}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Create Ankara charging-station grid features."
        )
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
    """Parse command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    if arguments.batch_size <= 0:
        parser.error(
            "--batch-size must be positive."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Run the complete Ankara charging-feature pipeline."""

    paths = resolve_paths()

    create_output_directories(
        paths
    )

    print("=" * 70)

    print(
        "VoltSight - Ankara Charging Feature Pipeline"
    )

    print("=" * 70)

    base_features, stations = load_inputs(
        paths
    )

    metrics = process_batches(
        base_features=base_features,
        stations=stations,
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
            stations,
            paths,
        )

    create_summary(
        features,
        stations,
        paths,
        batch_size=arguments.batch_size,
    )

    print_statistics(
        features,
        stations,
    )

    print("=" * 70)

    print(
        "Ankara charging feature pipeline "
        "completed successfully."
    )

    print("=" * 70)

    return features


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run CLI pipeline."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
