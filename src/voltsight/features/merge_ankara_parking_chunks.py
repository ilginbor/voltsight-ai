from __future__ import annotations

import argparse
import json
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

DEFAULT_CHUNK_SIZE_METERS = 8_000
DEFAULT_DOWNLOAD_BUFFER_METERS = 1_000

CHUNK_PLAN_LAYER_NAME = "road_download_chunks"
CHUNK_PARKING_LAYER_NAME = "parking_features_buffered"
MERGED_PARKING_LAYER_NAME = "parking_features"


@dataclass(frozen=True, slots=True)
class AnkaraParkingMergePaths:
    """Paths used by the Ankara parking merge pipeline."""

    chunk_plan_gpkg: Path
    download_directory: Path
    boundary_geojson: Path
    output_gpkg: Path
    manifest_csv: Path
    preview_png: Path
    summary_md: Path


def distance_token(distance_meters: int) -> str:
    """Create a deterministic filename token."""

    if distance_meters <= 0:
        raise ValueError(
            "Distance must be positive."
        )

    if distance_meters % 1_000 == 0:
        return f"{distance_meters // 1_000}km"

    return f"{distance_meters}m"


def resolve_paths(
    chunk_size_m: int = DEFAULT_CHUNK_SIZE_METERS,
) -> AnkaraParkingMergePaths:
    """Resolve deterministic Ankara parking paths."""

    token = distance_token(chunk_size_m)

    return AnkaraParkingMergePaths(
        chunk_plan_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / f"ankara_road_download_chunks_{token}.gpkg"
        ),
        download_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / f"ankara_parking_chunk_downloads_{token}"
        ),
        boundary_geojson=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "ankara_boundary_osm.geojson"
        ),
        output_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_parking_features.gpkg"
        ),
        manifest_csv=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_parking_merge_manifest.csv"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / "ankara_parking_features_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / "ankara_parking_features_summary.md"
        ),
    )


def create_output_directories(
    paths: AnkaraParkingMergePaths,
) -> None:
    """Create all required output directories."""

    for directory in {
        paths.output_gpkg.parent,
        paths.preview_png.parent,
        paths.summary_md.parent,
    }:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def metadata_path(
    paths: AnkaraParkingMergePaths,
    chunk_id: str,
) -> Path:
    """Return one chunk metadata path."""

    return (
        paths.download_directory
        / f"{chunk_id}_metadata.json"
    )


def parking_path(
    paths: AnkaraParkingMergePaths,
    chunk_id: str,
) -> Path:
    """Return one chunk parking GeoPackage path."""

    return (
        paths.download_directory
        / f"{chunk_id}_parking.gpkg"
    )


def load_chunk_plan(
    paths: AnkaraParkingMergePaths,
) -> gpd.GeoDataFrame:
    """Load and validate the Ankara chunk plan."""

    if not paths.chunk_plan_gpkg.exists():
        raise FileNotFoundError(
            "Ankara chunk plan was not found:\n"
            f"{paths.chunk_plan_gpkg}"
        )

    chunks = gpd.read_file(
        paths.chunk_plan_gpkg,
        layer=CHUNK_PLAN_LAYER_NAME,
    )

    required_columns = {
        "chunk_id",
        "chunk_order",
        "grid_cell_count",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(chunks.columns)
    )

    if missing_columns:
        raise ValueError(
            "Chunk plan is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if chunks.empty:
        raise ValueError(
            "The Ankara chunk plan is empty."
        )

    if chunks.crs is None:
        raise ValueError(
            "The Ankara chunk plan has no CRS."
        )

    if not chunks.crs.is_projected:
        raise ValueError(
            "The Ankara chunk plan must use a projected CRS."
        )

    if chunks["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

    return (
        chunks
        .sort_values("chunk_order")
        .reset_index(drop=True)
    )


def read_metadata(path: Path) -> dict[str, Any]:
    """Read one successful parking metadata file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk metadata was not found: {path}"
        )

    try:
        metadata = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid metadata JSON: {path}"
        ) from error

    if metadata.get("status") != "success":
        raise RuntimeError(
            "Chunk is not successfully completed: "
            f"{metadata.get('chunk_id', path.stem)}"
        )

    return metadata


def validate_chunk_results(
    chunks: gpd.GeoDataFrame,
    paths: AnkaraParkingMergePaths,
) -> pd.DataFrame:
    """Validate all 488 chunk results."""

    records: list[dict[str, Any]] = []

    for _, chunk in chunks.iterrows():
        chunk_id = str(
            chunk["chunk_id"]
        )

        metadata = read_metadata(
            metadata_path(
                paths,
                chunk_id,
            )
        )

        is_empty = bool(
            metadata.get(
                "is_empty",
                False,
            )
        )

        output_path = parking_path(
            paths,
            chunk_id,
        )

        if (
            not is_empty
            and not output_path.exists()
        ):
            raise FileNotFoundError(
                "Successful non-empty chunk has no "
                f"GeoPackage: {chunk_id}"
            )

        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_order": int(
                    chunk["chunk_order"]
                ),
                "grid_cell_count": int(
                    chunk["grid_cell_count"]
                ),
                "is_empty": is_empty,
                "parking_feature_count": int(
                    metadata.get(
                        "parking_feature_count",
                        0,
                    )
                ),
                "polygon_feature_count": int(
                    metadata.get(
                        "polygon_feature_count",
                        0,
                    )
                ),
                "point_feature_count": int(
                    metadata.get(
                        "point_feature_count",
                        0,
                    )
                ),
                "known_capacity_count": int(
                    metadata.get(
                        "known_capacity_count",
                        0,
                    )
                ),
                "output_path": (
                    output_path.as_posix()
                    if not is_empty
                    else ""
                ),
            }
        )

    manifest = pd.DataFrame(records)

    print(
        "Validated parking chunks: "
        f"{len(manifest):,}"
    )

    print(
        "Empty-success chunks: "
        f"{int(manifest['is_empty'].sum()):,}"
    )

    print(
        "Raw metadata parking records: "
        f"{int(manifest['parking_feature_count'].sum()):,}"
    )

    return manifest


def load_relevant_footprint(
    paths: AnkaraParkingMergePaths,
    target_crs: Any,
    download_buffer_m: int,
) -> Any:
    """Load Ankara and create the relevant parking footprint."""

    if not paths.boundary_geojson.exists():
        raise FileNotFoundError(
            "Ankara boundary was not found:\n"
            f"{paths.boundary_geojson}"
        )

    boundary = gpd.read_file(
        paths.boundary_geojson
    )

    if boundary.empty:
        raise ValueError(
            "The Ankara boundary is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "The Ankara boundary has no CRS."
        )

    boundary = boundary.to_crs(
        target_crs
    )

    footprint = (
        boundary.geometry
        .union_all()
        .buffer(download_buffer_m)
    )

    if footprint.is_empty:
        raise ValueError(
            "The buffered Ankara footprint is empty."
        )

    return footprint


def load_raw_parking(
    manifest: pd.DataFrame,
    paths: AnkaraParkingMergePaths,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Load every non-empty parking chunk."""

    frames: list[gpd.GeoDataFrame] = []

    non_empty_manifest = manifest.loc[
        ~manifest["is_empty"]
    ]

    for position, row in enumerate(
        non_empty_manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        chunk_id = str(
            row.chunk_id
        )

        parking = gpd.read_file(
            parking_path(
                paths,
                chunk_id,
            ),
            layer=CHUNK_PARKING_LAYER_NAME,
        )

        if parking.crs is None:
            raise ValueError(
                f"Parking CRS is missing for {chunk_id}."
            )

        if parking.crs != target_crs:
            parking = parking.to_crs(
                target_crs
            )

        required_columns = {
            "parking_id",
            "source_chunk_id",
            "source_chunk_order",
            "capacity_numeric",
            "parking_area_m2",
            "geometry",
        }

        missing_columns = (
            required_columns
            - set(parking.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{chunk_id} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        frames.append(parking)

        if (
            position % 20 == 0
            or position == len(
                non_empty_manifest
            )
        ):
            print(
                "Loaded non-empty chunks: "
                f"{position:,}/"
                f"{len(non_empty_manifest):,}"
            )

    if not frames:
        raise RuntimeError(
            "No Ankara parking records were loaded."
        )

    raw_parking = gpd.GeoDataFrame(
        pd.concat(
            frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=target_crs,
    )

    expected_count = int(
        manifest[
            "parking_feature_count"
        ].sum()
    )

    if len(raw_parking) != expected_count:
        raise RuntimeError(
            "Loaded parking row count does not match "
            "the metadata total. "
            f"Expected {expected_count:,}, "
            f"loaded {len(raw_parking):,}."
        )

    print(
        "Loaded raw parking records: "
        f"{len(raw_parking):,}"
    )

    return raw_parking


def filter_to_relevant_footprint(
    parking: gpd.GeoDataFrame,
    footprint: Any,
) -> gpd.GeoDataFrame:
    """Keep parking records relevant to Ankara and its buffer."""

    result = parking.loc[
        parking.geometry.notna()
    ].copy()

    result = result.loc[
        ~result.geometry.is_empty
    ].copy()

    result = result.loc[
        result.geometry.intersects(
            footprint
        )
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=parking.crs,
    )


def geometry_measure(geometry: Any) -> float:
    """Measure geometry completeness for deterministic deduplication."""

    if geometry.geom_type in {
        "Polygon",
        "MultiPolygon",
    }:
        return float(
            geometry.area
        )

    if geometry.geom_type in {
        "LineString",
        "MultiLineString",
    }:
        return float(
            geometry.length
        )

    return 0.0


def deduplicate_parking_features(
    parking: gpd.GeoDataFrame,
) -> tuple[
    gpd.GeoDataFrame,
    int,
    int,
]:
    """Deduplicate overlapping chunk records by OSM parking ID."""

    if parking.empty:
        raise ValueError(
            "Parking dataset is empty."
        )

    result = parking.copy()

    result["parking_id"] = (
        result["parking_id"].astype(str)
    )

    result[
        "source_chunk_order"
    ] = pd.to_numeric(
        result["source_chunk_order"],
        errors="raise",
    ).astype(int)

    result["_geometry_measure"] = (
        result.geometry.apply(
            geometry_measure
        )
    )

    result["_geometry_key"] = (
        result.geometry.apply(
            lambda geometry: (
                geometry.wkb_hex
            )
        )
    )

    occurrence_counts = (
        result.groupby(
            "parking_id"
        ).size()
    )

    geometry_variant_counts = (
        result.groupby(
            "parking_id"
        )["_geometry_key"]
        .nunique()
    )

    geometry_variant_id_count = int(
        (
            geometry_variant_counts > 1
        ).sum()
    )

    before_count = len(result)

    result = result.sort_values(
        by=[
            "parking_id",
            "_geometry_measure",
            "source_chunk_order",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    )

    result = result.drop_duplicates(
        subset="parking_id",
        keep="first",
    ).copy()

    duplicate_count = (
        before_count - len(result)
    )

    result[
        "source_chunk_occurrence_count"
    ] = (
        result["parking_id"]
        .map(occurrence_counts)
        .astype(int)
    )

    result[
        "geometry_variant_count"
    ] = (
        result["parking_id"]
        .map(geometry_variant_counts)
        .astype(int)
    )

    result.drop(
        columns=[
            "_geometry_measure",
            "_geometry_key",
        ],
        inplace=True,
    )

    polygon_mask = (
        result.geometry.geom_type.isin(
            {
                "Polygon",
                "MultiPolygon",
            }
        )
    )

    result["parking_area_m2"] = 0.0

    result.loc[
        polygon_mask,
        "parking_area_m2",
    ] = (
        result.loc[
            polygon_mask
        ].geometry.area
    )

    result[
        "parking_area_m2"
    ] = result[
        "parking_area_m2"
    ].round(2)

    result = result.sort_values(
        by=[
            "source_chunk_order",
            "parking_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return (
        gpd.GeoDataFrame(
            result,
            geometry="geometry",
            crs=parking.crs,
        ),
        duplicate_count,
        geometry_variant_id_count,
    )


def validate_merged_parking(
    parking: gpd.GeoDataFrame,
) -> None:
    """Validate the merged Ankara parking dataset."""

    required_columns = {
        "parking_id",
        "source_chunk_id",
        "source_chunk_order",
        "source_chunk_occurrence_count",
        "geometry_variant_count",
        "capacity_numeric",
        "parking_area_m2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(parking.columns)
    )

    if missing_columns:
        raise ValueError(
            "Merged parking is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if parking.empty:
        raise ValueError(
            "Merged Ankara parking is empty."
        )

    if parking["parking_id"].duplicated().any():
        raise ValueError(
            "Duplicate parking IDs remained."
        )

    if parking.geometry.isna().any():
        raise ValueError(
            "Missing parking geometries were found."
        )

    if parking.geometry.is_empty.any():
        raise ValueError(
            "Empty parking geometries were found."
        )

    if not parking.geometry.is_valid.all():
        raise ValueError(
            "Invalid parking geometries were found."
        )

    if (
        parking["parking_area_m2"] < 0
    ).any():
        raise ValueError(
            "Negative parking area was found."
        )

    known_capacity = pd.to_numeric(
        parking["capacity_numeric"],
        errors="coerce",
    ).dropna()

    if not np.isfinite(
        known_capacity.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite parking capacity was found."
        )

    if (
        known_capacity < 0
    ).any():
        raise ValueError(
            "Negative parking capacity was found."
        )

    if (
        parking[
            "source_chunk_occurrence_count"
        ] < 1
    ).any():
        raise ValueError(
            "Invalid source occurrence count."
        )

    print(
        "Merged Ankara parking validation "
        "completed successfully."
    )


def save_outputs(
    parking: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    paths: AnkaraParkingMergePaths,
) -> None:
    """Save merged parking and the chunk manifest."""

    for path in (
        paths.output_gpkg,
        paths.manifest_csv,
    ):
        if path.exists():
            path.unlink()

    parking.to_file(
        paths.output_gpkg,
        layer=MERGED_PARKING_LAYER_NAME,
        driver="GPKG",
    )

    manifest.to_csv(
        paths.manifest_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Merged parking saved: {paths.output_gpkg}"
    )

    print(
        f"Merge manifest saved: {paths.manifest_csv}"
    )


def create_preview(
    parking: gpd.GeoDataFrame,
    footprint: Any,
    paths: AnkaraParkingMergePaths,
) -> None:
    """Create an Ankara parking preview."""

    parking_points = parking[
        [
            "parking_id",
            "geometry",
        ]
    ].copy()

    parking_points["geometry"] = (
        parking_points
        .geometry
        .representative_point()
    )

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    gpd.GeoSeries(
        [footprint],
        crs=parking.crs,
    ).boundary.plot(
        ax=axis,
        linewidth=1.0,
    )

    parking_points.plot(
        ax=axis,
        markersize=2.5,
        alpha=0.65,
    )

    axis.set_title(
        "VoltSight - Ankara OSM Parking Features"
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
    parking: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    raw_count: int,
    filtered_count: int,
    duplicate_count: int,
    geometry_variant_id_count: int,
    paths: AnkaraParkingMergePaths,
) -> None:
    """Create the Ankara parking merge summary."""

    polygon_count = int(
        parking.geometry.geom_type.isin(
            {
                "Polygon",
                "MultiPolygon",
            }
        ).sum()
    )

    point_count = int(
        parking.geometry.geom_type.isin(
            {
                "Point",
                "MultiPoint",
            }
        ).sum()
    )

    line_count = int(
        parking.geometry.geom_type.isin(
            {
                "LineString",
                "MultiLineString",
            }
        ).sum()
    )

    known_capacity_count = int(
        parking[
            "capacity_numeric"
        ].notna().sum()
    )

    summary = f"""# Ankara Parking Dataset Summary

## Chunk Results

- Total chunks: {len(manifest):,}
- Empty-success chunks: {int(manifest["is_empty"].sum()):,}
- Non-empty chunks: {int((~manifest["is_empty"]).sum()):,}
- Raw downloaded parking records: {raw_count:,}
- Records after Ankara buffer filter: {filtered_count:,}
- Duplicate chunk occurrences removed: {duplicate_count:,}
- Parking IDs with multiple geometry variants: {geometry_variant_id_count:,}
- Final unique parking features: {len(parking):,}

## Geometry Statistics

- Point or multipoint features: {point_count:,}
- Line or multiline features: {line_count:,}
- Polygon or multipolygon features: {polygon_count:,}
- Features with known numeric capacity: {known_capacity_count:,}
- Total known capacity: {parking["capacity_numeric"].sum():,.0f}
- Total mapped polygon area: {parking["parking_area_m2"].sum():,.2f} m²
- Analysis CRS: {parking.crs}

## Generated Outputs

- `data/interim/ankara_parking_features.gpkg`
- `data/interim/ankara_parking_merge_manifest.csv`
- `docs/ankara_parking_features_preview.png`

## Method

OpenStreetMap parking records downloaded from overlapping Ankara
chunks were merged using their stable OSM-derived `parking_id`.

When one parking feature appeared in multiple chunks, the record with
the most complete geometry was retained. Ties were resolved using the
lowest source chunk order.

Only parking features intersecting Ankara and its one-kilometre
download buffer were retained. This preserves nearby parking needed
for the 500-metre and 1,000-metre accessibility calculations.

## Data Limitation

OpenStreetMap parking coverage and capacity fields can be incomplete.
The dataset represents mapped parking availability rather than a
complete official parking inventory.

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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge downloaded Ankara parking chunks."
        )
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

    parser.add_argument(
        "--skip-preview",
        action="store_true",
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(argv)

    if arguments.chunk_size_m <= 0:
        parser.error(
            "--chunk-size-m must be positive."
        )

    if arguments.download_buffer_m < 0:
        parser.error(
            "--download-buffer-m cannot be negative."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Run the Ankara parking merge pipeline."""

    paths = resolve_paths(
        arguments.chunk_size_m
    )

    create_output_directories(paths)

    print("=" * 70)
    print("VoltSight - Ankara Parking Chunk Merge")
    print("=" * 70)

    chunks = load_chunk_plan(paths)

    manifest = validate_chunk_results(
        chunks,
        paths,
    )

    footprint = load_relevant_footprint(
        paths,
        chunks.crs,
        arguments.download_buffer_m,
    )

    raw_parking = load_raw_parking(
        manifest,
        paths,
        chunks.crs,
    )

    filtered_parking = (
        filter_to_relevant_footprint(
            raw_parking,
            footprint,
        )
    )

    (
        parking,
        duplicate_count,
        geometry_variant_id_count,
    ) = deduplicate_parking_features(
        filtered_parking
    )

    validate_merged_parking(parking)

    save_outputs(
        parking,
        manifest,
        paths,
    )

    if not arguments.skip_preview:
        create_preview(
            parking,
            footprint,
            paths,
        )

    create_summary(
        parking=parking,
        manifest=manifest,
        raw_count=len(raw_parking),
        filtered_count=len(filtered_parking),
        duplicate_count=duplicate_count,
        geometry_variant_id_count=(
            geometry_variant_id_count
        ),
        paths=paths,
    )

    print("-" * 70)

    print(
        "Raw parking records: "
        f"{len(raw_parking):,}"
    )

    print(
        "Records after spatial filter: "
        f"{len(filtered_parking):,}"
    )

    print(
        "Duplicate records removed: "
        f"{duplicate_count:,}"
    )

    print(
        "Unique parking features: "
        f"{len(parking):,}"
    )

    print(
        "Known-capacity features: "
        f"{int(parking['capacity_numeric'].notna().sum()):,}"
    )

    print("=" * 70)

    print(
        "Ankara parking merge completed successfully."
    )

    print("=" * 70)

    return parking


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the CLI pipeline."""

    arguments = parse_arguments(argv)

    run_pipeline(arguments)


if __name__ == "__main__":
    main()
