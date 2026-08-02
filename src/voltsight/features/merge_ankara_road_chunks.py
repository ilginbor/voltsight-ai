from __future__ import annotations

import argparse
import json
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
from shapely import normalize, set_precision


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


CHUNK_PLAN_LAYER_NAME = "road_download_chunks"
CHUNK_ROADS_LAYER_NAME = "drive_roads_buffered"
MERGED_ROADS_LAYER_NAME = "drive_roads"

DEFAULT_CHUNK_SIZE_METERS = 8_000
GEOMETRY_PRECISION_METERS = 0.01


@dataclass(
    frozen=True,
    slots=True,
)
class AnkaraRoadMergePaths:
    """Paths used by the Ankara road merge pipeline."""

    chunk_plan_gpkg: Path
    download_directory: Path
    output_gpkg: Path
    manifest_csv: Path
    preview_png: Path
    summary_md: Path


def distance_token(
    distance_meters: int,
) -> str:
    """Create a deterministic filename token."""

    if distance_meters <= 0:
        raise ValueError(
            "Distance must be positive."
        )

    if distance_meters % 1_000 == 0:
        return (
            f"{distance_meters // 1_000}km"
        )

    return f"{distance_meters}m"


def resolve_paths(
    chunk_size_m: int = DEFAULT_CHUNK_SIZE_METERS,
) -> AnkaraRoadMergePaths:
    """Resolve deterministic Ankara merge paths."""

    token = distance_token(
        chunk_size_m
    )

    return AnkaraRoadMergePaths(
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
            / f"ankara_road_chunk_downloads_{token}"
        ),
        output_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_drive_roads.gpkg"
        ),
        manifest_csv=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "ankara_road_merge_manifest.csv"
        ),
        preview_png=(
            PROJECT_ROOT
            / "docs"
            / "ankara_road_network_preview.png"
        ),
        summary_md=(
            PROJECT_ROOT
            / "docs"
            / "ankara_road_network_summary.md"
        ),
    )


def create_output_directories(
    paths: AnkaraRoadMergePaths,
) -> None:
    """Create output directories."""

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
    paths: AnkaraRoadMergePaths,
    chunk_id: str,
) -> Path:
    """Return one chunk metadata path."""

    return (
        paths.download_directory
        / f"{chunk_id}_metadata.json"
    )


def road_output_path(
    paths: AnkaraRoadMergePaths,
    chunk_id: str,
) -> Path:
    """Return one chunk road GeoPackage path."""

    return (
        paths.download_directory
        / f"{chunk_id}_drive_roads.gpkg"
    )


def load_chunk_plan(
    paths: AnkaraRoadMergePaths,
) -> gpd.GeoDataFrame:
    """Load and validate the complete chunk plan."""

    if not paths.chunk_plan_gpkg.exists():
        raise FileNotFoundError(
            "Ankara road chunk plan was not found:\n"
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

    missing = (
        required_columns
        - set(chunks.columns)
    )

    if missing:
        raise ValueError(
            "Chunk plan is missing columns: "
            f"{sorted(missing)}"
        )

    if chunks.empty:
        raise ValueError(
            "Chunk plan is empty."
        )

    if chunks.crs is None:
        raise ValueError(
            "Chunk plan has no CRS."
        )

    if not chunks.crs.is_projected:
        raise ValueError(
            "Chunk plan must use a projected CRS."
        )

    if chunks["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

    if not chunks.geometry.is_valid.all():
        raise ValueError(
            "Invalid chunk geometries were found."
        )

    return chunks.sort_values(
        "chunk_order"
    ).reset_index(
        drop=True
    )


def read_metadata(
    path: Path,
) -> dict[str, Any]:
    """Read and validate one chunk metadata file."""

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


def validate_download_completion(
    chunks: gpd.GeoDataFrame,
    paths: AnkaraRoadMergePaths,
) -> pd.DataFrame:
    """Ensure every chunk has a successful result."""

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

        roads_path = road_output_path(
            paths,
            chunk_id,
        )

        if (
            not is_empty
            and not roads_path.exists()
        ):
            raise FileNotFoundError(
                "Successful non-empty chunk has no "
                f"road GeoPackage: {chunk_id}"
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
                "downloaded_road_edges": int(
                    metadata.get(
                        "road_edge_count",
                        0,
                    )
                ),
                "downloaded_main_road_edges": int(
                    metadata.get(
                        "main_road_edge_count",
                        0,
                    )
                ),
                "roads_path": (
                    roads_path.as_posix()
                    if not is_empty
                    else ""
                ),
            }
        )

    manifest = pd.DataFrame(
        records
    )

    print(
        f"Validated chunk results: {len(manifest):,}"
    )

    print(
        "Empty-success chunks: "
        f"{int(manifest['is_empty'].sum()):,}"
    )

    print(
        "Raw downloaded road edges: "
        f"{int(manifest['downloaded_road_edges'].sum()):,}"
    )

    return manifest


def clip_roads_to_core(
    roads: gpd.GeoDataFrame,
    core_geometry: Any,
    target_crs: Any,
    chunk_id: str,
    chunk_order: int,
) -> gpd.GeoDataFrame:
    """Clip buffered roads to one non-overlapping core chunk."""

    if roads.empty:
        return roads.copy()

    if roads.crs is None:
        raise ValueError(
            f"Road CRS is missing for {chunk_id}."
        )

    if roads.crs != target_crs:
        roads = roads.to_crs(
            target_crs
        )

    roads = roads.loc[
        roads.geometry.notna()
    ].copy()

    roads = roads.loc[
        ~roads.geometry.is_empty
    ].copy()

    roads = roads.loc[
        roads.geometry.intersects(
            core_geometry
        )
    ].copy()

    if roads.empty:
        return roads

    roads["geometry"] = (
        roads.geometry.intersection(
            core_geometry
        )
    )

    roads = roads.explode(
        index_parts=False,
        ignore_index=True,
    )

    roads = roads.loc[
        roads.geometry.notna()
    ].copy()

    roads = roads.loc[
        ~roads.geometry.is_empty
    ].copy()

    roads = roads.loc[
        roads.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()

    roads["edge_length_m"] = (
        roads.geometry.length.round(2)
    )

    roads = roads.loc[
        roads["edge_length_m"] > 0
    ].copy()

    roads["core_chunk_id"] = (
        chunk_id
    )

    roads["core_chunk_order"] = int(
        chunk_order
    )

    return gpd.GeoDataFrame(
        roads,
        geometry="geometry",
        crs=target_crs,
    )


def canonical_geometry_key(
    geometry: Any,
) -> str:
    """Create an orientation-independent geometry key."""

    precise_geometry = set_precision(
        geometry,
        grid_size=(
            GEOMETRY_PRECISION_METERS
        ),
    )

    normalized_geometry = normalize(
        precise_geometry
    )

    return normalized_geometry.wkb_hex


def deduplicate_road_pieces(
    roads: gpd.GeoDataFrame,
) -> tuple[
    gpd.GeoDataFrame,
    int,
]:
    """Remove exact boundary duplicates after core clipping."""

    if roads.empty:
        return roads.copy(), 0

    result = roads.copy()

    result["_geometry_key"] = (
        result.geometry.apply(
            canonical_geometry_key
        )
    )

    duplicate_subset = [
        "osm_id",
        "highway",
        "_geometry_key",
    ]

    before_count = len(
        result
    )

    result = result.sort_values(
        by=[
            "core_chunk_order",
            "source_chunk_order",
            "road_id",
        ],
        kind="stable",
    )

    result = result.drop_duplicates(
        subset=duplicate_subset,
        keep="first",
    ).copy()

    duplicate_count = (
        before_count - len(result)
    )

    result.drop(
        columns="_geometry_key",
        inplace=True,
    )

    result.reset_index(
        drop=True,
        inplace=True,
    )

    if "road_id" in result.columns:
        result.rename(
            columns={
                "road_id": "source_road_id",
            },
            inplace=True,
        )

    id_width = max(
        8,
        len(str(len(result))),
    )

    result.insert(
        0,
        "road_id",
        [
            (
                "ANK_ROAD_"
                f"{index:0{id_width}d}"
            )
            for index in range(
                1,
                len(result) + 1,
            )
        ],
    )

    return (
        gpd.GeoDataFrame(
            result,
            geometry="geometry",
            crs=roads.crs,
        ),
        duplicate_count,
    )


def merge_chunk_roads(
    chunks: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    paths: AnkaraRoadMergePaths,
) -> tuple[
    gpd.GeoDataFrame,
    int,
    int,
]:
    """Read, clip and merge every successful chunk."""

    metadata_by_id = (
        manifest.set_index(
            "chunk_id"
        )
        .to_dict(
            orient="index"
        )
    )

    clipped_frames: list[
        gpd.GeoDataFrame
    ] = []

    clipped_edge_count = 0

    for position, (_, chunk) in enumerate(
        chunks.iterrows(),
        start=1,
    ):
        chunk_id = str(
            chunk["chunk_id"]
        )

        record = metadata_by_id[
            chunk_id
        ]

        if record["is_empty"]:
            continue

        roads = gpd.read_file(
            road_output_path(
                paths,
                chunk_id,
            ),
            layer=CHUNK_ROADS_LAYER_NAME,
        )

        clipped = clip_roads_to_core(
            roads=roads,
            core_geometry=chunk.geometry,
            target_crs=chunks.crs,
            chunk_id=chunk_id,
            chunk_order=int(
                chunk["chunk_order"]
            ),
        )

        if not clipped.empty:
            clipped_frames.append(
                clipped
            )

            clipped_edge_count += len(
                clipped
            )

        if (
            position % 25 == 0
            or position == len(chunks)
        ):
            print(
                "Processed chunks: "
                f"{position:,}/{len(chunks):,}"
            )

    if not clipped_frames:
        raise RuntimeError(
            "No clipped Ankara roads were generated."
        )

    combined = gpd.GeoDataFrame(
        pd.concat(
            clipped_frames,
            ignore_index=True,
        ),
        geometry="geometry",
        crs=chunks.crs,
    )

    deduplicated, duplicate_count = (
        deduplicate_road_pieces(
            combined
        )
    )

    return (
        deduplicated,
        clipped_edge_count,
        duplicate_count,
    )


def validate_merged_roads(
    roads: gpd.GeoDataFrame,
) -> None:
    """Validate the final Ankara road network."""

    required_columns = {
        "road_id",
        "source_road_id",
        "source_chunk_id",
        "core_chunk_id",
        "osm_id",
        "highway",
        "is_main_road",
        "edge_length_m",
        "geometry",
    }

    missing = (
        required_columns
        - set(roads.columns)
    )

    if missing:
        raise ValueError(
            "Merged roads are missing columns: "
            f"{sorted(missing)}"
        )

    if roads.empty:
        raise ValueError(
            "Merged Ankara road network is empty."
        )

    if roads["road_id"].duplicated().any():
        raise ValueError(
            "Duplicate merged road IDs were found."
        )

    if roads.geometry.isna().any():
        raise ValueError(
            "Merged roads contain missing geometry."
        )

    if not roads.geometry.is_valid.all():
        raise ValueError(
            "Merged roads contain invalid geometry."
        )

    lengths = roads[
        "edge_length_m"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        lengths
    ).all():
        raise ValueError(
            "Road lengths contain non-finite values."
        )

    if (
        lengths <= 0
    ).any():
        raise ValueError(
            "Road lengths must be positive."
        )

    print(
        "Merged Ankara road validation "
        "completed successfully."
    )


def save_outputs(
    roads: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    paths: AnkaraRoadMergePaths,
) -> None:
    """Save the merged roads and merge manifest."""

    for path in (
        paths.output_gpkg,
        paths.manifest_csv,
    ):
        if path.exists():
            path.unlink()

    roads.to_file(
        paths.output_gpkg,
        layer=MERGED_ROADS_LAYER_NAME,
        driver="GPKG",
    )

    manifest.to_csv(
        paths.manifest_csv,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Merged roads saved: {paths.output_gpkg}"
    )

    print(
        f"Merge manifest saved: {paths.manifest_csv}"
    )


def create_preview(
    chunks: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    paths: AnkaraRoadMergePaths,
) -> None:
    """Create an Ankara road-network preview."""

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    roads.plot(
        ax=axis,
        linewidth=0.05,
        alpha=0.45,
    )

    main_roads = roads.loc[
        roads["is_main_road"]
    ]

    main_roads.plot(
        ax=axis,
        linewidth=0.18,
        alpha=0.80,
    )

    gpd.GeoSeries(
        [
            chunks.geometry.union_all()
        ],
        crs=chunks.crs,
    ).boundary.plot(
        ax=axis,
        linewidth=1.2,
    )

    axis.set_title(
        "VoltSight - Ankara Drivable Road Network"
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
        f"Road preview saved: {paths.preview_png}"
    )


def create_summary(
    roads: gpd.GeoDataFrame,
    manifest: pd.DataFrame,
    clipped_edge_count: int,
    duplicate_count: int,
    paths: AnkaraRoadMergePaths,
) -> None:
    """Create the Ankara road merge summary."""

    raw_edge_count = int(
        manifest[
            "downloaded_road_edges"
        ].sum()
    )

    total_length_km = (
        roads["edge_length_m"].sum()
        / 1_000
    )

    main_roads = roads.loc[
        roads["is_main_road"]
    ]

    main_length_km = (
        main_roads[
            "edge_length_m"
        ].sum()
        / 1_000
    )

    summary = f"""# Ankara Road Network Summary

## Chunk Results

- Total road chunks: {len(manifest):,}
- Empty-success chunks: {int(manifest["is_empty"].sum()):,}
- Raw downloaded road edges: {raw_edge_count:,}
- Road pieces after core clipping: {clipped_edge_count:,}
- Exact boundary duplicates removed: {duplicate_count:,}
- Final merged road pieces: {len(roads):,}

## Network Statistics

- Main-road pieces: {int(roads["is_main_road"].sum()):,}
- Total road length: {total_length_km:,.2f} km
- Total main-road length: {main_length_km:,.2f} km
- Analysis CRS: {roads.crs}

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

{datetime.now(timezone.utc).isoformat()}
"""

    paths.summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Road summary saved: {paths.summary_md}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the merge command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Clip and merge downloaded Ankara road chunks."
        )
    )

    parser.add_argument(
        "--chunk-size-m",
        type=int,
        default=DEFAULT_CHUNK_SIZE_METERS,
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
    """Parse command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    if arguments.chunk_size_m <= 0:
        parser.error(
            "--chunk-size-m must be positive."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Run the Ankara road merge pipeline."""

    paths = resolve_paths(
        arguments.chunk_size_m
    )

    create_output_directories(
        paths
    )

    print("=" * 70)
    print("VoltSight - Ankara Road Chunk Merge")
    print("=" * 70)

    chunks = load_chunk_plan(
        paths
    )

    manifest = validate_download_completion(
        chunks,
        paths,
    )

    (
        roads,
        clipped_edge_count,
        duplicate_count,
    ) = merge_chunk_roads(
        chunks,
        manifest,
        paths,
    )

    validate_merged_roads(
        roads
    )

    save_outputs(
        roads,
        manifest,
        paths,
    )

    if not arguments.skip_preview:
        create_preview(
            chunks,
            roads,
            paths,
        )

    create_summary(
        roads=roads,
        manifest=manifest,
        clipped_edge_count=clipped_edge_count,
        duplicate_count=duplicate_count,
        paths=paths,
    )

    print("=" * 70)

    print(
        "Ankara road merge completed successfully."
    )

    print(
        f"Final road pieces: {len(roads):,}"
    )

    print(
        "Main-road pieces: "
        f"{int(roads['is_main_road'].sum()):,}"
    )

    print(
        "Total road length: "
        f"{roads['edge_length_m'].sum() / 1_000:,.2f} km"
    )

    print("=" * 70)

    return roads


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
