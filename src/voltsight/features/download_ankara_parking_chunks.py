from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import osmnx as ox
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from voltsight.features.create_parking_features import (  # noqa: E402
    MAX_QUERY_AREA_SIZE_M2,
    OVERPASS_REQUEST_TIMEOUT_SECONDS,
    download_parking_features,
    prepare_parking_features,
)


DEFAULT_CHUNK_SIZE_METERS = 8_000
DEFAULT_DOWNLOAD_BUFFER_METERS = 1_000

CHUNK_PLAN_LAYER_NAME = "road_download_chunks"
PARKING_LAYER_NAME = "parking_features_buffered"


@dataclass(
    frozen=True,
    slots=True,
)
class AnkaraParkingDownloadPaths:
    """Filesystem paths for Ankara parking chunk downloads."""

    chunk_plan_gpkg: Path
    output_directory: Path
    cache_root: Path


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
) -> AnkaraParkingDownloadPaths:
    """Resolve Ankara parking chunk input and output paths."""

    token = distance_token(
        chunk_size_m
    )

    return AnkaraParkingDownloadPaths(
        chunk_plan_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / f"ankara_road_download_chunks_{token}.gpkg"
        ),
        output_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / f"ankara_parking_chunk_downloads_{token}"
        ),
        cache_root=(
            PROJECT_ROOT
            / "cache"
            / "ankara"
            / "parking_chunks"
            / token
        ),
    )


def parking_output_path(
    paths: AnkaraParkingDownloadPaths,
    chunk_id: str,
) -> Path:
    """Return the GeoPackage path for one parking chunk."""

    return (
        paths.output_directory
        / f"{chunk_id}_parking.gpkg"
    )


def metadata_output_path(
    paths: AnkaraParkingDownloadPaths,
    chunk_id: str,
) -> Path:
    """Return the metadata path for one parking chunk."""

    return (
        paths.output_directory
        / f"{chunk_id}_metadata.json"
    )


def create_output_directories(
    paths: AnkaraParkingDownloadPaths,
) -> None:
    """Create output and cache directories."""

    paths.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_chunk_plan(
    paths: AnkaraParkingDownloadPaths,
) -> gpd.GeoDataFrame:
    """Load and validate the existing Ankara chunk plan."""

    if not paths.chunk_plan_gpkg.exists():
        raise FileNotFoundError(
            "Ankara chunk plan was not found.\n"
            "Run create_ankara_road_chunk_plan.py first.\n"
            f"Missing file: {paths.chunk_plan_gpkg}"
        )

    chunks = gpd.read_file(
        paths.chunk_plan_gpkg,
        layer=CHUNK_PLAN_LAYER_NAME,
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
            "The chunk plan must use a projected CRS."
        )

    required_columns = {
        "chunk_id",
        "chunk_order",
        "grid_cell_count",
        "download_buffer_m",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(chunks.columns)
    )

    if missing_columns:
        raise ValueError(
            "The Ankara chunk plan is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if chunks["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate chunk identifiers were found."
        )

    if chunks.geometry.isna().any():
        raise ValueError(
            "The chunk plan contains missing geometry."
        )

    if not chunks.geometry.is_valid.all():
        raise ValueError(
            "The chunk plan contains invalid geometry."
        )

    chunks = chunks.sort_values(
        "chunk_order"
    ).reset_index(
        drop=True
    )

    print(
        f"Loaded Ankara chunks: {len(chunks):,}"
    )

    return chunks


def select_chunks(
    chunks: gpd.GeoDataFrame,
    *,
    chunk_ids: Sequence[str] | None,
    download_all: bool,
    start_order: int,
    limit: int | None,
) -> gpd.GeoDataFrame:
    """Select explicit chunks or an ordered resumable range."""

    if start_order <= 0:
        raise ValueError(
            "Start order must be positive."
        )

    if limit is not None and limit <= 0:
        raise ValueError(
            "Limit must be positive."
        )

    selected = chunks.copy()

    if chunk_ids:
        requested_ids = list(
            dict.fromkeys(
                chunk_ids
            )
        )

        available_ids = set(
            selected["chunk_id"]
            .astype(str)
            .tolist()
        )

        missing_ids = [
            chunk_id
            for chunk_id in requested_ids
            if chunk_id not in available_ids
        ]

        if missing_ids:
            raise ValueError(
                "Unknown chunk identifiers: "
                f"{missing_ids}"
            )

        selected = selected.loc[
            selected["chunk_id"].isin(
                requested_ids
            )
        ].copy()

    elif download_all:
        selected = selected.loc[
            selected["chunk_order"]
            >= start_order
        ].copy()

        if limit is not None:
            selected = selected.head(
                limit
            ).copy()

    else:
        raise ValueError(
            "Provide at least one --chunk-id or use --all."
        )

    selected = selected.sort_values(
        "chunk_order"
    ).reset_index(
        drop=True
    )

    if selected.empty:
        raise ValueError(
            "No chunks matched the requested selection."
        )

    return selected


def configure_osmnx_for_chunk(
    cache_directory: Path,
) -> None:
    """Configure OSMnx for one resumable parking chunk."""

    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(
        cache_directory
    )
    ox.settings.log_console = True

    ox.settings.requests_timeout = (
        OVERPASS_REQUEST_TIMEOUT_SECONDS
    )

    ox.settings.max_query_area_size = (
        MAX_QUERY_AREA_SIZE_M2
    )

    ox.settings.overpass_rate_limit = False

    ox.settings.http_user_agent = (
        "VoltSight/0.2 "
        "educational-geospatial-research-project"
    )

    ox.settings.http_referer = (
        "https://github.com/ilginbor/voltsight-ai"
    )


def create_download_polygon(
    core_geometry: Any,
    projected_crs: Any,
    download_buffer_m: int,
) -> tuple[Any, float]:
    """Buffer one core chunk and convert it to EPSG:4326."""

    if download_buffer_m < 0:
        raise ValueError(
            "Download buffer cannot be negative."
        )

    if core_geometry is None:
        raise ValueError(
            "Chunk geometry is missing."
        )

    if core_geometry.is_empty:
        raise ValueError(
            "Chunk geometry is empty."
        )

    buffered_geometry = core_geometry.buffer(
        download_buffer_m
    )

    buffered_area_m2 = float(
        buffered_geometry.area
    )

    if buffered_area_m2 <= 0:
        raise ValueError(
            "Buffered geometry has no area."
        )

    if buffered_area_m2 > MAX_QUERY_AREA_SIZE_M2:
        raise ValueError(
            "Buffered parking chunk exceeds the "
            "configured Overpass query limit. "
            f"Area: {buffered_area_m2 / 1_000_000:,.2f} km², "
            f"limit: {MAX_QUERY_AREA_SIZE_M2 / 1_000_000:,.2f} km²."
        )

    buffered_frame = gpd.GeoDataFrame(
        geometry=[
            buffered_geometry
        ],
        crs=projected_crs,
    )

    buffered_wgs84 = buffered_frame.to_crs(
        epsg=4326
    )

    return (
        buffered_wgs84.geometry.iloc[0],
        buffered_area_m2,
    )


def prepare_chunk_parking(
    raw_parking: gpd.GeoDataFrame,
    target_crs: Any,
    chunk_id: str,
    chunk_order: int,
    download_buffer_m: int,
) -> gpd.GeoDataFrame:
    """Clean one chunk and add source-chunk metadata."""

    parking = prepare_parking_features(
        raw_parking,
        target_crs,
    )

    parking.insert(
        0,
        "source_chunk_id",
        chunk_id,
    )

    parking.insert(
        1,
        "source_chunk_order",
        int(chunk_order),
    )

    parking.insert(
        2,
        "download_buffer_m",
        int(download_buffer_m),
    )

    if parking["parking_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate parking IDs remained in {chunk_id}."
        )

    return parking


def save_chunk_parking(
    parking: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save one cleaned parking chunk."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    parking.to_file(
        output_path,
        layer=PARKING_LAYER_NAME,
        driver="GPKG",
    )

    print(
        f"Parking chunk saved: {output_path}"
    )


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def write_metadata(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write chunk metadata."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def read_metadata(
    path: Path,
) -> dict[str, Any] | None:
    """Read valid chunk metadata when available."""

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


def chunk_is_complete(
    output_path: Path,
    metadata_path: Path,
) -> bool:
    """Return whether a reusable parking result exists."""

    metadata = read_metadata(
        metadata_path
    )

    if not metadata:
        return False

    if metadata.get("status") != "success":
        return False

    if (
        metadata.get("is_empty") is True
        and int(
            metadata.get(
                "parking_feature_count",
                -1,
            )
        ) == 0
    ):
        return True

    if not output_path.exists():
        return False

    if output_path.stat().st_size <= 0:
        return False

    return (
        metadata.get("output_path")
        == output_path.as_posix()
    )


EMPTY_PARKING_RESPONSE_MARKERS = (
    "No data elements in server response",
    "No matching features",
    "The query returned no parking features",
)


def is_confirmed_empty_parking_response(
    error: Exception,
) -> bool:
    """
    Detect a parking-free chunk confirmed by multiple endpoints.

    A single empty response combined with network failures remains
    retryable. At least two no-data endpoint results are required.
    """

    error_text = str(
        error
    )

    empty_response_count = sum(
        error_text.count(
            marker
        )
        for marker in EMPTY_PARKING_RESPONSE_MARKERS
    )

    return empty_response_count >= 2


def process_chunk(
    chunk: pd.Series,
    chunks_crs: Any,
    paths: AnkaraParkingDownloadPaths,
    *,
    force: bool,
    download_buffer_override_m: int | None,
) -> str:
    """Download and save one Ankara parking chunk."""

    chunk_id = str(
        chunk["chunk_id"]
    )

    chunk_order = int(
        chunk["chunk_order"]
    )

    grid_cell_count = int(
        chunk["grid_cell_count"]
    )

    configured_buffer_m = int(
        chunk["download_buffer_m"]
    )

    download_buffer_m = (
        int(download_buffer_override_m)
        if download_buffer_override_m is not None
        else configured_buffer_m
    )

    output_path = parking_output_path(
        paths,
        chunk_id,
    )

    metadata_path = metadata_output_path(
        paths,
        chunk_id,
    )

    if (
        not force
        and chunk_is_complete(
            output_path,
            metadata_path,
        )
    ):
        print(
            f"[SKIP] {chunk_id} already completed."
        )

        return "skipped"

    started_at = utc_now()

    print("=" * 70)

    print(
        f"Processing parking chunk: {chunk_id}"
    )

    print(
        f"Chunk order: {chunk_order:,}"
    )

    print(
        f"Assigned grid cells: {grid_cell_count:,}"
    )

    print(
        f"Download buffer: {download_buffer_m:,} m"
    )

    cache_directory = (
        paths.cache_root
        / chunk_id
    )

    try:
        configure_osmnx_for_chunk(
            cache_directory
        )

        (
            download_polygon,
            buffered_area_m2,
        ) = create_download_polygon(
            core_geometry=chunk.geometry,
            projected_crs=chunks_crs,
            download_buffer_m=download_buffer_m,
        )

        print(
            "Buffered download area: "
            f"{buffered_area_m2 / 1_000_000:,.2f} km²"
        )

        raw_parking = download_parking_features(
            download_polygon
        )

        parking = prepare_chunk_parking(
            raw_parking=raw_parking,
            target_crs=chunks_crs,
            chunk_id=chunk_id,
            chunk_order=chunk_order,
            download_buffer_m=download_buffer_m,
        )

        save_chunk_parking(
            parking,
            output_path,
        )

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

        metadata = {
            "chunk_id": chunk_id,
            "chunk_order": chunk_order,
            "grid_cell_count": grid_cell_count,
            "status": "success",
            "is_empty": False,
            "download_buffer_m": download_buffer_m,
            "buffered_download_area_km2": round(
                buffered_area_m2
                / 1_000_000.0,
                4,
            ),
            "parking_feature_count": int(
                len(parking)
            ),
            "polygon_feature_count": polygon_count,
            "point_feature_count": point_count,
            "known_capacity_count": int(
                parking[
                    "capacity_numeric"
                ].notna().sum()
            ),
            "output_path": output_path.as_posix(),
            "cache_directory": cache_directory.as_posix(),
            "overpass_endpoint": str(
                ox.settings.overpass_url
            ),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "error": None,
        }

        write_metadata(
            metadata_path,
            metadata,
        )

        print(
            f"[SUCCESS] {chunk_id}: "
            f"{len(parking):,} parking features"
        )

        return "success"

    except Exception as error:
        if is_confirmed_empty_parking_response(
            error
        ):
            if output_path.exists():
                output_path.unlink()

            metadata = {
                "chunk_id": chunk_id,
                "chunk_order": chunk_order,
                "grid_cell_count": grid_cell_count,
                "status": "success",
                "is_empty": True,
                "download_buffer_m": download_buffer_m,
                "parking_feature_count": 0,
                "polygon_feature_count": 0,
                "point_feature_count": 0,
                "known_capacity_count": 0,
                "output_path": None,
                "cache_directory": cache_directory.as_posix(),
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "error": None,
                "empty_reason": (
                    "Multiple Overpass endpoints returned "
                    "no matching amenity=parking features."
                ),
            }

            write_metadata(
                metadata_path,
                metadata,
            )

            print(
                f"[EMPTY] {chunk_id}: "
                "no mapped parking features"
            )

            return "success"

        metadata = {
            "chunk_id": chunk_id,
            "chunk_order": chunk_order,
            "grid_cell_count": grid_cell_count,
            "status": "failed",
            "is_empty": False,
            "download_buffer_m": download_buffer_m,
            "output_path": output_path.as_posix(),
            "cache_directory": cache_directory.as_posix(),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }

        write_metadata(
            metadata_path,
            metadata,
        )

        print(
            f"[FAILED] {chunk_id}: "
            f"{type(error).__name__}: {error}"
        )

        return "failed"


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Ankara parking chunk downloader CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Download Ankara amenity=parking data "
            "as resumable chunks."
        )
    )

    selection = parser.add_mutually_exclusive_group(
        required=True
    )

    selection.add_argument(
        "--chunk-id",
        action="append",
        dest="chunk_ids",
        help=(
            "Download one chunk ID. "
            "May be provided multiple times."
        ),
    )

    selection.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help=(
            "Download all chunks."
        ),
    )

    parser.add_argument(
        "--chunk-size-m",
        type=int,
        default=DEFAULT_CHUNK_SIZE_METERS,
    )

    parser.add_argument(
        "--start-order",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--download-buffer-m",
        type=int,
        default=None,
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

    if arguments.chunk_size_m <= 0:
        parser.error(
            "--chunk-size-m must be positive."
        )

    if arguments.start_order <= 0:
        parser.error(
            "--start-order must be positive."
        )

    if (
        arguments.limit is not None
        and arguments.limit <= 0
    ):
        parser.error(
            "--limit must be positive."
        )

    if (
        arguments.download_buffer_m is not None
        and arguments.download_buffer_m < 0
    ):
        parser.error(
            "--download-buffer-m cannot be negative."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> None:
    """Run selected Ankara parking chunk downloads."""

    paths = resolve_paths(
        arguments.chunk_size_m
    )

    create_output_directories(
        paths
    )

    chunks = load_chunk_plan(
        paths
    )

    selected = select_chunks(
        chunks,
        chunk_ids=arguments.chunk_ids,
        download_all=arguments.download_all,
        start_order=arguments.start_order,
        limit=arguments.limit,
    )

    print("=" * 70)

    print(
        "VoltSight - Ankara Parking Chunk Downloader"
    )

    print("=" * 70)

    print(
        f"Selected chunks: {len(selected):,}"
    )

    print(
        f"First chunk: {selected.iloc[0]['chunk_id']}"
    )

    print(
        f"Last chunk: {selected.iloc[-1]['chunk_id']}"
    )

    result_counts = {
        "success": 0,
        "skipped": 0,
        "failed": 0,
    }

    for _, chunk in selected.iterrows():
        result = process_chunk(
            chunk,
            chunks_crs=chunks.crs,
            paths=paths,
            force=arguments.force,
            download_buffer_override_m=(
                arguments.download_buffer_m
            ),
        )

        result_counts[result] += 1

    print("=" * 70)

    print(
        "Ankara parking chunk download summary"
    )

    print(
        f"Successful: {result_counts['success']:,}"
    )

    print(
        f"Skipped: {result_counts['skipped']:,}"
    )

    print(
        f"Failed: {result_counts['failed']:,}"
    )

    print("=" * 70)

    if result_counts["failed"] > 0:
        raise RuntimeError(
            "One or more parking chunks failed. "
            "Successful chunks remain saved and "
            "will be skipped on the next run."
        )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the command-line parking downloader."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
