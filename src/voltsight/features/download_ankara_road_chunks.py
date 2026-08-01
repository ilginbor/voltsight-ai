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

from voltsight.features.create_road_features import (  # noqa: E402
    MAX_QUERY_AREA_SIZE_M2,
    NETWORK_TYPE,
    OVERPASS_REQUEST_TIMEOUT_SECONDS,
    download_road_network,
    prepare_road_edges,
)


DEFAULT_CHUNK_SIZE_METERS = 8_000
DEFAULT_DOWNLOAD_BUFFER_METERS = 1_000

CHUNK_PLAN_LAYER_NAME = "road_download_chunks"
ROADS_LAYER_NAME = "drive_roads_buffered"


@dataclass(
    frozen=True,
    slots=True,
)
class AnkaraRoadDownloadPaths:
    """Filesystem paths for chunk-based Ankara road downloads."""

    chunk_plan_gpkg: Path
    output_directory: Path
    cache_root: Path


def resolve_paths(
    chunk_size_m: int = DEFAULT_CHUNK_SIZE_METERS,
) -> AnkaraRoadDownloadPaths:
    """Resolve deterministic chunk input and output paths."""

    if chunk_size_m <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    if chunk_size_m % 1_000 == 0:
        chunk_token = (
            f"{chunk_size_m // 1_000}km"
        )
    else:
        chunk_token = (
            f"{chunk_size_m}m"
        )

    return AnkaraRoadDownloadPaths(
        chunk_plan_gpkg=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / (
                "ankara_road_download_chunks_"
                f"{chunk_token}.gpkg"
            )
        ),
        output_directory=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / (
                "ankara_road_chunk_downloads_"
                f"{chunk_token}"
            )
        ),
        cache_root=(
            PROJECT_ROOT
            / "cache"
            / "ankara"
            / "road_chunks"
            / chunk_token
        ),
    )


def chunk_output_path(
    paths: AnkaraRoadDownloadPaths,
    chunk_id: str,
) -> Path:
    """Return the GeoPackage path for one downloaded chunk."""

    return (
        paths.output_directory
        / f"{chunk_id}_drive_roads.gpkg"
    )


def chunk_metadata_path(
    paths: AnkaraRoadDownloadPaths,
    chunk_id: str,
) -> Path:
    """Return the metadata path for one downloaded chunk."""

    return (
        paths.output_directory
        / f"{chunk_id}_metadata.json"
    )


def create_output_directories(
    paths: AnkaraRoadDownloadPaths,
) -> None:
    """Create chunk output and cache directories."""

    paths.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_chunk_plan(
    paths: AnkaraRoadDownloadPaths,
) -> gpd.GeoDataFrame:
    """Load and validate the Ankara road chunk plan."""

    if not paths.chunk_plan_gpkg.exists():
        raise FileNotFoundError(
            "Ankara road chunk plan was not found.\n"
            "Run create_ankara_road_chunk_plan.py first.\n"
            f"Missing file: {paths.chunk_plan_gpkg}"
        )

    chunks = gpd.read_file(
        paths.chunk_plan_gpkg,
        layer=CHUNK_PLAN_LAYER_NAME,
    )

    if chunks.empty:
        raise ValueError(
            "The Ankara road chunk plan is empty."
        )

    if chunks.crs is None:
        raise ValueError(
            "The Ankara road chunk plan has no CRS."
        )

    if not chunks.crs.is_projected:
        raise ValueError(
            "The road chunk plan must use a projected CRS."
        )

    required_columns = {
        "chunk_id",
        "chunk_order",
        "grid_cell_count",
        "core_area_km2",
        "download_buffer_m",
        "estimated_download_area_km2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(chunks.columns)
    )

    if missing_columns:
        raise ValueError(
            "The road chunk plan is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if chunks["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate chunk identifiers were found."
        )

    if chunks.geometry.isna().any():
        raise ValueError(
            "The road chunk plan contains missing geometry."
        )

    if not chunks.geometry.is_valid.all():
        raise ValueError(
            "The road chunk plan contains invalid geometry."
        )

    chunks = chunks.sort_values(
        "chunk_order"
    ).reset_index(
        drop=True
    )

    print(
        f"Loaded road download chunks: {len(chunks):,}"
    )

    print(
        "Chunks containing Ankara grid cells: "
        f"{int((chunks['grid_cell_count'] > 0).sum()):,}"
    )

    return chunks


def select_chunks(
    chunks: gpd.GeoDataFrame,
    *,
    chunk_ids: Sequence[str] | None,
    download_all: bool,
    start_order: int,
    limit: int | None,
    include_zero_grid: bool,
) -> gpd.GeoDataFrame:
    """Select explicitly requested chunks or a resumable range."""

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

        if not include_zero_grid:
            selected = selected.loc[
                selected["grid_cell_count"]
                > 0
            ].copy()

        if limit is not None:
            selected = selected.head(
                limit
            ).copy()

    else:
        raise ValueError(
            "Select at least one --chunk-id or use --all."
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
    """Configure OSMnx for one resumable road chunk."""

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
    """Buffer one core chunk and convert it to WGS84."""

    if download_buffer_m < 0:
        raise ValueError(
            "Download buffer cannot be negative."
        )

    if core_geometry is None:
        raise ValueError(
            "Chunk core geometry is missing."
        )

    if core_geometry.is_empty:
        raise ValueError(
            "Chunk core geometry is empty."
        )

    buffered_geometry = core_geometry.buffer(
        download_buffer_m
    )

    buffered_area_m2 = float(
        buffered_geometry.area
    )

    if buffered_area_m2 <= 0:
        raise ValueError(
            "Buffered download geometry has no area."
        )

    if buffered_area_m2 > MAX_QUERY_AREA_SIZE_M2:
        raise ValueError(
            "Buffered chunk exceeds the configured "
            "Overpass query limit. "
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


def prepare_chunk_roads(
    road_graph: Any,
    chunk_id: str,
    chunk_order: int,
    download_buffer_m: int,
) -> gpd.GeoDataFrame:
    """Prepare road edges with globally unique chunk identifiers."""

    roads = prepare_road_edges(
        road_graph
    )

    if roads.empty:
        raise RuntimeError(
            f"No road edges were prepared for {chunk_id}."
        )

    roads["road_id"] = [
        (
            f"{chunk_id}_"
            f"ROAD_{index:06d}"
        )
        for index in range(
            1,
            len(roads) + 1,
        )
    ]

    roads.insert(
        0,
        "source_chunk_id",
        chunk_id,
    )

    roads.insert(
        1,
        "source_chunk_order",
        int(chunk_order),
    )

    roads.insert(
        2,
        "download_buffer_m",
        int(download_buffer_m),
    )

    if roads["road_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate road IDs were generated for {chunk_id}."
        )

    return roads


def save_chunk_roads(
    roads: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save one buffered chunk road network."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    roads.to_file(
        output_path,
        layer=ROADS_LAYER_NAME,
        driver="GPKG",
    )

    print(
        f"Chunk road output saved: {output_path}"
    )


def write_metadata(
    metadata_path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write one chunk metadata record."""

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = metadata_path.with_suffix(
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
        metadata_path
    )


def read_metadata(
    metadata_path: Path,
) -> dict[str, Any] | None:
    """Read chunk metadata when it exists and is valid JSON."""

    if not metadata_path.exists():
        return None

    try:
        return json.loads(
            metadata_path.read_text(
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
    """Return whether a chunk has a successful reusable result."""

    metadata = read_metadata(
        metadata_path
    )

    if not metadata:
        return False

    if metadata.get("status") != "success":
        return False

    if (
        metadata.get("is_empty") is True
        and int(metadata.get("road_edge_count", -1)) == 0
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


EMPTY_ROAD_RESPONSE_MARKERS = (
    (
        "InsufficientResponseError: "
        "No data elements in server response"
    ),
    (
        "ValueError: Found no graph nodes "
        "within the requested polygon"
    ),
)


def is_confirmed_empty_road_response(
    error: Exception,
) -> bool:
    """
    Detect a road-free chunk confirmed by multiple endpoints.

    OSMnx can report a genuinely empty road query in two stages:

    - the Overpass response contains no matching data elements;
    - data exists around the query, but no graph nodes remain inside
      the requested polygon.

    At least two matching endpoint results are required. A single
    empty response combined with connection failures remains retryable.
    """

    error_text = str(
        error
    )

    empty_response_count = sum(
        error_text.count(
            marker
        )
        for marker in EMPTY_ROAD_RESPONSE_MARKERS
    )

    return empty_response_count >= 2


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def process_chunk(
    chunk: pd.Series,
    chunks_crs: Any,
    paths: AnkaraRoadDownloadPaths,
    *,
    force: bool,
    download_buffer_override_m: int | None,
) -> str:
    """Download, prepare and save one Ankara road chunk."""

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
        download_buffer_override_m
        if download_buffer_override_m is not None
        else configured_buffer_m
    )

    output_path = chunk_output_path(
        paths,
        chunk_id,
    )

    metadata_path = chunk_metadata_path(
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
        f"Processing chunk: {chunk_id}"
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

        road_graph = download_road_network(
            download_polygon,
            chunks_crs,
        )

        roads = prepare_chunk_roads(
            road_graph=road_graph,
            chunk_id=chunk_id,
            chunk_order=chunk_order,
            download_buffer_m=download_buffer_m,
        )

        save_chunk_roads(
            roads,
            output_path,
        )

        metadata = {
            "chunk_id": chunk_id,
            "chunk_order": chunk_order,
            "grid_cell_count": grid_cell_count,
            "status": "success",
            "network_type": NETWORK_TYPE,
            "download_buffer_m": download_buffer_m,
            "buffered_download_area_km2": round(
                buffered_area_m2
                / 1_000_000.0,
                4,
            ),
            "is_empty": False,
            "road_edge_count": int(
                len(roads)
            ),
            "main_road_edge_count": int(
                roads["is_main_road"].sum()
            ),
            "overpass_endpoint": str(
                ox.settings.overpass_url
            ),
            "output_path": output_path.as_posix(),
            "cache_directory": cache_directory.as_posix(),
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
            f"{len(roads):,} road edges"
        )

        return "success"

    except Exception as error:
        if is_confirmed_empty_road_response(
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
                "network_type": NETWORK_TYPE,
                "download_buffer_m": download_buffer_m,
                "road_edge_count": 0,
                "main_road_edge_count": 0,
                "output_path": None,
                "cache_directory": cache_directory.as_posix(),
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "error": None,
                "empty_reason": (
                    "Multiple Overpass endpoints returned "
                    "no matching drivable-road elements."
                ),
            }

            write_metadata(
                metadata_path,
                metadata,
            )

            print(
                f"[EMPTY] {chunk_id}: "
                "no matching drivable roads"
            )

            return "success"

        metadata = {
            "chunk_id": chunk_id,
            "chunk_order": chunk_order,
            "grid_cell_count": grid_cell_count,
            "status": "failed",
            "is_empty": False,
            "network_type": NETWORK_TYPE,
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
    """Build the Ankara road chunk downloader CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Download Ankara road data as resumable "
            "OpenStreetMap chunks."
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
            "Download all selected chunks."
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
        help=(
            "First chunk order used with --all."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of chunks used with --all."
        ),
    )

    parser.add_argument(
        "--include-zero-grid",
        action="store_true",
        help=(
            "Include boundary chunks containing no grid centroids."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Download again even when a completed output exists."
        ),
    )

    parser.add_argument(
        "--download-buffer-m",
        type=int,
        default=None,
        help=(
            "Override the buffer stored in the chunk plan."
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
    """Run selected resumable Ankara road downloads."""

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
        include_zero_grid=(
            arguments.include_zero_grid
        ),
    )

    print("=" * 70)

    print(
        "VoltSight - Ankara Road Chunk Downloader"
    )

    print("=" * 70)

    print(
        f"Selected chunks: {len(selected):,}"
    )

    print(
        "First chunk: "
        f"{selected.iloc[0]['chunk_id']}"
    )

    print(
        "Last chunk: "
        f"{selected.iloc[-1]['chunk_id']}"
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
        "Ankara road chunk download summary"
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
            "One or more road chunks failed. "
            "Successful chunks remain saved and "
            "will be skipped on the next run."
        )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the Ankara road chunk downloader."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
