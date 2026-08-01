from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from voltsight.features.download_ankara_road_chunks import (
    AnkaraRoadDownloadPaths,
    chunk_is_complete,
    chunk_metadata_path,
    chunk_output_path,
    create_download_polygon,
    is_confirmed_empty_road_response,
    parse_arguments,
    resolve_paths,
    select_chunks,
)


def create_synthetic_chunks() -> gpd.GeoDataFrame:
    """Create a small deterministic chunk plan."""

    return gpd.GeoDataFrame(
        {
            "chunk_id": [
                "ANK_ROAD_0001",
                "ANK_ROAD_0002",
                "ANK_ROAD_0003",
            ],
            "chunk_order": [
                1,
                2,
                3,
            ],
            "grid_cell_count": [
                10,
                0,
                8,
            ],
            "core_area_km2": [
                64.0,
                12.0,
                64.0,
            ],
            "download_buffer_m": [
                1_000,
                1_000,
                1_000,
            ],
            "estimated_download_area_km2": [
                99.14,
                30.0,
                99.14,
            ],
        },
        geometry=[
            box(0, 0, 8_000, 8_000),
            box(8_000, 0, 10_000, 6_000),
            box(16_000, 0, 24_000, 8_000),
        ],
        crs="EPSG:32636",
    )


def test_default_pilot_arguments() -> None:
    """An explicit chunk ID must be accepted."""

    arguments = parse_arguments(
        [
            "--chunk-id",
            "ANK_ROAD_0319",
        ]
    )

    assert arguments.chunk_ids == [
        "ANK_ROAD_0319"
    ]

    assert not arguments.download_all
    assert arguments.chunk_size_m == 8_000
    assert not arguments.force


def test_resolve_paths_uses_8km_plan() -> None:
    """Default paths must point to the 8-km plan."""

    paths = resolve_paths()

    assert paths.chunk_plan_gpkg.name == (
        "ankara_road_download_chunks_8km.gpkg"
    )

    assert paths.output_directory.name == (
        "ankara_road_chunk_downloads_8km"
    )


def test_chunk_output_names_are_deterministic(
    tmp_path: Path,
) -> None:
    """One chunk must have stable output names."""

    paths = AnkaraRoadDownloadPaths(
        chunk_plan_gpkg=(
            tmp_path / "chunks.gpkg"
        ),
        output_directory=(
            tmp_path / "outputs"
        ),
        cache_root=(
            tmp_path / "cache"
        ),
    )

    assert chunk_output_path(
        paths,
        "ANK_ROAD_0319",
    ).name == (
        "ANK_ROAD_0319_drive_roads.gpkg"
    )

    assert chunk_metadata_path(
        paths,
        "ANK_ROAD_0319",
    ).name == (
        "ANK_ROAD_0319_metadata.json"
    )


def test_select_single_chunk() -> None:
    """An explicitly requested chunk must be selected."""

    chunks = create_synthetic_chunks()

    selected = select_chunks(
        chunks,
        chunk_ids=[
            "ANK_ROAD_0003"
        ],
        download_all=False,
        start_order=1,
        limit=None,
        include_zero_grid=False,
    )

    assert selected[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0003"
    ]


def test_unknown_chunk_is_rejected() -> None:
    """Unknown chunk identifiers must fail clearly."""

    chunks = create_synthetic_chunks()

    with pytest.raises(
        ValueError,
        match="Unknown chunk identifiers",
    ):
        select_chunks(
            chunks,
            chunk_ids=[
                "ANK_ROAD_9999"
            ],
            download_all=False,
            start_order=1,
            limit=None,
            include_zero_grid=False,
        )


def test_all_selection_skips_zero_grid_chunks() -> None:
    """Zero-grid boundary chunks are skipped by default."""

    chunks = create_synthetic_chunks()

    selected = select_chunks(
        chunks,
        chunk_ids=None,
        download_all=True,
        start_order=1,
        limit=None,
        include_zero_grid=False,
    )

    assert selected[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0001",
        "ANK_ROAD_0003",
    ]


def test_all_selection_supports_resume_and_limit() -> None:
    """All mode must support order-based resume and batching."""

    chunks = create_synthetic_chunks()

    selected = select_chunks(
        chunks,
        chunk_ids=None,
        download_all=True,
        start_order=2,
        limit=1,
        include_zero_grid=True,
    )

    assert selected[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0002"
    ]


def test_download_polygon_stays_under_query_limit() -> None:
    """An 8-km core and 1-km buffer must remain under 100 km²."""

    polygon_wgs84, area_m2 = (
        create_download_polygon(
            core_geometry=box(
                500_000,
                4_400_000,
                508_000,
                4_408_000,
            ),
            projected_crs="EPSG:32636",
            download_buffer_m=1_000,
        )
    )

    assert not polygon_wgs84.is_empty
    assert area_m2 < 100_000_000
    assert area_m2 > 64_000_000


def test_completed_output_requires_success_metadata(
    tmp_path: Path,
) -> None:
    """Resuming requires both an output and successful metadata."""

    output_path = (
        tmp_path / "roads.gpkg"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    output_path.write_bytes(
        b"placeholder"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "status": "success",
                "output_path": (
                    output_path.as_posix()
                ),
            }
        ),
        encoding="utf-8",
    )

    assert chunk_is_complete(
        output_path,
        metadata_path,
    )



def test_confirmed_empty_response_is_detected() -> None:
    """Two independent no-data responses confirm an empty chunk."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: InsufficientResponseError: "
        "No data elements in server response.\n"
        "- endpoint-b: InsufficientResponseError: "
        "No data elements in server response."
    )

    assert is_confirmed_empty_road_response(
        error
    )


def test_single_empty_response_is_not_enough() -> None:
    """One empty response plus network failures remains retryable."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: ConnectionError\n"
        "- endpoint-b: InsufficientResponseError: "
        "No data elements in server response."
    )

    assert not is_confirmed_empty_road_response(
        error
    )


def test_empty_success_metadata_needs_no_geopackage(
    tmp_path: Path,
) -> None:
    """A confirmed road-free chunk is reusable without a GIS file."""

    output_path = (
        tmp_path / "missing_roads.gpkg"
    )

    metadata_path = (
        tmp_path / "empty_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "status": "success",
                "is_empty": True,
                "road_edge_count": 0,
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    assert chunk_is_complete(
        output_path,
        metadata_path,
    )



def test_two_no_graph_node_responses_confirm_empty_chunk() -> None:
    """Two no-node endpoint results confirm a road-free chunk."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: ValueError: Found no graph nodes "
        "within the requested polygon.\n"
        "- endpoint-b: ValueError: Found no graph nodes "
        "within the requested polygon."
    )

    assert is_confirmed_empty_road_response(
        error
    )


def test_mixed_empty_response_types_confirm_empty_chunk() -> None:
    """Different OSMnx no-data messages may confirm the same result."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: InsufficientResponseError: "
        "No data elements in server response.\n"
        "- endpoint-b: ValueError: Found no graph nodes "
        "within the requested polygon."
    )

    assert is_confirmed_empty_road_response(
        error
    )
