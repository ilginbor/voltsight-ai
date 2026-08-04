from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from voltsight.features.download_ankara_parking_chunks import (
    AnkaraParkingDownloadPaths,
    chunk_is_complete,
    create_download_polygon,
    is_confirmed_empty_parking_response,
    metadata_output_path,
    parse_arguments,
    parking_output_path,
    resolve_paths,
    select_chunks,
)


def create_synthetic_chunks() -> gpd.GeoDataFrame:
    """Create three synthetic Ankara chunks."""

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
            "download_buffer_m": [
                1_000,
                1_000,
                1_000,
            ],
        },
        geometry=[
            box(0, 0, 8_000, 8_000),
            box(8_000, 0, 16_000, 8_000),
            box(16_000, 0, 24_000, 8_000),
        ],
        crs="EPSG:32636",
    )


def test_explicit_chunk_arguments() -> None:
    """An explicit pilot chunk must be accepted."""

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


def test_default_paths_use_8km_chunks() -> None:
    """Default output paths must use the 8-km plan."""

    paths = resolve_paths()

    assert paths.chunk_plan_gpkg.name == (
        "ankara_road_download_chunks_8km.gpkg"
    )

    assert paths.output_directory.name == (
        "ankara_parking_chunk_downloads_8km"
    )


def test_output_names_are_deterministic(
    tmp_path: Path,
) -> None:
    """Parking and metadata names must remain stable."""

    paths = AnkaraParkingDownloadPaths(
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

    assert parking_output_path(
        paths,
        "ANK_ROAD_0319",
    ).name == (
        "ANK_ROAD_0319_parking.gpkg"
    )

    assert metadata_output_path(
        paths,
        "ANK_ROAD_0319",
    ).name == (
        "ANK_ROAD_0319_metadata.json"
    )


def test_select_single_chunk() -> None:
    """One requested chunk must be selected."""

    selected = select_chunks(
        create_synthetic_chunks(),
        chunk_ids=[
            "ANK_ROAD_0002"
        ],
        download_all=False,
        start_order=1,
        limit=None,
    )

    assert selected[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0002"
    ]


def test_all_selection_supports_resume_and_limit() -> None:
    """All mode must support order-based batching."""

    selected = select_chunks(
        create_synthetic_chunks(),
        chunk_ids=None,
        download_all=True,
        start_order=2,
        limit=1,
    )

    assert selected[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0002"
    ]


def test_unknown_chunk_is_rejected() -> None:
    """Unknown chunk identifiers must fail clearly."""

    with pytest.raises(
        ValueError,
        match="Unknown chunk identifiers",
    ):
        select_chunks(
            create_synthetic_chunks(),
            chunk_ids=[
                "ANK_ROAD_9999"
            ],
            download_all=False,
            start_order=1,
            limit=None,
        )


def test_download_polygon_stays_under_limit() -> None:
    """An 8-km core plus buffer must stay below 100 km²."""

    polygon, area_m2 = (
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

    assert not polygon.is_empty
    assert area_m2 < 100_000_000
    assert area_m2 > 64_000_000


def test_multiple_empty_results_confirm_empty_chunk() -> None:
    """Two endpoint no-data results confirm an empty chunk."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: InsufficientResponseError: "
        "No data elements in server response.\n"
        "- endpoint-b: InsufficientResponseError: "
        "No matching features."
    )

    assert is_confirmed_empty_parking_response(
        error
    )


def test_single_empty_result_remains_retryable() -> None:
    """One empty result plus a connection error is not enough."""

    error = RuntimeError(
        "Attempt details:\n"
        "- endpoint-a: ConnectionError\n"
        "- endpoint-b: InsufficientResponseError: "
        "No data elements in server response."
    )

    assert not is_confirmed_empty_parking_response(
        error
    )


def test_empty_success_requires_no_geopackage(
    tmp_path: Path,
) -> None:
    """Confirmed empty chunks are reusable without a GIS file."""

    output_path = (
        tmp_path / "missing_parking.gpkg"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "status": "success",
                "is_empty": True,
                "parking_feature_count": 0,
                "output_path": None,
            }
        ),
        encoding="utf-8",
    )

    assert chunk_is_complete(
        output_path,
        metadata_path,
    )


def test_non_empty_success_requires_output(
    tmp_path: Path,
) -> None:
    """Non-empty metadata alone is not a completed result."""

    output_path = (
        tmp_path / "missing_parking.gpkg"
    )

    metadata_path = (
        tmp_path / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "status": "success",
                "is_empty": False,
                "parking_feature_count": 2,
                "output_path": output_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    assert not chunk_is_complete(
        output_path,
        metadata_path,
    )
