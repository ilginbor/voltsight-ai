from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from voltsight.features.create_ankara_road_chunk_plan import (
    RoadChunkPlanPaths,
    add_download_area_estimates,
    assign_grid_cells_to_chunks,
    create_core_chunks,
    create_summary,
    parse_arguments,
    resolve_paths,
    validate_parameters,
)


def create_synthetic_grid() -> gpd.GeoDataFrame:
    """Create four synthetic 10-km grid cells."""

    geometries = [
        box(0, 0, 10_000, 10_000),
        box(0, 10_000, 10_000, 20_000),
        box(10_000, 0, 20_000, 10_000),
        box(10_000, 10_000, 20_000, 20_000),
    ]

    return gpd.GeoDataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
                "ANK_000004",
            ],
            "grid_size_m": [10_000] * 4,
            "cell_area_m2": [
                100_000_000
            ] * 4,
        },
        geometry=geometries,
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """The Ankara plan must default to 500-m grid and 20-km chunks."""

    arguments = parse_arguments(
        []
    )

    assert arguments.grid_size_m == 500
    assert arguments.chunk_size_m == 20_000
    assert arguments.download_buffer_m == 1_000


def test_path_resolution() -> None:
    """Chunk outputs must use deterministic Ankara names."""

    paths = resolve_paths(
        grid_size_m=500,
        chunk_size_m=20_000,
    )

    assert paths.grid_gpkg.name == (
        "ankara_grid_500m.gpkg"
    )

    assert paths.grid_layer_name == (
        "ankara_grid_500m"
    )

    assert paths.chunk_gpkg.name == (
        "ankara_road_download_chunks_20km.gpkg"
    )

    assert paths.preview_png.name == (
        "ankara_road_download_chunks_20km_preview.png"
    )


def test_invalid_chunk_size_is_rejected() -> None:
    """Chunk dimensions must be positive."""

    with pytest.raises(
        ValueError,
        match="Chunk size must be positive",
    ):
        validate_parameters(
            grid_size_m=500,
            chunk_size_m=0,
            download_buffer_m=1_000,
        )


def test_invalid_buffer_is_rejected() -> None:
    """The buffer cannot be as large as the core chunk."""

    with pytest.raises(
        ValueError,
        match="smaller than the chunk size",
    ):
        validate_parameters(
            grid_size_m=500,
            chunk_size_m=20_000,
            download_buffer_m=20_000,
        )


def test_square_boundary_creates_four_chunks() -> None:
    """A 40-km square must create four 20-km core chunks."""

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                0,
                0,
                40_000,
                40_000,
            )
        ],
        crs="EPSG:32636",
    )

    chunks = create_core_chunks(
        boundary,
        chunk_size_m=20_000,
    )

    assert len(chunks) == 4

    assert chunks[
        "chunk_id"
    ].tolist() == [
        "ANK_ROAD_0001",
        "ANK_ROAD_0002",
        "ANK_ROAD_0003",
        "ANK_ROAD_0004",
    ]

    assert chunks[
        "chunk_order"
    ].tolist() == [
        1,
        2,
        3,
        4,
    ]

    assert set(
        chunks["core_area_km2"]
    ) == {
        400.0,
    }


def test_grid_cells_are_assigned_once() -> None:
    """Every synthetic grid centroid must belong to one chunk."""

    grid = create_synthetic_grid()

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                0,
                0,
                20_000,
                20_000,
            )
        ],
        crs=grid.crs,
    )

    chunks = create_core_chunks(
        boundary,
        chunk_size_m=10_000,
    )

    assigned = assign_grid_cells_to_chunks(
        chunks,
        grid,
    )

    assert int(
        assigned[
            "grid_cell_count"
        ].sum()
    ) == 4

    assert sorted(
        assigned[
            "grid_cell_count"
        ].tolist()
    ) == [
        1,
        1,
        1,
        1,
    ]


def test_buffer_estimates_are_larger_than_core() -> None:
    """Buffered download areas must not be smaller than core areas."""

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                0,
                0,
                20_000,
                20_000,
            )
        ],
        crs="EPSG:32636",
    )

    chunks = create_core_chunks(
        boundary,
        chunk_size_m=20_000,
    )

    result = add_download_area_estimates(
        chunks,
        download_buffer_m=1_000,
    )

    assert (
        result[
            "estimated_download_area_km2"
        ]
        >= result["core_area_km2"]
    ).all()

    assert set(
        result["download_buffer_m"]
    ) == {
        1_000,
    }


def test_non_rectangular_boundary_is_clipped() -> None:
    """Chunks must retain only the area inside the boundary."""

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                0,
                0,
                30_000,
                10_000,
            )
        ],
        crs="EPSG:32636",
    )

    chunks = create_core_chunks(
        boundary,
        chunk_size_m=20_000,
    )

    assert len(chunks) == 2

    assert float(
        chunks["core_area_km2"].sum()
    ) == pytest.approx(
        300.0,
        abs=0.01,
    )



def test_summary_is_created_in_chunk_order(
    tmp_path: Path,
) -> None:
    """Summary creation must sort before selecting report columns."""

    grid = create_synthetic_grid()

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                0,
                0,
                20_000,
                20_000,
            )
        ],
        crs=grid.crs,
    )

    chunks = create_core_chunks(
        boundary,
        chunk_size_m=10_000,
    )

    chunks = assign_grid_cells_to_chunks(
        chunks,
        grid,
    )

    chunks = add_download_area_estimates(
        chunks,
        download_buffer_m=1_000,
    )

    paths = RoadChunkPlanPaths(
        boundary_geojson=(
            tmp_path / "boundary.geojson"
        ),
        grid_gpkg=(
            tmp_path / "grid.gpkg"
        ),
        grid_layer_name=(
            "ankara_grid_10000m"
        ),
        chunk_gpkg=(
            tmp_path / "chunks.gpkg"
        ),
        preview_png=(
            tmp_path / "preview.png"
        ),
        summary_md=(
            tmp_path / "summary.md"
        ),
        chunk_cache_directory=(
            tmp_path / "cache"
        ),
    )

    create_summary(
        chunks=chunks,
        grid=grid,
        paths=paths,
        grid_size_m=10_000,
        chunk_size_m=10_000,
        download_buffer_m=1_000,
    )

    assert paths.summary_md.exists()

    summary = paths.summary_md.read_text(
        encoding="utf-8"
    )

    first_position = summary.index(
        "ANK_ROAD_0001"
    )

    last_position = summary.index(
        "ANK_ROAD_0004"
    )

    assert first_position < last_position
