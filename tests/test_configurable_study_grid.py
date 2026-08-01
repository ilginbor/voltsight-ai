from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from voltsight.core.study_areas import (
    get_study_area,
)
from voltsight.data.create_study_grid import (
    generate_square_grid,
    get_polygon_result,
    parse_arguments,
    resolve_pipeline_paths,
)


def test_default_cli_arguments() -> None:
    """The legacy default execution must remain Çankaya at 250 m."""

    arguments = parse_arguments(
        []
    )

    assert arguments.study_area == "cankaya"
    assert arguments.grid_size_m is None
    assert not arguments.reuse_boundary
    assert not arguments.force_download


def test_ankara_cli_arguments() -> None:
    """Ankara and its default grid may be selected through the CLI."""

    arguments = parse_arguments(
        [
            "--study-area",
            "ankara",
            "--reuse-boundary",
        ]
    )

    assert arguments.study_area == "ankara"
    assert arguments.grid_size_m is None
    assert arguments.reuse_boundary
    assert not arguments.force_download


def test_grid_size_override_argument() -> None:
    """Supported grid sizes must be accepted."""

    arguments = parse_arguments(
        [
            "--study-area",
            "ankara",
            "--grid-size-m",
            "250",
        ]
    )

    assert arguments.grid_size_m == 250


def test_invalid_grid_size_is_rejected() -> None:
    """Unsupported grid sizes must fail during argument parsing."""

    with pytest.raises(
        SystemExit,
    ):
        parse_arguments(
            [
                "--grid-size-m",
                "300",
            ]
        )


def test_conflicting_boundary_flags_are_rejected() -> None:
    """Boundary reuse and forced download cannot run together."""

    with pytest.raises(
        SystemExit,
    ):
        parse_arguments(
            [
                "--reuse-boundary",
                "--force-download",
            ]
        )


def test_cankaya_legacy_document_paths() -> None:
    """The original Çankaya preview and summary names must remain."""

    paths = resolve_pipeline_paths(
        get_study_area(
            "cankaya"
        )
    )

    assert paths.grid_size_m == 250

    assert paths.grid_gpkg.name == (
        "cankaya_grid_250m.gpkg"
    )

    assert paths.grid_layer_name == (
        "cankaya_grid_250m"
    )

    assert paths.grid_preview_png.name == (
        "cankaya_grid_preview.png"
    )

    assert paths.grid_summary_md.name == (
        "cankaya_grid_summary.md"
    )


def test_ankara_default_paths() -> None:
    """Ankara must default to deterministic 500-metre outputs."""

    paths = resolve_pipeline_paths(
        get_study_area(
            "ankara"
        )
    )

    assert paths.grid_size_m == 500

    assert paths.boundary_geojson.name == (
        "ankara_boundary_osm.geojson"
    )

    assert paths.grid_gpkg.name == (
        "ankara_grid_500m.gpkg"
    )

    assert paths.grid_geojson.name == (
        "ankara_grid_500m.geojson"
    )

    assert paths.grid_layer_name == (
        "ankara_grid_500m"
    )


def test_polygon_filter_removes_non_polygon_geometry() -> None:
    """Only polygonal boundary results may be used."""

    result = gpd.GeoDataFrame(
        {
            "name": [
                "polygon",
                "point",
            ],
        },
        geometry=[
            box(
                0,
                0,
                1,
                1,
            ),
            box(
                0,
                0,
                1,
                1,
            ).centroid,
        ],
        crs="EPSG:4326",
    )

    filtered = get_polygon_result(
        result
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "polygon"


def test_generate_cankaya_compatible_grid() -> None:
    """A synthetic 1-km square must produce four 500-metre cells."""

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                500_000,
                4_400_000,
                501_000,
                4_401_000,
            )
        ],
        crs="EPSG:32636",
    )

    grid = generate_square_grid(
        boundary_projected=boundary,
        projected_crs="EPSG:32636",
        grid_size=500,
        grid_prefix="CKY",
        district_name="Çankaya",
        city_name="Ankara",
    )

    assert len(grid) == 4

    assert grid[
        "grid_id"
    ].tolist() == [
        "CKY_00001",
        "CKY_00002",
        "CKY_00003",
        "CKY_00004",
    ]

    assert set(
        grid["district"]
    ) == {
        "Çankaya",
    }

    assert set(
        grid["cell_area_m2"]
    ) == {
        250_000.0,
    }


def test_generate_ankara_grid_identifiers() -> None:
    """Province grids must use six-digit ANK identifiers."""

    boundary = gpd.GeoDataFrame(
        geometry=[
            box(
                500_000,
                4_400_000,
                501_000,
                4_401_000,
            )
        ],
        crs="EPSG:32636",
    )

    grid = generate_square_grid(
        boundary_projected=boundary,
        projected_crs="EPSG:32636",
        grid_size=500,
        grid_prefix="ANK",
        district_name=None,
        city_name="Ankara",
    )

    assert len(grid) == 4

    assert grid[
        "grid_id"
    ].tolist() == [
        "ANK_000001",
        "ANK_000002",
        "ANK_000003",
        "ANK_000004",
    ]

    assert set(
        grid["district"]
    ) == {
        "Province-wide",
    }

    assert grid[
        "center_longitude"
    ].notna().all()

    assert grid[
        "center_latitude"
    ].notna().all()
