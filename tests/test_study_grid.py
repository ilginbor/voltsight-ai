from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cankaya_boundary_osm.geojson"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_250m.gpkg"
)

EXPECTED_GRID_SIZE_METERS = 250
EXPECTED_CELL_AREA_M2 = 62_500
AREA_TOLERANCE_M2 = 1.0

REQUIRED_GRID_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "grid_size_m",
    "cell_area_m2",
    "center_longitude",
    "center_latitude",
    "geometry",
}


@pytest.fixture(scope="session")
def boundary() -> gpd.GeoDataFrame:
    """
    Load the generated Çankaya administrative boundary.

    The grid generation pipeline must be executed before running
    this test suite.
    """

    if not BOUNDARY_PATH.exists():
        pytest.fail(
            "The Çankaya boundary file does not exist. "
            "Run create_study_grid.py before executing the tests."
        )

    return gpd.read_file(BOUNDARY_PATH)


@pytest.fixture(scope="session")
def grid() -> gpd.GeoDataFrame:
    """
    Load the generated 250 x 250 meter analysis grid.

    GeoPackage is used because it preserves the projected coordinate
    system required for metric geometry validation.
    """

    if not GRID_PATH.exists():
        pytest.fail(
            "The Çankaya grid file does not exist. "
            "Run create_study_grid.py before executing the tests."
        )

    return gpd.read_file(
        GRID_PATH,
        layer="cankaya_grid_250m",
    )


def test_boundary_is_not_empty(
    boundary: gpd.GeoDataFrame,
) -> None:
    """The downloaded district boundary must contain geometry."""

    assert not boundary.empty
    assert boundary.geometry.notna().all()
    assert not boundary.geometry.is_empty.any()


def test_boundary_has_coordinate_system(
    boundary: gpd.GeoDataFrame,
) -> None:
    """The boundary must have a defined coordinate system."""

    assert boundary.crs is not None


def test_boundary_is_wgs84(
    boundary: gpd.GeoDataFrame,
) -> None:
    """The raw web-compatible boundary must use EPSG:4326."""

    assert boundary.crs is not None
    assert boundary.crs.to_epsg() == 4326


def test_boundary_geometries_are_valid(
    boundary: gpd.GeoDataFrame,
) -> None:
    """All boundary geometries must be topologically valid."""

    assert boundary.geometry.is_valid.all()


def test_boundary_contains_polygon_geometry(
    boundary: gpd.GeoDataFrame,
) -> None:
    """The study boundary must be represented by polygons."""

    accepted_geometry_types = {
        "Polygon",
        "MultiPolygon",
    }

    geometry_types = set(
        boundary.geometry.geom_type.unique()
    )

    assert geometry_types.issubset(
        accepted_geometry_types
    )


def test_grid_is_not_empty(
    grid: gpd.GeoDataFrame,
) -> None:
    """The generated analysis grid must contain cells."""

    assert not grid.empty
    assert len(grid) > 0


def test_grid_has_projected_coordinate_system(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    The GeoPackage grid must use a projected coordinate system.

    Meter-based size and area calculations must not be performed
    directly in geographic latitude-longitude coordinates.
    """

    assert grid.crs is not None
    assert grid.crs.is_projected
    assert not grid.crs.is_geographic


def test_grid_contains_required_columns(
    grid: gpd.GeoDataFrame,
) -> None:
    """The machine learning grid schema must be complete."""

    existing_columns = set(grid.columns)
    missing_columns = (
        REQUIRED_GRID_COLUMNS - existing_columns
    )

    assert not missing_columns, (
        "Missing grid columns: "
        f"{sorted(missing_columns)}"
    )


def test_grid_geometries_are_not_missing(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every grid record must contain geometry."""

    assert grid.geometry.notna().all()
    assert not grid.geometry.is_empty.any()


def test_grid_geometries_are_valid(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every grid geometry must be topologically valid."""

    invalid_count = int(
        (~grid.geometry.is_valid).sum()
    )

    assert invalid_count == 0, (
        f"Invalid grid geometry count: {invalid_count}"
    )


def test_all_grid_geometries_are_polygons(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every retained analysis cell must be a Polygon."""

    geometry_types = set(
        grid.geometry.geom_type.unique()
    )

    assert geometry_types == {"Polygon"}


def test_grid_ids_are_complete_and_unique(
    grid: gpd.GeoDataFrame,
) -> None:
    """Each analysis cell must have a unique grid identifier."""

    assert grid["grid_id"].notna().all()
    assert not grid["grid_id"].duplicated().any()


def test_grid_ids_use_expected_prefix(
    grid: gpd.GeoDataFrame,
) -> None:
    """Çankaya grid identifiers must start with CKY_."""

    assert grid["grid_id"].str.startswith(
        "CKY_"
    ).all()


def test_grid_geometries_are_unique(
    grid: gpd.GeoDataFrame,
) -> None:
    """The dataset must not contain duplicate square geometries."""

    geometry_bytes = grid.geometry.to_wkb()

    duplicate_count = int(
        geometry_bytes.duplicated().sum()
    )

    assert duplicate_count == 0, (
        f"Duplicate geometry count: {duplicate_count}"
    )


def test_grid_size_attribute_is_correct(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every record must indicate a 250 meter grid size."""

    assert grid["grid_size_m"].notna().all()

    assert (
        grid["grid_size_m"]
        == EXPECTED_GRID_SIZE_METERS
    ).all()


def test_calculated_cell_areas_are_correct(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every square cell must cover approximately 62,500 m²."""

    calculated_areas = grid.geometry.area

    area_differences = (
        calculated_areas - EXPECTED_CELL_AREA_M2
    ).abs()

    maximum_difference = float(
        area_differences.max()
    )

    assert maximum_difference <= AREA_TOLERANCE_M2, (
        "Unexpected grid area difference. "
        f"Maximum difference: {maximum_difference:.6f} m²"
    )


def test_stored_cell_areas_match_geometry(
    grid: gpd.GeoDataFrame,
) -> None:
    """Stored cell areas must match geometry-derived areas."""

    calculated_areas = grid.geometry.area
    stored_areas = grid["cell_area_m2"]

    area_differences = (
        calculated_areas - stored_areas
    ).abs()

    maximum_difference = float(
        area_differences.max()
    )

    assert maximum_difference <= AREA_TOLERANCE_M2, (
        "Stored and calculated grid areas do not match. "
        f"Maximum difference: {maximum_difference:.6f} m²"
    )


def test_grid_district_and_city_are_correct(
    grid: gpd.GeoDataFrame,
) -> None:
    """All generated cells must belong to Çankaya, Ankara."""

    assert set(grid["district"].dropna().unique()) == {
        "Çankaya"
    }

    assert set(grid["city"].dropna().unique()) == {
        "Ankara"
    }


def test_grid_center_coordinates_are_complete(
    grid: gpd.GeoDataFrame,
) -> None:
    """Every cell must contain WGS84 center coordinates."""

    assert grid["center_longitude"].notna().all()
    assert grid["center_latitude"].notna().all()


def test_grid_center_coordinates_are_in_valid_ranges(
    grid: gpd.GeoDataFrame,
) -> None:
    """Longitude and latitude values must use valid global ranges."""

    assert grid["center_longitude"].between(
        -180,
        180,
    ).all()

    assert grid["center_latitude"].between(
        -90,
        90,
    ).all()


def test_grid_centers_are_inside_boundary(
    boundary: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> None:
    """
    Every retained square must have its center inside or on the
    Çankaya administrative boundary.
    """

    projected_boundary = boundary.to_crs(
        grid.crs
    )

    district_geometry = (
        projected_boundary.geometry.union_all()
    )

    grid_centers = grid.geometry.centroid

    outside_centers = [
        center
        for center in grid_centers
        if not district_geometry.covers(center)
    ]

    assert not outside_centers, (
        "Some grid centers fall outside the Çankaya boundary. "
        f"Outside center count: {len(outside_centers)}"
    )


def test_grid_total_bounds_are_finite(
    grid: gpd.GeoDataFrame,
) -> None:
    """Grid bounding coordinates must contain finite values."""

    bounds = grid.total_bounds

    assert len(bounds) == 4
    assert all(
        value == value
        for value in bounds
    )


def test_grid_contains_reasonable_number_of_cells(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    Detect obviously incorrect boundary or grid generation results.

    This is intentionally a broad range rather than an exact count,
    because OpenStreetMap boundaries may receive minor updates.
    """

    assert 1_000 < len(grid) < 20_000, (
        "The generated grid cell count appears unreasonable. "
        f"Cell count: {len(grid)}"
    )