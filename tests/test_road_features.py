from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STUDY_GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_250m.gpkg"
)

ROAD_NETWORK_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_drive_roads.gpkg"
)

ROAD_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.gpkg"
)

ROAD_FEATURES_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.csv"
)

ROAD_FEATURE_PREVIEW_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_road_features_preview.png"
)

ROAD_FEATURE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_road_features_summary.md"
)

STUDY_GRID_LAYER = "cankaya_grid_250m"
ROAD_NETWORK_LAYER = "drive_roads"
ROAD_FEATURE_LAYER = "grid_road_features"

EXPECTED_GRID_SIZE_METERS = 250
EXPECTED_CELL_AREA_M2 = 62_500

DENSITY_TOLERANCE = 0.0001
LENGTH_TOLERANCE_METERS = 0.02

MAIN_ROAD_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}

REQUIRED_FEATURE_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "grid_size_m",
    "cell_area_m2",
    "center_longitude",
    "center_latitude",
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
    "geometry",
}

REQUIRED_ROAD_COLUMNS = {
    "road_id",
    "osm_id",
    "name",
    "highway",
    "is_main_road",
    "edge_length_m",
    "geometry",
}

NUMERIC_FEATURE_COLUMNS = [
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
]


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail with a clear message when a generated file is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run create_study_grid.py and "
            "create_road_features.py before running these tests."
        )


def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize boolean-like values read from GIS file formats."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def split_highway_types(
    value: object,
) -> set[str]:
    """Split semicolon-separated OpenStreetMap highway values."""

    if value is None:
        return set()

    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


@pytest.fixture(scope="session")
def study_grid() -> gpd.GeoDataFrame:
    """Load the original 250 x 250 meter study grid."""

    ensure_file_exists(
        STUDY_GRID_PATH,
        "Study-grid GeoPackage",
    )

    return gpd.read_file(
        STUDY_GRID_PATH,
        layer=STUDY_GRID_LAYER,
    )


@pytest.fixture(scope="session")
def road_network() -> gpd.GeoDataFrame:
    """Load the processed physical road network."""

    ensure_file_exists(
        ROAD_NETWORK_PATH,
        "Road-network GeoPackage",
    )

    return gpd.read_file(
        ROAD_NETWORK_PATH,
        layer=ROAD_NETWORK_LAYER,
    )


@pytest.fixture(scope="session")
def road_features() -> gpd.GeoDataFrame:
    """Load the generated grid-level road features."""

    ensure_file_exists(
        ROAD_FEATURES_PATH,
        "Road-feature GeoPackage",
    )

    return gpd.read_file(
        ROAD_FEATURES_PATH,
        layer=ROAD_FEATURE_LAYER,
    )


@pytest.fixture(scope="session")
def road_features_csv() -> pd.DataFrame:
    """Load the machine-learning compatible CSV output."""

    ensure_file_exists(
        ROAD_FEATURES_CSV_PATH,
        "Road-feature CSV",
    )

    return pd.read_csv(
        ROAD_FEATURES_CSV_PATH,
        encoding="utf-8",
    )


def test_road_feature_dataset_is_not_empty(
    road_features: gpd.GeoDataFrame,
) -> None:
    """The generated road-feature dataset must contain records."""

    assert not road_features.empty
    assert len(road_features) > 0


def test_road_feature_row_count_matches_study_grid(
    study_grid: gpd.GeoDataFrame,
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every study-grid cell must have one road-feature record."""

    assert len(road_features) == len(study_grid)


def test_road_feature_dataset_contains_required_columns(
    road_features: gpd.GeoDataFrame,
) -> None:
    """The road-feature schema must contain all expected columns."""

    missing_columns = (
        REQUIRED_FEATURE_COLUMNS
        - set(road_features.columns)
    )

    assert not missing_columns, (
        "Missing road-feature columns: "
        f"{sorted(missing_columns)}"
    )


def test_road_feature_crs_is_projected(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Road distances and lengths must use a projected CRS."""

    assert road_features.crs is not None
    assert road_features.crs.is_projected
    assert not road_features.crs.is_geographic


def test_road_feature_crs_matches_study_grid(
    study_grid: gpd.GeoDataFrame,
    road_features: gpd.GeoDataFrame,
) -> None:
    """The feature output must preserve the study-grid CRS."""

    assert study_grid.crs is not None
    assert road_features.crs == study_grid.crs


def test_road_feature_geometries_are_complete(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every feature record must contain a non-empty geometry."""

    assert road_features.geometry.notna().all()
    assert not road_features.geometry.is_empty.any()


def test_road_feature_geometries_are_valid(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every grid-cell geometry must be topologically valid."""

    invalid_count = int(
        (~road_features.geometry.is_valid).sum()
    )

    assert invalid_count == 0, (
        f"Invalid feature geometry count: {invalid_count}"
    )


def test_road_feature_geometries_are_polygons(
    road_features: gpd.GeoDataFrame,
) -> None:
    """All road-feature records must remain square polygons."""

    geometry_types = set(
        road_features.geometry.geom_type.unique()
    )

    assert geometry_types == {"Polygon"}


def test_road_feature_grid_ids_are_complete_and_unique(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every feature record must have one unique grid identifier."""

    assert road_features["grid_id"].notna().all()
    assert not road_features["grid_id"].duplicated().any()


def test_road_feature_grid_ids_match_study_grid(
    study_grid: gpd.GeoDataFrame,
    road_features: gpd.GeoDataFrame,
) -> None:
    """Road features must cover exactly the original grid IDs."""

    study_grid_ids = set(
        study_grid["grid_id"].astype(str)
    )

    feature_grid_ids = set(
        road_features["grid_id"].astype(str)
    )

    assert feature_grid_ids == study_grid_ids


def test_grid_size_and_area_are_preserved(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Feature engineering must not alter grid dimensions."""

    assert (
        road_features["grid_size_m"]
        == EXPECTED_GRID_SIZE_METERS
    ).all()

    assert np.allclose(
        road_features["cell_area_m2"].to_numpy(
            dtype=float
        ),
        EXPECTED_CELL_AREA_M2,
        atol=1.0,
    )


def test_numeric_road_features_are_complete(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Numeric model features must not contain missing values."""

    for column in NUMERIC_FEATURE_COLUMNS:
        missing_count = int(
            road_features[column].isna().sum()
        )

        assert missing_count == 0, (
            f"{column} contains "
            f"{missing_count} missing values."
        )


def test_numeric_road_features_are_finite(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Numeric model features must contain only finite values."""

    for column in NUMERIC_FEATURE_COLUMNS:
        values = road_features[column].to_numpy(
            dtype=float
        )

        assert np.isfinite(values).all(), (
            f"{column} contains non-finite values."
        )


def test_numeric_road_features_are_non_negative(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Distances, lengths, counts and densities cannot be negative."""

    for column in NUMERIC_FEATURE_COLUMNS:
        values = road_features[column].to_numpy(
            dtype=float
        )

        assert (values >= 0).all(), (
            f"{column} contains negative values."
        )


def test_segment_counts_are_whole_numbers(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Road segment counts must be integer-like values."""

    count_columns = [
        "road_segment_count",
        "main_road_segment_count",
    ]

    for column in count_columns:
        values = road_features[column].to_numpy(
            dtype=float
        )

        assert np.allclose(
            values,
            np.round(values),
        ), f"{column} contains fractional counts."


def test_main_road_length_does_not_exceed_total_length(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Main-road length is a subset of total road length."""

    difference = (
        road_features["main_road_length_m"]
        - road_features["road_length_m"]
    )

    assert (
        difference
        <= LENGTH_TOLERANCE_METERS
    ).all()


def test_main_road_count_does_not_exceed_total_count(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Main-road segment count cannot exceed total segment count."""

    assert (
        road_features["main_road_segment_count"]
        <= road_features["road_segment_count"]
    ).all()


def test_road_density_matches_length_and_area(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Stored density must match road length divided by cell area."""

    road_length_km = (
        road_features["road_length_m"]
        / 1_000
    )

    cell_area_km2 = (
        road_features["cell_area_m2"]
        / 1_000_000
    )

    expected_density = (
        road_length_km
        / cell_area_km2
    )

    stored_density = road_features[
        "road_density_km_per_km2"
    ]

    assert np.allclose(
        stored_density.to_numpy(dtype=float),
        expected_density.to_numpy(dtype=float),
        atol=DENSITY_TOLERANCE,
    )


def test_zero_road_cells_have_zero_density_and_count(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Cells without roads must also have zero density and count."""

    zero_road_cells = road_features[
        road_features["road_length_m"] == 0
    ]

    assert not zero_road_cells.empty

    assert (
        zero_road_cells["road_segment_count"]
        == 0
    ).all()

    assert (
        zero_road_cells["road_density_km_per_km2"]
        == 0
    ).all()

    assert (
        zero_road_cells["main_road_length_m"]
        == 0
    ).all()

    assert (
        zero_road_cells["main_road_segment_count"]
        == 0
    ).all()


def test_dataset_contains_cells_with_and_without_roads(
    road_features: gpd.GeoDataFrame,
) -> None:
    """The study area should contain both urban and road-free cells."""

    cells_with_roads = int(
        (
            road_features["road_length_m"]
            > 0
        ).sum()
    )

    cells_without_roads = int(
        (
            road_features["road_length_m"]
            == 0
        ).sum()
    )

    assert cells_with_roads > 0
    assert cells_without_roads > 0

    assert (
        cells_with_roads
        + cells_without_roads
        == len(road_features)
    )


def test_main_road_distances_are_complete(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every grid cell must be matched to a nearest main road."""

    distances = road_features[
        "distance_to_main_road_m"
    ]

    assert distances.notna().all()
    assert np.isfinite(
        distances.to_numpy(dtype=float)
    ).all()


def test_nearest_main_road_types_are_complete(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Every nearest-road result must contain a highway type."""

    nearest_types = (
        road_features["nearest_main_road_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert nearest_types.ne("").all()


def test_nearest_main_road_types_are_valid(
    road_features: gpd.GeoDataFrame,
) -> None:
    """Nearest-road classes must include a configured main-road type."""

    invalid_values: list[str] = []

    for value in road_features[
        "nearest_main_road_type"
    ]:
        highway_types = split_highway_types(
            value
        )

        if not (
            highway_types
            & MAIN_ROAD_TYPES
        ):
            invalid_values.append(
                str(value)
            )

    assert not invalid_values, (
        "Unexpected nearest main-road types: "
        f"{sorted(set(invalid_values))[:10]}"
    )


def test_road_network_is_not_empty(
    road_network: gpd.GeoDataFrame,
) -> None:
    """The downloaded physical road network must contain edges."""

    assert not road_network.empty
    assert len(road_network) > 0


def test_road_network_contains_required_columns(
    road_network: gpd.GeoDataFrame,
) -> None:
    """The processed road-network schema must be complete."""

    missing_columns = (
        REQUIRED_ROAD_COLUMNS
        - set(road_network.columns)
    )

    assert not missing_columns, (
        "Missing road-network columns: "
        f"{sorted(missing_columns)}"
    )


def test_road_network_geometries_are_valid_lines(
    road_network: gpd.GeoDataFrame,
) -> None:
    """Road-network geometries must be valid line features."""

    assert road_network.geometry.notna().all()
    assert not road_network.geometry.is_empty.any()
    assert road_network.geometry.is_valid.all()

    geometry_types = set(
        road_network.geometry.geom_type.unique()
    )

    assert geometry_types.issubset(
        {
            "LineString",
            "MultiLineString",
        }
    )


def test_road_edge_lengths_match_geometry(
    road_network: gpd.GeoDataFrame,
) -> None:
    """Stored road-edge lengths must match geometric lengths."""

    calculated_lengths = (
        road_network.geometry.length
    )

    stored_lengths = road_network[
        "edge_length_m"
    ].astype(float)

    assert np.allclose(
        stored_lengths.to_numpy(),
        calculated_lengths.to_numpy(),
        atol=LENGTH_TOLERANCE_METERS,
    )


def test_road_network_contains_main_roads(
    road_network: gpd.GeoDataFrame,
) -> None:
    """At least one downloaded edge must be classified as main road."""

    main_road_mask = normalize_boolean_series(
        road_network["is_main_road"]
    )

    assert main_road_mask.any()


def test_csv_row_count_matches_geospatial_output(
    road_features: gpd.GeoDataFrame,
    road_features_csv: pd.DataFrame,
) -> None:
    """The machine-learning CSV must contain one row per grid cell."""

    assert len(road_features_csv) == len(
        road_features
    )


def test_csv_grid_ids_match_geospatial_output(
    road_features: gpd.GeoDataFrame,
    road_features_csv: pd.DataFrame,
) -> None:
    """CSV and GeoPackage outputs must describe the same grid IDs."""

    geospatial_ids = set(
        road_features["grid_id"].astype(str)
    )

    csv_ids = set(
        road_features_csv["grid_id"].astype(str)
    )

    assert csv_ids == geospatial_ids


def test_csv_contains_required_model_features(
    road_features_csv: pd.DataFrame,
) -> None:
    """The CSV must include all numeric road model inputs."""

    required_csv_columns = {
        "grid_id",
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
        "road_density_km_per_km2",
        "distance_to_main_road_m",
        "nearest_main_road_type",
    }

    missing_columns = (
        required_csv_columns
        - set(road_features_csv.columns)
    )

    assert not missing_columns, (
        "Missing CSV columns: "
        f"{sorted(missing_columns)}"
    )


def test_documentation_outputs_exist() -> None:
    """The pipeline must produce a preview and summary document."""

    assert ROAD_FEATURE_PREVIEW_PATH.exists()
    assert ROAD_FEATURE_PREVIEW_PATH.stat().st_size > 0

    assert ROAD_FEATURE_SUMMARY_PATH.exists()
    assert ROAD_FEATURE_SUMMARY_PATH.stat().st_size > 0