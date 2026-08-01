from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.gpkg"
)

CHARGING_STATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.gpkg"
)

CHARGING_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.gpkg"
)

CHARGING_FEATURES_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.csv"
)

CHARGING_PREVIEW_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_features_preview.png"
)

CHARGING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_features_summary.md"
)

BASE_FEATURE_LAYER = "grid_parking_features"
CHARGING_STATION_LAYER = "charging_stations_merged"
CHARGING_FEATURE_LAYER = "grid_charging_features"

DISTANCE_TOLERANCE_METERS = 0.02

SUPPORTED_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
}

REQUIRED_STATION_COLUMNS = {
    "station_id",
    "osm_element_type",
    "osm_id",
    "name",
    "operator",
    "brand",
    "network",
    "capacity",
    "capacity_numeric",
    "connector_types",
    "mapped_socket_type_count",
    "known_socket_count",
    "has_ac_connector",
    "has_dc_connector",
    "geometry_type",
    "geometry",
}

REQUIRED_FEATURE_COLUMNS = {
    "grid_id",
    "charging_station_count",
    "has_existing_charging_station",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "known_charging_capacity",
    "charging_capacity_record_count",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
    "geometry",
}

PREVIOUS_ROAD_FEATURE_COLUMNS = {
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
}

PREVIOUS_PARKING_FEATURE_COLUMNS = {
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
}

NUMERIC_CHARGING_COLUMNS = [
    "charging_station_count",
    "has_existing_charging_station",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "known_charging_capacity",
    "charging_capacity_record_count",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
]

INTEGER_CHARGING_COLUMNS = [
    "charging_station_count",
    "has_existing_charging_station",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "charging_capacity_record_count",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
]


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a required pipeline output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run the study-grid, road, parking and charging "
            "pipelines before executing these tests."
        )


def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize boolean-like values read from GIS files."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return (
            series.fillna(0)
            .astype(int)
            .astype(bool)
        )

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


def create_station_points(
    charging_stations: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create one representative point for each station."""

    station_points = charging_stations[
        [
            "station_id",
            "capacity_numeric",
            "has_ac_connector",
            "has_dc_connector",
            "geometry",
        ]
    ].copy()

    station_points["has_ac_connector"] = (
        normalize_boolean_series(
            station_points["has_ac_connector"]
        )
    )

    station_points["has_dc_connector"] = (
        normalize_boolean_series(
            station_points["has_dc_connector"]
        )
    )

    station_points["geometry"] = (
        station_points.geometry.representative_point()
    )

    return gpd.GeoDataFrame(
        station_points,
        geometry="geometry",
        crs=charging_stations.crs,
    )


def calculate_expected_radius_counts(
    base_features: gpd.GeoDataFrame,
    station_points: gpd.GeoDataFrame,
    radius_meters: int,
) -> pd.Series:
    """Recalculate station counts around grid centroids."""

    grid_buffers = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_buffers["geometry"] = (
        grid_buffers.geometry
        .centroid
        .buffer(radius_meters)
    )

    joined = gpd.sjoin(
        grid_buffers,
        station_points[
            [
                "station_id",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    valid_joined = joined[
        joined["station_id"].notna()
    ].copy()

    if valid_joined.empty:
        return pd.Series(
            dtype=int,
            name="expected_count",
        )

    return (
        valid_joined
        .groupby("grid_id")["station_id"]
        .nunique()
    )


@pytest.fixture(scope="session")
def base_features() -> gpd.GeoDataFrame:
    """Load the existing parking-feature grid."""

    ensure_file_exists(
        BASE_FEATURES_PATH,
        "Parking-feature GeoPackage",
    )

    return gpd.read_file(
        BASE_FEATURES_PATH,
        layer=BASE_FEATURE_LAYER,
    )


@pytest.fixture(scope="session")
def charging_stations() -> gpd.GeoDataFrame:
    """Load the cleaned charging-station dataset."""

    ensure_file_exists(
        CHARGING_STATIONS_PATH,
        "Charging-station GeoPackage",
    )

    return gpd.read_file(
        CHARGING_STATIONS_PATH,
        layer=CHARGING_STATION_LAYER,
    )


@pytest.fixture(scope="session")
def charging_features() -> gpd.GeoDataFrame:
    """Load the grid-level charging features."""

    ensure_file_exists(
        CHARGING_FEATURES_PATH,
        "Charging-feature GeoPackage",
    )

    return gpd.read_file(
        CHARGING_FEATURES_PATH,
        layer=CHARGING_FEATURE_LAYER,
    )


@pytest.fixture(scope="session")
def charging_features_csv() -> pd.DataFrame:
    """Load the machine-learning charging CSV."""

    ensure_file_exists(
        CHARGING_FEATURES_CSV_PATH,
        "Charging-feature CSV",
    )

    return pd.read_csv(
        CHARGING_FEATURES_CSV_PATH,
        encoding="utf-8",
    )


def test_charging_station_dataset_is_not_empty(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """The charging-station dataset must contain records."""

    assert not charging_stations.empty
    assert len(charging_stations) > 0


def test_charging_station_dataset_has_required_columns(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """The cleaned station schema must be complete."""

    missing_columns = (
        REQUIRED_STATION_COLUMNS
        - set(charging_stations.columns)
    )

    assert not missing_columns, (
        "Missing charging-station columns: "
        f"{sorted(missing_columns)}"
    )


def test_station_ids_are_complete_and_unique(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Every mapped station must have a unique identifier."""

    station_ids = (
        charging_stations["station_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert station_ids.ne("").all()
    assert not station_ids.duplicated().any()


def test_station_geometries_are_complete(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Every station must contain non-empty geometry."""

    assert charging_stations.geometry.notna().all()
    assert not charging_stations.geometry.is_empty.any()


def test_station_geometries_are_valid(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Every station geometry must be valid."""

    assert charging_stations.geometry.is_valid.all()


def test_station_geometry_types_are_supported(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Only supported station geometries may remain."""

    geometry_types = set(
        charging_stations.geometry.geom_type.unique()
    )

    assert geometry_types.issubset(
        SUPPORTED_GEOMETRY_TYPES
    )


def test_stored_geometry_types_match_actual_geometry(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Stored geometry_type values must be correct."""

    stored_types = (
        charging_stations["geometry_type"]
        .astype(str)
        .to_numpy()
    )

    actual_types = (
        charging_stations.geometry.geom_type
        .astype(str)
        .to_numpy()
    )

    assert np.array_equal(
        stored_types,
        actual_types,
    )


def test_station_crs_is_projected(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Station distance calculations require projected CRS."""

    assert charging_stations.crs is not None
    assert charging_stations.crs.is_projected
    assert not charging_stations.crs.is_geographic


def test_station_crs_matches_base_features(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Stations and the analysis grid must use the same CRS."""

    assert base_features.crs is not None
    assert charging_stations.crs == base_features.crs


def test_station_capacities_are_non_negative(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Known numeric station capacities cannot be negative."""

    capacities = (
        charging_stations["capacity_numeric"]
        .dropna()
        .astype(float)
    )

    if capacities.empty:
        pytest.skip(
            "No numeric station capacities were mapped."
        )

    assert np.isfinite(
        capacities.to_numpy()
    ).all()

    assert (
        capacities >= 0
    ).all()


def test_socket_counts_are_non_negative(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Mapped connector counts cannot be negative."""

    columns = [
        "mapped_socket_type_count",
        "known_socket_count",
    ]

    for column in columns:
        values = charging_stations[
            column
        ].to_numpy(dtype=float)

        assert np.isfinite(values).all()
        assert (values >= 0).all()


def test_mapped_socket_type_count_is_integer_like(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Mapped socket-type counts must be whole numbers."""

    values = charging_stations[
        "mapped_socket_type_count"
    ].to_numpy(dtype=float)

    assert np.allclose(
        values,
        np.round(values),
    )


def test_connector_flags_are_normalizable(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """AC and DC connector flags must be valid booleans."""

    ac_flags = normalize_boolean_series(
        charging_stations["has_ac_connector"]
    )

    dc_flags = normalize_boolean_series(
        charging_stations["has_dc_connector"]
    )

    assert len(ac_flags) == len(
        charging_stations
    )

    assert len(dc_flags) == len(
        charging_stations
    )


def test_charging_feature_dataset_is_not_empty(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """The grid-level charging dataset must contain records."""

    assert not charging_features.empty
    assert len(charging_features) > 0


def test_charging_feature_row_count_matches_base_grid(
    base_features: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Every grid cell must have one charging record."""

    assert len(charging_features) == len(
        base_features
    )


def test_charging_features_have_required_columns(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """All expected charging features must exist."""

    missing_columns = (
        REQUIRED_FEATURE_COLUMNS
        - set(charging_features.columns)
    )

    assert not missing_columns, (
        "Missing charging-feature columns: "
        f"{sorted(missing_columns)}"
    )


def test_previous_road_and_parking_columns_are_preserved(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging engineering must preserve previous features."""

    expected_columns = (
        PREVIOUS_ROAD_FEATURE_COLUMNS
        | PREVIOUS_PARKING_FEATURE_COLUMNS
    )

    missing_columns = (
        expected_columns
        - set(charging_features.columns)
    )

    assert not missing_columns, (
        "Previous feature columns were lost: "
        f"{sorted(missing_columns)}"
    )


def test_charging_feature_crs_matches_base_grid(
    base_features: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """The output must preserve the projected analysis CRS."""

    assert charging_features.crs is not None
    assert charging_features.crs.is_projected
    assert charging_features.crs == base_features.crs


def test_charging_feature_grid_ids_are_unique(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Grid identifiers must remain complete and unique."""

    grid_ids = (
        charging_features["grid_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert grid_ids.ne("").all()
    assert not grid_ids.duplicated().any()


def test_charging_feature_grid_ids_match_base_grid(
    base_features: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging output must cover exactly the base grid."""

    base_ids = set(
        base_features["grid_id"].astype(str)
    )

    output_ids = set(
        charging_features["grid_id"].astype(str)
    )

    assert output_ids == base_ids


def test_charging_feature_geometries_are_valid_polygons(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Output geometries must remain valid grid polygons."""

    assert charging_features.geometry.notna().all()
    assert not charging_features.geometry.is_empty.any()
    assert charging_features.geometry.is_valid.all()

    geometry_types = set(
        charging_features.geometry.geom_type.unique()
    )

    assert geometry_types == {"Polygon"}


def test_numeric_charging_features_are_complete(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging model columns must not contain missing values."""

    for column in NUMERIC_CHARGING_COLUMNS:
        missing_count = int(
            charging_features[column]
            .isna()
            .sum()
        )

        assert missing_count == 0, (
            f"{column} contains "
            f"{missing_count} missing values."
        )


def test_numeric_charging_features_are_finite(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging features must contain finite values."""

    for column in NUMERIC_CHARGING_COLUMNS:
        values = charging_features[
            column
        ].to_numpy(dtype=float)

        assert np.isfinite(values).all(), (
            f"{column} contains non-finite values."
        )


def test_numeric_charging_features_are_non_negative(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Counts, capacities and distances cannot be negative."""

    for column in NUMERIC_CHARGING_COLUMNS:
        values = charging_features[
            column
        ].to_numpy(dtype=float)

        assert (values >= 0).all(), (
            f"{column} contains negative values."
        )


def test_charging_count_columns_are_whole_numbers(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging count and target columns must be integer-like."""

    for column in INTEGER_CHARGING_COLUMNS:
        values = charging_features[
            column
        ].to_numpy(dtype=float)

        assert np.allclose(
            values,
            np.round(values),
        ), f"{column} contains fractional values."


def test_existing_station_target_is_binary(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """The station-presence target must contain only zero or one."""

    target_values = set(
        charging_features[
            "has_existing_charging_station"
        ]
        .astype(int)
        .unique()
    )

    assert target_values.issubset(
        {
            0,
            1,
        }
    )


def test_existing_station_target_matches_local_count(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """The target must equal whether local count is positive."""

    expected_target = (
        charging_features[
            "charging_station_count"
        ]
        > 0
    ).astype(int)

    stored_target = charging_features[
        "has_existing_charging_station"
    ].astype(int)

    assert expected_target.equals(
        stored_target
    )


def test_radius_station_counts_are_monotonic(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """A 1 km count cannot exceed its 2 km count."""

    assert (
        charging_features[
            "charging_station_count_within_1000m"
        ]
        <= charging_features[
            "charging_station_count_within_2000m"
        ]
    ).all()


def test_local_station_count_does_not_exceed_1km_count(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """A local station must also fall within 1 km."""

    assert (
        charging_features[
            "charging_station_count"
        ]
        <= charging_features[
            "charging_station_count_within_1000m"
        ]
    ).all()


def test_ac_and_dc_counts_do_not_exceed_total_1km_count(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Connector-specific counts are subsets of all stations."""

    total_counts = charging_features[
        "charging_station_count_within_1000m"
    ]

    assert (
        charging_features[
            "ac_station_count_within_1000m"
        ]
        <= total_counts
    ).all()

    assert (
        charging_features[
            "dc_station_count_within_1000m"
        ]
        <= total_counts
    ).all()


def test_capacity_record_count_does_not_exceed_local_count(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Capacity records are a subset of local stations."""

    assert (
        charging_features[
            "charging_capacity_record_count"
        ]
        <= charging_features[
            "charging_station_count"
        ]
    ).all()


def test_zero_capacity_record_cells_have_zero_capacity(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Cells without capacity records must store zero capacity."""

    no_capacity_cells = charging_features[
        charging_features[
            "charging_capacity_record_count"
        ]
        == 0
    ]

    assert (
        no_capacity_cells[
            "known_charging_capacity"
        ]
        == 0
    ).all()


def test_dataset_contains_positive_and_negative_targets(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """The target must contain station and non-station cells."""

    target = charging_features[
        "has_existing_charging_station"
    ].astype(int)

    assert (target == 1).any()
    assert (target == 0).any()


def test_nearest_station_distances_are_complete(
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Every grid cell must have a nearest-station distance."""

    distances = charging_features[
        "distance_to_nearest_charging_station_m"
    ].to_numpy(dtype=float)

    assert np.isfinite(distances).all()
    assert (distances >= 0).all()


def test_local_station_features_match_spatial_assignment(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Local station count and capacity must match GIS assignment."""

    station_points = create_station_points(
        charging_stations
    )

    grid_cells = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    assignments = gpd.sjoin(
        station_points,
        grid_cells,
        how="inner",
        predicate="intersects",
    )

    assignments = assignments.sort_values(
        by=[
            "station_id",
            "grid_id",
        ]
    )

    assignments = assignments.drop_duplicates(
        subset=["station_id"],
        keep="first",
    )

    grouped = assignments.groupby(
        "grid_id"
    ).agg(
        expected_count=(
            "station_id",
            "nunique",
        ),
        expected_capacity=(
            "capacity_numeric",
            "sum",
        ),
        expected_capacity_records=(
            "capacity_numeric",
            "count",
        ),
    )

    expected_count = (
        charging_features["grid_id"]
        .map(grouped["expected_count"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    expected_capacity = (
        charging_features["grid_id"]
        .map(grouped["expected_capacity"])
        .fillna(0)
        .astype(float)
        .to_numpy()
    )

    expected_capacity_records = (
        charging_features["grid_id"]
        .map(
            grouped[
                "expected_capacity_records"
            ]
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    assert np.array_equal(
        charging_features[
            "charging_station_count"
        ].astype(int).to_numpy(),
        expected_count,
    )

    assert np.allclose(
        charging_features[
            "known_charging_capacity"
        ].astype(float).to_numpy(),
        expected_capacity,
        atol=0.01,
    )

    assert np.array_equal(
        charging_features[
            "charging_capacity_record_count"
        ].astype(int).to_numpy(),
        expected_capacity_records,
    )


@pytest.mark.parametrize(
    (
        "radius_meters",
        "stored_column",
    ),
    [
        (
            1_000,
            "charging_station_count_within_1000m",
        ),
        (
            2_000,
            "charging_station_count_within_2000m",
        ),
    ],
)
def test_radius_counts_match_spatial_calculation(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
    radius_meters: int,
    stored_column: str,
) -> None:
    """Stored radius counts must match GIS calculations."""

    station_points = create_station_points(
        charging_stations
    )

    expected_counts = (
        calculate_expected_radius_counts(
            base_features,
            station_points,
            radius_meters,
        )
    )

    expected = (
        charging_features["grid_id"]
        .map(expected_counts)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    stored = (
        charging_features[stored_column]
        .astype(int)
        .to_numpy()
    )

    assert np.array_equal(
        stored,
        expected,
    )


def test_ac_radius_counts_match_spatial_calculation(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """AC station counts must match AC-tagged station points."""

    station_points = create_station_points(
        charging_stations
    )

    ac_points = station_points[
        station_points["has_ac_connector"]
    ].copy()

    expected_counts = (
        calculate_expected_radius_counts(
            base_features,
            ac_points,
            1_000,
        )
    )

    expected = (
        charging_features["grid_id"]
        .map(expected_counts)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    stored = charging_features[
        "ac_station_count_within_1000m"
    ].astype(int).to_numpy()

    assert np.array_equal(
        stored,
        expected,
    )


def test_dc_radius_counts_match_spatial_calculation(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """DC station counts must match DC-tagged station points."""

    station_points = create_station_points(
        charging_stations
    )

    dc_points = station_points[
        station_points["has_dc_connector"]
    ].copy()

    expected_counts = (
        calculate_expected_radius_counts(
            base_features,
            dc_points,
            1_000,
        )
    )

    expected = (
        charging_features["grid_id"]
        .map(expected_counts)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    stored = charging_features[
        "dc_station_count_within_1000m"
    ].astype(int).to_numpy()

    assert np.array_equal(
        stored,
        expected,
    )


def test_nearest_distances_match_spatial_calculation(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Stored nearest distances must match GIS calculations."""

    grid_centers = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_centers["geometry"] = (
        grid_centers.geometry.centroid
    )

    nearest = gpd.sjoin_nearest(
        grid_centers,
        charging_stations[
            [
                "station_id",
                "geometry",
            ]
        ],
        how="left",
        distance_col="expected_distance",
    )

    nearest = nearest.sort_values(
        by=[
            "grid_id",
            "expected_distance",
            "station_id",
        ],
        na_position="last",
    )

    nearest = nearest.drop_duplicates(
        subset=["grid_id"],
        keep="first",
    )

    expected_by_grid = nearest.set_index(
        "grid_id"
    )["expected_distance"]

    expected = (
        charging_features["grid_id"]
        .map(expected_by_grid)
        .astype(float)
        .to_numpy()
    )

    stored = charging_features[
        "distance_to_nearest_charging_station_m"
    ].astype(float).to_numpy()

    assert np.allclose(
        stored,
        expected,
        atol=DISTANCE_TOLERANCE_METERS,
    )


def test_previous_features_are_unchanged(
    base_features: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Adding charging columns must not alter earlier features."""

    comparison_columns = [
        "grid_id",
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
        "road_density_km_per_km2",
        "distance_to_main_road_m",
        "parking_count",
        "parking_area_m2",
        "parking_area_ratio",
        "distance_to_nearest_parking_m",
        "parking_count_within_500m",
        "parking_count_within_1000m",
        "known_parking_capacity",
        "parking_capacity_record_count",
    ]

    base_sorted = (
        base_features[comparison_columns]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    output_sorted = (
        charging_features[comparison_columns]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    assert base_sorted["grid_id"].equals(
        output_sorted["grid_id"]
    )

    for column in comparison_columns:
        if column == "grid_id":
            continue

        assert np.allclose(
            base_sorted[column].to_numpy(
                dtype=float
            ),
            output_sorted[column].to_numpy(
                dtype=float
            ),
            atol=0.0001,
        ), f"Previous feature changed: {column}"


def test_grid_geometries_are_unchanged(
    base_features: gpd.GeoDataFrame,
    charging_features: gpd.GeoDataFrame,
) -> None:
    """Charging engineering must preserve grid geometry."""

    base_geometry = (
        base_features[
            [
                "grid_id",
                "geometry",
            ]
        ]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    output_geometry = (
        charging_features[
            [
                "grid_id",
                "geometry",
            ]
        ]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    assert np.array_equal(
        base_geometry.geometry.to_wkb(),
        output_geometry.geometry.to_wkb(),
    )


def test_csv_row_count_matches_geospatial_output(
    charging_features: gpd.GeoDataFrame,
    charging_features_csv: pd.DataFrame,
) -> None:
    """CSV output must have one row per grid cell."""

    assert len(charging_features_csv) == len(
        charging_features
    )


def test_csv_grid_ids_match_geospatial_output(
    charging_features: gpd.GeoDataFrame,
    charging_features_csv: pd.DataFrame,
) -> None:
    """CSV and GeoPackage must contain the same grid IDs."""

    geospatial_ids = set(
        charging_features["grid_id"].astype(str)
    )

    csv_ids = set(
        charging_features_csv["grid_id"].astype(str)
    )

    assert csv_ids == geospatial_ids


def test_csv_contains_charging_model_columns(
    charging_features_csv: pd.DataFrame,
) -> None:
    """The machine-learning CSV must include charging columns."""

    required_csv_columns = {
        "grid_id",
        "charging_station_count",
        "has_existing_charging_station",
        "distance_to_nearest_charging_station_m",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "known_charging_capacity",
        "charging_capacity_record_count",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    }

    missing_columns = (
        required_csv_columns
        - set(charging_features_csv.columns)
    )

    assert not missing_columns, (
        "Missing charging CSV columns: "
        f"{sorted(missing_columns)}"
    )


def test_csv_does_not_contain_geometry(
    charging_features_csv: pd.DataFrame,
) -> None:
    """Tabular machine-learning output must omit geometry."""

    assert (
        "geometry"
        not in charging_features_csv.columns
    )


def test_documentation_outputs_exist() -> None:
    """The pipeline must produce preview and summary files."""

    assert CHARGING_PREVIEW_PATH.exists()
    assert CHARGING_PREVIEW_PATH.stat().st_size > 0

    assert CHARGING_SUMMARY_PATH.exists()
    assert CHARGING_SUMMARY_PATH.stat().st_size > 0