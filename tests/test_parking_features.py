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
    / "cankaya_grid_road_features.gpkg"
)

PARKING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_parking_features.gpkg"
)

PARKING_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.gpkg"
)

PARKING_FEATURES_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.csv"
)

PARKING_PREVIEW_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_parking_features_preview.png"
)

PARKING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_parking_features_summary.md"
)

BASE_FEATURE_LAYER = "grid_road_features"
PARKING_DATA_LAYER = "parking_features"
PARKING_FEATURE_LAYER = "grid_parking_features"

AREA_TOLERANCE_M2 = 0.05
RATIO_TOLERANCE = 0.000001

SUPPORTED_PARKING_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
}

POLYGON_GEOMETRY_TYPES = {
    "Polygon",
    "MultiPolygon",
}

REQUIRED_PARKING_DATA_COLUMNS = {
    "parking_id",
    "osm_element_type",
    "osm_id",
    "name",
    "parking",
    "access",
    "fee",
    "capacity",
    "capacity_numeric",
    "operator",
    "surface",
    "covered",
    "supervised",
    "geometry_type",
    "parking_area_m2",
    "geometry",
}

REQUIRED_FEATURE_COLUMNS = {
    "grid_id",
    "cell_area_m2",
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
    "geometry",
}

REQUIRED_ROAD_COLUMNS = {
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
}

NUMERIC_PARKING_FEATURE_COLUMNS = [
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
]

INTEGER_FEATURE_COLUMNS = [
    "parking_count",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "parking_capacity_record_count",
]


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail with a useful message when a pipeline output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run create_study_grid.py, create_road_features.py and "
            "create_parking_features.py before executing these tests."
        )


@pytest.fixture(scope="session")
def base_features() -> gpd.GeoDataFrame:
    """Load the existing road-feature grid."""

    ensure_file_exists(
        BASE_FEATURES_PATH,
        "Road-feature GeoPackage",
    )

    return gpd.read_file(
        BASE_FEATURES_PATH,
        layer=BASE_FEATURE_LAYER,
    )


@pytest.fixture(scope="session")
def parking_data() -> gpd.GeoDataFrame:
    """Load the cleaned OpenStreetMap parking dataset."""

    ensure_file_exists(
        PARKING_DATA_PATH,
        "Parking GeoPackage",
    )

    return gpd.read_file(
        PARKING_DATA_PATH,
        layer=PARKING_DATA_LAYER,
    )


@pytest.fixture(scope="session")
def parking_features() -> gpd.GeoDataFrame:
    """Load the grid-level parking feature dataset."""

    ensure_file_exists(
        PARKING_FEATURES_PATH,
        "Parking-feature GeoPackage",
    )

    return gpd.read_file(
        PARKING_FEATURES_PATH,
        layer=PARKING_FEATURE_LAYER,
    )


@pytest.fixture(scope="session")
def parking_features_csv() -> pd.DataFrame:
    """Load the machine-learning compatible parking CSV."""

    ensure_file_exists(
        PARKING_FEATURES_CSV_PATH,
        "Parking-feature CSV",
    )

    return pd.read_csv(
        PARKING_FEATURES_CSV_PATH,
        encoding="utf-8",
    )


def test_parking_data_is_not_empty(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """The downloaded parking dataset must contain records."""

    assert not parking_data.empty
    assert len(parking_data) > 0


def test_parking_data_contains_required_columns(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """The cleaned parking schema must be complete."""

    missing_columns = (
        REQUIRED_PARKING_DATA_COLUMNS
        - set(parking_data.columns)
    )

    assert not missing_columns, (
        "Missing parking-data columns: "
        f"{sorted(missing_columns)}"
    )


def test_parking_ids_are_complete_and_unique(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Every parking feature must have one unique identifier."""

    parking_ids = (
        parking_data["parking_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert parking_ids.ne("").all()
    assert not parking_ids.duplicated().any()


def test_parking_geometries_are_complete(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Parking records must contain non-empty geometries."""

    assert parking_data.geometry.notna().all()
    assert not parking_data.geometry.is_empty.any()


def test_parking_geometries_are_valid(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """All cleaned parking geometries must be valid."""

    invalid_count = int(
        (~parking_data.geometry.is_valid).sum()
    )

    assert invalid_count == 0, (
        f"Invalid parking geometry count: {invalid_count}"
    )


def test_parking_geometry_types_are_supported(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """The pipeline must retain only supported geometry types."""

    geometry_types = set(
        parking_data.geometry.geom_type.unique()
    )

    assert geometry_types.issubset(
        SUPPORTED_PARKING_GEOMETRY_TYPES
    )


def test_stored_geometry_type_matches_geometry(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """The geometry_type field must match the actual geometry."""

    stored_types = (
        parking_data["geometry_type"]
        .astype(str)
        .to_numpy()
    )

    calculated_types = (
        parking_data.geometry.geom_type
        .astype(str)
        .to_numpy()
    )

    assert np.array_equal(
        stored_types,
        calculated_types,
    )


def test_parking_crs_is_projected(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Metric parking calculations must use a projected CRS."""

    assert parking_data.crs is not None
    assert parking_data.crs.is_projected
    assert not parking_data.crs.is_geographic


def test_parking_crs_matches_base_features(
    base_features: gpd.GeoDataFrame,
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Parking data and analysis grid must use the same CRS."""

    assert base_features.crs is not None
    assert parking_data.crs == base_features.crs


def test_parking_polygon_areas_match_geometry(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Polygon parking areas must match their geometric areas."""

    polygon_mask = (
        parking_data.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
        )
    )

    polygon_parking = parking_data.loc[
        polygon_mask
    ]

    assert not polygon_parking.empty

    stored_areas = (
        polygon_parking["parking_area_m2"]
        .astype(float)
        .to_numpy()
    )

    calculated_areas = (
        polygon_parking.geometry.area
        .to_numpy()
    )

    assert np.allclose(
        stored_areas,
        calculated_areas,
        atol=AREA_TOLERANCE_M2,
    )


def test_non_polygon_parking_has_zero_area(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Point and line parking records must not have polygon area."""

    non_polygon_parking = parking_data.loc[
        ~parking_data.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
        )
    ]

    if non_polygon_parking.empty:
        pytest.skip(
            "No non-polygon parking geometries were downloaded."
        )

    assert (
        non_polygon_parking["parking_area_m2"]
        .astype(float)
        == 0
    ).all()


def test_numeric_capacities_are_non_negative(
    parking_data: gpd.GeoDataFrame,
) -> None:
    """Known numeric capacities cannot be negative."""

    known_capacities = (
        parking_data["capacity_numeric"]
        .dropna()
        .astype(float)
    )

    if known_capacities.empty:
        pytest.skip(
            "No numeric parking-capacity values were available."
        )

    assert np.isfinite(
        known_capacities.to_numpy()
    ).all()

    assert (
        known_capacities >= 0
    ).all()


def test_parking_feature_dataset_is_not_empty(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """The grid-level feature dataset must contain rows."""

    assert not parking_features.empty
    assert len(parking_features) > 0


def test_parking_feature_row_count_matches_base_grid(
    base_features: gpd.GeoDataFrame,
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Every base grid cell must have one parking-feature record."""

    assert len(parking_features) == len(
        base_features
    )


def test_parking_features_contain_required_columns(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """The feature output must contain all parking columns."""

    missing_columns = (
        REQUIRED_FEATURE_COLUMNS
        - set(parking_features.columns)
    )

    assert not missing_columns, (
        "Missing parking-feature columns: "
        f"{sorted(missing_columns)}"
    )


def test_road_columns_are_preserved(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking engineering must preserve previous road features."""

    missing_columns = (
        REQUIRED_ROAD_COLUMNS
        - set(parking_features.columns)
    )

    assert not missing_columns, (
        "Missing previously generated road columns: "
        f"{sorted(missing_columns)}"
    )


def test_parking_feature_crs_matches_base_grid(
    base_features: gpd.GeoDataFrame,
    parking_features: gpd.GeoDataFrame,
) -> None:
    """The output must preserve the projected analysis CRS."""

    assert parking_features.crs is not None
    assert parking_features.crs.is_projected
    assert parking_features.crs == base_features.crs


def test_parking_feature_grid_ids_are_unique(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Grid identifiers must remain complete and unique."""

    grid_ids = (
        parking_features["grid_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert grid_ids.ne("").all()
    assert not grid_ids.duplicated().any()


def test_parking_feature_grid_ids_match_base_grid(
    base_features: gpd.GeoDataFrame,
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking output must cover exactly the original grid IDs."""

    base_ids = set(
        base_features["grid_id"].astype(str)
    )

    output_ids = set(
        parking_features["grid_id"].astype(str)
    )

    assert output_ids == base_ids


def test_parking_feature_geometries_are_complete(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Every parking-feature record must retain grid geometry."""

    assert parking_features.geometry.notna().all()
    assert not parking_features.geometry.is_empty.any()


def test_parking_feature_geometries_are_valid_polygons(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Output geometries must remain valid grid polygons."""

    assert parking_features.geometry.is_valid.all()

    geometry_types = set(
        parking_features.geometry.geom_type.unique()
    )

    assert geometry_types == {"Polygon"}


def test_numeric_parking_features_are_complete(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking model features must not contain missing values."""

    for column in NUMERIC_PARKING_FEATURE_COLUMNS:
        missing_count = int(
            parking_features[column].isna().sum()
        )

        assert missing_count == 0, (
            f"{column} contains "
            f"{missing_count} missing values."
        )


def test_numeric_parking_features_are_finite(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking features must contain only finite numbers."""

    for column in NUMERIC_PARKING_FEATURE_COLUMNS:
        values = parking_features[column].to_numpy(
            dtype=float
        )

        assert np.isfinite(values).all(), (
            f"{column} contains non-finite values."
        )


def test_numeric_parking_features_are_non_negative(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Counts, areas, capacities and distances cannot be negative."""

    for column in NUMERIC_PARKING_FEATURE_COLUMNS:
        values = parking_features[column].to_numpy(
            dtype=float
        )

        assert (values >= 0).all(), (
            f"{column} contains negative values."
        )


def test_parking_count_columns_are_whole_numbers(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking count columns must contain integer-like values."""

    for column in INTEGER_FEATURE_COLUMNS:
        values = parking_features[column].to_numpy(
            dtype=float
        )

        assert np.allclose(
            values,
            np.round(values),
        ), f"{column} contains fractional values."


def test_radius_counts_are_monotonic(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """A 500 m count cannot exceed its corresponding 1,000 m count."""

    assert (
        parking_features[
            "parking_count_within_500m"
        ]
        <= parking_features[
            "parking_count_within_1000m"
        ]
    ).all()


def test_local_parking_count_does_not_exceed_500m_count(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking assigned to a cell must also be within 500 m."""

    assert (
        parking_features["parking_count"]
        <= parking_features[
            "parking_count_within_500m"
        ]
    ).all()


def test_capacity_record_count_does_not_exceed_parking_count(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Known-capacity records are a subset of local parking records."""

    assert (
        parking_features[
            "parking_capacity_record_count"
        ]
        <= parking_features[
            "parking_count"
        ]
    ).all()


def test_zero_capacity_record_cells_have_zero_known_capacity(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Cells without capacity records must store zero known capacity."""

    no_capacity_records = parking_features[
        parking_features[
            "parking_capacity_record_count"
        ]
        == 0
    ]

    assert (
        no_capacity_records[
            "known_parking_capacity"
        ]
        == 0
    ).all()


def test_parking_area_does_not_exceed_cell_area(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Clipped parking area cannot exceed the grid-cell area."""

    assert (
        parking_features["parking_area_m2"]
        <= (
            parking_features["cell_area_m2"]
            + AREA_TOLERANCE_M2
        )
    ).all()


def test_parking_area_ratio_is_in_valid_range(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Parking area ratio must be between zero and one."""

    ratios = parking_features[
        "parking_area_ratio"
    ].astype(float)

    assert (
        ratios >= 0
    ).all()

    assert (
        ratios
        <= 1 + RATIO_TOLERANCE
    ).all()


def test_parking_area_ratio_matches_area_and_cell_size(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Stored area ratio must equal parking area divided by cell area."""

    expected_ratio = (
        parking_features["parking_area_m2"]
        / parking_features["cell_area_m2"]
    )

    stored_ratio = parking_features[
        "parking_area_ratio"
    ]

    assert np.allclose(
        stored_ratio.to_numpy(dtype=float),
        expected_ratio.to_numpy(dtype=float),
        atol=RATIO_TOLERANCE,
    )


def test_nearest_parking_distances_are_complete(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Every grid cell must be matched to a parking geometry."""

    distances = parking_features[
        "distance_to_nearest_parking_m"
    ].to_numpy(dtype=float)

    assert np.isfinite(distances).all()
    assert (distances >= 0).all()


def test_dataset_contains_cells_with_local_parking(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """At least one grid cell must contain mapped parking."""

    local_parking_cells = int(
        (
            parking_features["parking_count"]
            > 0
        ).sum()
    )

    assert local_parking_cells > 0


def test_dataset_contains_parking_accessible_cells(
    parking_features: gpd.GeoDataFrame,
) -> None:
    """At least one grid cell must have parking within each radius."""

    cells_within_500m = int(
        (
            parking_features[
                "parking_count_within_500m"
            ]
            > 0
        ).sum()
    )

    cells_within_1000m = int(
        (
            parking_features[
                "parking_count_within_1000m"
            ]
            > 0
        ).sum()
    )

    assert cells_within_500m > 0
    assert cells_within_1000m > 0
    assert cells_within_1000m >= cells_within_500m


def test_local_parking_counts_match_spatial_assignment(
    base_features: gpd.GeoDataFrame,
    parking_data: gpd.GeoDataFrame,
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Stored local counts must match representative-point assignment."""

    parking_points = parking_data[
        [
            "parking_id",
            "geometry",
        ]
    ].copy()

    parking_points["geometry"] = (
        parking_points.geometry.representative_point()
    )

    grid_cells = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    assignments = gpd.sjoin(
        parking_points,
        grid_cells,
        how="inner",
        predicate="intersects",
    )

    assignments = assignments.sort_values(
        by=[
            "parking_id",
            "grid_id",
        ]
    )

    assignments = assignments.drop_duplicates(
        subset=["parking_id"],
        keep="first",
    )

    expected_counts = (
        assignments.groupby("grid_id")[
            "parking_id"
        ]
        .nunique()
    )

    expected = (
        parking_features["grid_id"]
        .map(expected_counts)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    stored = (
        parking_features["parking_count"]
        .astype(int)
        .to_numpy()
    )

    assert np.array_equal(
        stored,
        expected,
    )


def test_previous_road_values_are_unchanged(
    base_features: gpd.GeoDataFrame,
    parking_features: gpd.GeoDataFrame,
) -> None:
    """Adding parking features must not modify road measurements."""

    comparison_columns = [
        "grid_id",
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
        "road_density_km_per_km2",
        "distance_to_main_road_m",
    ]

    base_sorted = (
        base_features[comparison_columns]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    output_sorted = (
        parking_features[comparison_columns]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    assert base_sorted["grid_id"].equals(
        output_sorted["grid_id"]
    )

    numeric_columns = [
        column
        for column in comparison_columns
        if column != "grid_id"
    ]

    for column in numeric_columns:
        assert np.allclose(
            base_sorted[column].to_numpy(
                dtype=float
            ),
            output_sorted[column].to_numpy(
                dtype=float
            ),
            atol=0.0001,
        ), f"Previous road feature changed: {column}"


def test_csv_row_count_matches_geospatial_output(
    parking_features: gpd.GeoDataFrame,
    parking_features_csv: pd.DataFrame,
) -> None:
    """CSV output must contain one row per grid cell."""

    assert len(parking_features_csv) == len(
        parking_features
    )


def test_csv_grid_ids_match_geospatial_output(
    parking_features: gpd.GeoDataFrame,
    parking_features_csv: pd.DataFrame,
) -> None:
    """CSV and GeoPackage must describe the same grid cells."""

    geospatial_ids = set(
        parking_features["grid_id"].astype(str)
    )

    csv_ids = set(
        parking_features_csv["grid_id"].astype(str)
    )

    assert csv_ids == geospatial_ids


def test_csv_contains_parking_model_features(
    parking_features_csv: pd.DataFrame,
) -> None:
    """The machine-learning CSV must contain parking features."""

    required_csv_columns = {
        "grid_id",
        "parking_count",
        "parking_area_m2",
        "parking_area_ratio",
        "distance_to_nearest_parking_m",
        "parking_count_within_500m",
        "parking_count_within_1000m",
        "known_parking_capacity",
        "parking_capacity_record_count",
    }

    missing_columns = (
        required_csv_columns
        - set(parking_features_csv.columns)
    )

    assert not missing_columns, (
        "Missing parking CSV columns: "
        f"{sorted(missing_columns)}"
    )


def test_csv_does_not_contain_geometry_column(
    parking_features_csv: pd.DataFrame,
) -> None:
    """The tabular machine-learning output must omit geometry."""

    assert "geometry" not in parking_features_csv.columns


def test_documentation_outputs_exist() -> None:
    """The parking pipeline must produce preview and summary files."""

    assert PARKING_PREVIEW_PATH.exists()
    assert PARKING_PREVIEW_PATH.stat().st_size > 0

    assert PARKING_SUMMARY_PATH.exists()
    assert PARKING_SUMMARY_PATH.stat().st_size > 0