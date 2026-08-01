from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cankaya_boundary_osm.geojson"
)

BASE_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.gpkg"
)

PARKING_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_parking_features.gpkg"
)

FEATURE_GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.gpkg"
)

FEATURE_GEOJSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.geojson"
)

FEATURE_CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_parking_features.csv"
)

PREVIEW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_parking_features_preview.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_parking_features_summary.md"
)

CACHE_DIRECTORY = PROJECT_ROOT / "cache"

BASE_FEATURE_LAYER_NAME = "grid_road_features"
PARKING_LAYER_NAME = "parking_features"
FEATURE_LAYER_NAME = "grid_parking_features"

DOWNLOAD_BUFFER_METERS = 1_000
PARKING_COUNT_RADII_METERS = (500, 1_000)

OVERPASS_REQUEST_TIMEOUT_SECONDS = 600

# 100 square kilometers.
# OSMnx divides larger polygons into multiple Overpass requests.
MAX_QUERY_AREA_SIZE_M2 = 100_000_000

OVERPASS_ENDPOINTS = (
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass-api.de/api",
)

PARKING_TAGS = {
    "amenity": "parking",
}

SUPPORTED_GEOMETRY_TYPES = {
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

OPTIONAL_OSM_COLUMNS = (
    "name",
    "parking",
    "access",
    "fee",
    "capacity",
    "operator",
    "surface",
    "covered",
    "supervised",
)

PARKING_FEATURE_COLUMNS = (
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
)


def create_output_directories() -> None:
    """Create directories required by the parking pipeline."""

    directories = (
        PARKING_OUTPUT_PATH.parent,
        FEATURE_GPKG_OUTPUT_PATH.parent,
        PREVIEW_OUTPUT_PATH.parent,
        CACHE_DIRECTORY,
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_input_files() -> None:
    """Ensure that the boundary and road-feature files exist."""

    required_paths = (
        BOUNDARY_PATH,
        BASE_FEATURES_PATH,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if not missing_paths:
        return

    formatted_paths = "\n".join(
        f"- {path}"
        for path in missing_paths
    )

    raise FileNotFoundError(
        "Required input files are missing.\n"
        "Run create_study_grid.py and "
        "create_road_features.py first.\n"
        f"{formatted_paths}"
    )


def configure_osmnx() -> None:
    """Configure OSMnx caching, logging and Overpass behavior."""

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(
        CACHE_DIRECTORY
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
        "VoltSight/0.1 "
        "educational-geospatial-research-project "
        "(https://github.com/ilginbor/voltsight-ai)"
    )

    ox.settings.http_referer = (
        "https://github.com/ilginbor/voltsight-ai"
    )

    print(
        "Overpass request timeout: "
        f"{ox.settings.requests_timeout} seconds"
    )

    print(
        "Maximum Overpass query part area: "
        f"{ox.settings.max_query_area_size / 1_000_000:,.0f} km2"
    )


def load_study_data() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load the district boundary and grid-road features."""

    boundary = gpd.read_file(
        BOUNDARY_PATH
    )

    base_features = gpd.read_file(
        BASE_FEATURES_PATH,
        layer=BASE_FEATURE_LAYER_NAME,
    )

    if boundary.empty:
        raise ValueError(
            "The Cankaya boundary dataset is empty."
        )

    if base_features.empty:
        raise ValueError(
            "The grid-road feature dataset is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "The boundary coordinate system is missing."
        )

    if base_features.crs is None:
        raise ValueError(
            "The base feature coordinate system is missing."
        )

    if not base_features.crs.is_projected:
        raise ValueError(
            "The base feature dataset must use "
            "a projected coordinate system."
        )

    required_columns = {
        "grid_id",
        "cell_area_m2",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(base_features.columns)
    )

    if missing_columns:
        raise ValueError(
            "The base feature dataset is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        base_features["grid_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate grid identifiers were found."
        )

    boundary = boundary.to_crs(
        base_features.crs
    )

    print(
        f"Loaded grid cell count: "
        f"{len(base_features):,}"
    )

    print(
        "Analysis coordinate system: "
        f"{base_features.crs}"
    )

    return boundary, base_features


def create_download_polygon(
    boundary_projected: gpd.GeoDataFrame,
) -> Any:
    """Create a buffered EPSG:4326 polygon."""

    district_geometry = (
        boundary_projected
        .geometry
        .union_all()
    )

    buffered_geometry = (
        district_geometry.buffer(
            DOWNLOAD_BUFFER_METERS
        )
    )

    buffered_boundary = gpd.GeoDataFrame(
        {
            "name": [
                "Cankaya parking download area"
            ],
        },
        geometry=[buffered_geometry],
        crs=boundary_projected.crs,
    )

    buffered_boundary_wgs84 = (
        buffered_boundary.to_crs(
            epsg=4326
        )
    )

    return (
        buffered_boundary_wgs84
        .geometry
        .iloc[0]
    )


def download_parking_features(
    download_polygon: Any,
) -> gpd.GeoDataFrame:
    """Download parking features using endpoint fallbacks."""

    endpoint_errors: list[str] = []

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint

        print("-" * 70)

        print(
            f"Trying Overpass endpoint: "
            f"{endpoint}"
        )

        print(
            "Downloading parking features "
            "from OpenStreetMap..."
        )

        try:
            parking = (
                ox.features.features_from_polygon(
                    download_polygon,
                    tags=PARKING_TAGS,
                )
            )

            if parking.empty:
                raise RuntimeError(
                    "The query returned "
                    "no parking features."
                )

            print(
                f"Successful endpoint: "
                f"{endpoint}"
            )

            print(
                "Downloaded OSM parking records: "
                f"{len(parking):,}"
            )

            return parking

        except Exception as error:
            error_message = (
                f"{endpoint}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            endpoint_errors.append(
                error_message
            )

            print(
                f"Endpoint failed: "
                f"{error_message}"
            )

            print(
                "Trying the next "
                "Overpass endpoint..."
            )

    error_details = "\n".join(
        f"- {message}"
        for message in endpoint_errors
    )

    raise RuntimeError(
        "Parking features could not be downloaded "
        "from any configured Overpass endpoint.\n"
        "Check the internet connection and try again later.\n"
        f"Attempt details:\n{error_details}"
    )


def normalize_tag_values(
    value: Any,
) -> list[str]:
    """Convert OSM tag values into clean strings."""

    if value is None:
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            np.ndarray,
        ),
    ):
        normalized_values: list[str] = []

        for item in value:
            normalized_values.extend(
                normalize_tag_values(item)
            )

        return list(
            dict.fromkeys(
                normalized_values
            )
        )

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text_value = str(value).strip()

    if not text_value:
        return []

    return [text_value]


def stringify_tag_value(
    value: Any,
) -> str:
    """Convert an OSM tag into semicolon-separated text."""

    return ";".join(
        normalize_tag_values(value)
    )


def find_column(
    columns: pd.Index,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    """Find the first available candidate column."""

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not identify the {description} column. "
        f"Available columns: {list(columns)}"
    )


def parse_capacity(
    value: Any,
) -> float:
    """Parse a simple numeric parking-capacity value."""

    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass

    text_value = (
        str(value)
        .strip()
        .replace(",", ".")
    )

    if not text_value:
        return np.nan

    try:
        numeric_value = float(
            text_value
        )
    except ValueError:
        return np.nan

    if (
        not np.isfinite(numeric_value)
        or numeric_value < 0
    ):
        return np.nan

    return numeric_value


def prepare_parking_features(
    raw_parking: gpd.GeoDataFrame,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Clean, normalize and project OSM parking data."""

    if raw_parking.crs is None:
        raise ValueError(
            "Downloaded parking features have no CRS."
        )

    parking = (
        raw_parking
        .reset_index()
        .copy()
    )

    element_column = find_column(
        parking.columns,
        (
            "element",
            "element_type",
            "type",
        ),
        "OSM element type",
    )

    osm_id_column = find_column(
        parking.columns,
        (
            "id",
            "osmid",
            "osm_id",
        ),
        "OSM identifier",
    )

    parking.rename(
        columns={
            element_column: (
                "osm_element_type"
            ),
            osm_id_column: "osm_id",
        },
        inplace=True,
    )

    parking = parking[
        parking.geometry.notna()
    ].copy()

    parking = parking[
        ~parking.geometry.is_empty
    ].copy()

    parking = parking[
        parking.geometry.geom_type.isin(
            SUPPORTED_GEOMETRY_TYPES
        )
    ].copy()

    if parking.empty:
        raise ValueError(
            "No supported parking geometries "
            "remained after cleaning."
        )

    parking[
        "osm_element_type"
    ] = parking[
        "osm_element_type"
    ].astype(str)

    parking[
        "osm_id"
    ] = parking[
        "osm_id"
    ].astype(str)

    parking["parking_id"] = (
        parking["osm_element_type"]
        + "_"
        + parking["osm_id"]
    )

    parking.drop_duplicates(
        subset=["parking_id"],
        keep="first",
        inplace=True,
    )

    for column in OPTIONAL_OSM_COLUMNS:
        if column not in parking.columns:
            parking[column] = ""

        parking[column] = (
            parking[column].apply(
                stringify_tag_value
            )
        )

    parking[
        "capacity_numeric"
    ] = parking[
        "capacity"
    ].apply(
        parse_capacity
    )

    parking = parking.to_crs(
        target_crs
    )

    parking[
        "geometry_type"
    ] = parking.geometry.geom_type

    polygon_mask = (
        parking.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
        )
    )

    parking[
        "parking_area_m2"
    ] = 0.0

    parking.loc[
        polygon_mask,
        "parking_area_m2",
    ] = (
        parking.loc[
            polygon_mask
        ].geometry.area
    )

    parking[
        "parking_area_m2"
    ] = parking[
        "parking_area_m2"
    ].round(2)

    selected_columns = [
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
    ]

    parking = parking[
        selected_columns
    ].copy()

    parking.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        "Prepared unique parking features: "
        f"{len(parking):,}"
    )

    geometry_counts = parking[
        "geometry_type"
    ].value_counts()

    for (
        geometry_type,
        count,
    ) in geometry_counts.items():
        print(
            f"Parking {geometry_type} records: "
            f"{int(count):,}"
        )

    print(
        "Parking records with known capacity: "
        f"{int(parking['capacity_numeric'].notna().sum()):,}"
    )

    return gpd.GeoDataFrame(
        parking,
        geometry="geometry",
        crs=target_crs,
    )


def save_parking_features(
    parking: gpd.GeoDataFrame,
) -> None:
    """Save the cleaned parking dataset."""

    if PARKING_OUTPUT_PATH.exists():
        PARKING_OUTPUT_PATH.unlink()

    parking.to_file(
        PARKING_OUTPUT_PATH,
        layer=PARKING_LAYER_NAME,
        driver="GPKG",
    )

    print(
        "Parking features saved: "
        f"{PARKING_OUTPUT_PATH}"
    )


def create_parking_points(
    parking: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create one representative point per parking feature."""

    parking_points = parking[
        [
            "parking_id",
            "capacity_numeric",
            "geometry",
        ]
    ].copy()

    parking_points[
        "geometry"
    ] = (
        parking_points
        .geometry
        .representative_point()
    )

    return gpd.GeoDataFrame(
        parking_points,
        geometry="geometry",
        crs=parking.crs,
    )


def calculate_local_parking_features(
    base_features: gpd.GeoDataFrame,
    parking_points: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign each parking feature to one grid cell."""

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

    assignments.sort_values(
        by=[
            "parking_id",
            "grid_id",
        ],
        inplace=True,
    )

    assignments.drop_duplicates(
        subset=["parking_id"],
        keep="first",
        inplace=True,
    )

    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "parking_count",
                "known_parking_capacity",
                "parking_capacity_record_count",
            ]
        )

    local_features = (
        assignments
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            parking_count=(
                "parking_id",
                "nunique",
            ),
            known_parking_capacity=(
                "capacity_numeric",
                "sum",
            ),
            parking_capacity_record_count=(
                "capacity_numeric",
                "count",
            ),
        )
    )

    return local_features


def calculate_parking_area_features(
    base_features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate parking polygon area inside each grid."""

    polygon_parking = parking[
        parking.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
        )
    ].copy()

    if polygon_parking.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "parking_area_m2",
            ]
        )

    parking_union_geometry = (
        polygon_parking
        .geometry
        .union_all()
    )

    parking_union = gpd.GeoDataFrame(
        {
            "parking_union": [1],
        },
        geometry=[
            parking_union_geometry
        ],
        crs=parking.crs,
    )

    grid_cells = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    intersections = gpd.overlay(
        grid_cells,
        parking_union,
        how="intersection",
        keep_geom_type=False,
        make_valid=True,
    )

    if intersections.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "parking_area_m2",
            ]
        )

    intersections[
        "parking_area_m2"
    ] = intersections.geometry.area

    intersections = intersections[
        intersections["parking_area_m2"]
        > 0
    ].copy()

    area_features = (
        intersections
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            parking_area_m2=(
                "parking_area_m2",
                "sum",
            )
        )
    )

    return area_features


def calculate_nearest_parking_features(
    base_features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate distance to the nearest parking geometry."""

    grid_centers = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_centers[
        "geometry"
    ] = grid_centers.geometry.centroid

    nearest_matches = gpd.sjoin_nearest(
        grid_centers,
        parking[
            [
                "parking_id",
                "geometry",
            ]
        ],
        how="left",
        distance_col=(
            "distance_to_nearest_parking_m"
        ),
    )

    nearest_matches.sort_values(
        by=[
            "grid_id",
            "distance_to_nearest_parking_m",
            "parking_id",
        ],
        na_position="last",
        inplace=True,
    )

    nearest_matches.drop_duplicates(
        subset=["grid_id"],
        keep="first",
        inplace=True,
    )

    return nearest_matches[
        [
            "grid_id",
            "distance_to_nearest_parking_m",
        ]
    ].copy()


def calculate_radius_count_features(
    base_features: gpd.GeoDataFrame,
    parking_points: gpd.GeoDataFrame,
    radius_meters: int,
) -> pd.DataFrame:
    """Count unique parking features around grid centers."""

    grid_buffers = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_buffers[
        "geometry"
    ] = (
        grid_buffers
        .geometry
        .centroid
        .buffer(
            radius_meters
        )
    )

    joined = gpd.sjoin(
        grid_buffers,
        parking_points[
            [
                "parking_id",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    valid_joined = joined[
        joined["parking_id"].notna()
    ].copy()

    output_column = (
        f"parking_count_within_"
        f"{radius_meters}m"
    )

    if valid_joined.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                output_column,
            ]
        )

    counts = (
        valid_joined
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            **{
                output_column: (
                    "parking_id",
                    "nunique",
                )
            }
        )
    )

    return counts


def build_grid_parking_features(
    base_features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build and merge all parking features."""

    parking_points = (
        create_parking_points(
            parking
        )
    )

    local_features = (
        calculate_local_parking_features(
            base_features,
            parking_points,
        )
    )

    area_features = (
        calculate_parking_area_features(
            base_features,
            parking,
        )
    )

    nearest_features = (
        calculate_nearest_parking_features(
            base_features,
            parking,
        )
    )

    features = base_features.merge(
        local_features,
        on="grid_id",
        how="left",
    )

    features = features.merge(
        area_features,
        on="grid_id",
        how="left",
    )

    features = features.merge(
        nearest_features,
        on="grid_id",
        how="left",
    )

    for radius_meters in (
        PARKING_COUNT_RADII_METERS
    ):
        radius_features = (
            calculate_radius_count_features(
                base_features,
                parking_points,
                radius_meters,
            )
        )

        features = features.merge(
            radius_features,
            on="grid_id",
            how="left",
        )

    zero_fill_columns = [
        "parking_count",
        "parking_area_m2",
        "known_parking_capacity",
        "parking_capacity_record_count",
        "parking_count_within_500m",
        "parking_count_within_1000m",
    ]

    features[
        zero_fill_columns
    ] = features[
        zero_fill_columns
    ].fillna(0)

    integer_columns = [
        "parking_count",
        "parking_capacity_record_count",
        "parking_count_within_500m",
        "parking_count_within_1000m",
    ]

    for column in integer_columns:
        features[column] = (
            features[column]
            .astype(int)
        )

    features[
        "known_parking_capacity"
    ] = features[
        "known_parking_capacity"
    ].round(2)

    features[
        "parking_area_m2"
    ] = features[
        "parking_area_m2"
    ].round(2)

    features[
        "parking_area_ratio"
    ] = (
        features["parking_area_m2"]
        / features["cell_area_m2"]
    ).round(6)

    features[
        "distance_to_nearest_parking_m"
    ] = features[
        "distance_to_nearest_parking_m"
    ].round(2)

    return gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=base_features.crs,
    )


def validate_features(
    features: gpd.GeoDataFrame,
    base_features: gpd.GeoDataFrame,
) -> None:
    """Validate generated parking features."""

    required_columns = {
        "grid_id",
        *PARKING_FEATURE_COLUMNS,
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required parking-feature columns "
            "are missing: "
            f"{sorted(missing_columns)}"
        )

    if features.empty:
        raise ValueError(
            "The parking-feature dataset is empty."
        )

    if len(features) != len(
        base_features
    ):
        raise ValueError(
            "Parking feature row count does not "
            "match the base grid."
        )

    if (
        features["grid_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate grid identifiers were found."
        )

    numeric_columns = list(
        PARKING_FEATURE_COLUMNS
    )

    for column in numeric_columns:
        values = (
            features[column]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Column contains non-finite "
                f"values: {column}"
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                "Column contains negative "
                f"values: {column}"
            )

    if (
        features[
            "parking_area_ratio"
        ] > 1.000001
    ).any():
        raise ValueError(
            "At least one parking-area ratio "
            "exceeds the grid-cell area."
        )

    if (
        features[
            "parking_count_within_500m"
        ]
        > features[
            "parking_count_within_1000m"
        ]
    ).any():
        raise ValueError(
            "A 500 meter parking count exceeds "
            "its 1000 meter count."
        )

    if not (
        features
        .geometry
        .is_valid
        .all()
    ):
        raise ValueError(
            "Invalid geometries were found "
            "in the feature dataset."
        )

    print(
        "Parking feature validation "
        "completed successfully."
    )


def save_feature_outputs(
    features: gpd.GeoDataFrame,
) -> None:
    """Save GIS, web and machine-learning outputs."""

    output_paths = (
        FEATURE_GPKG_OUTPUT_PATH,
        FEATURE_GEOJSON_OUTPUT_PATH,
        FEATURE_CSV_OUTPUT_PATH,
    )

    for output_path in output_paths:
        if output_path.exists():
            output_path.unlink()

    features.to_file(
        FEATURE_GPKG_OUTPUT_PATH,
        layer=FEATURE_LAYER_NAME,
        driver="GPKG",
    )

    features_wgs84 = (
        features.to_crs(
            epsg=4326
        )
    )

    features_wgs84.to_file(
        FEATURE_GEOJSON_OUTPUT_PATH,
        driver="GeoJSON",
    )

    csv_features = pd.DataFrame(
        features.drop(
            columns="geometry"
        )
    )

    csv_features.to_csv(
        FEATURE_CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "Feature GeoPackage saved: "
        f"{FEATURE_GPKG_OUTPUT_PATH}"
    )

    print(
        "Feature GeoJSON saved: "
        f"{FEATURE_GEOJSON_OUTPUT_PATH}"
    )

    print(
        "Machine-learning CSV saved: "
        f"{FEATURE_CSV_OUTPUT_PATH}"
    )


def create_preview(
    features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> None:
    """Create parking accessibility preview."""

    parking_points = (
        create_parking_points(
            parking
        )
    )

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column=(
            "parking_count_within_1000m"
        ),
        legend=True,
        linewidth=0,
        alpha=0.9,
        legend_kwds={
            "label": (
                "Parking features "
                "within 1,000 meters"
            ),
            "shrink": 0.65,
        },
    )

    parking_points.plot(
        ax=axis,
        markersize=2.5,
        alpha=0.65,
    )

    axis.set_title(
        "VoltSight - "
        "Cankaya Parking Accessibility"
    )

    axis.set_aspect("equal")
    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        PREVIEW_OUTPUT_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Parking preview saved: "
        f"{PREVIEW_OUTPUT_PATH}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> None:
    """Create Markdown parking summary."""

    cells_with_local_parking = int(
        (
            features["parking_count"]
            > 0
        ).sum()
    )

    cells_with_parking_within_500m = int(
        (
            features[
                "parking_count_within_500m"
            ]
            > 0
        ).sum()
    )

    cells_with_parking_within_1000m = int(
        (
            features[
                "parking_count_within_1000m"
            ]
            > 0
        ).sum()
    )

    polygon_count = int(
        parking.geometry.geom_type.isin(
            POLYGON_GEOMETRY_TYPES
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

    known_capacity_count = int(
        parking[
            "capacity_numeric"
        ].notna().sum()
    )

    summary = f"""# Cankaya Parking Feature Summary

## Source

- Parking data source: OpenStreetMap
- OSM tag query: `amenity=parking`
- Download method: OSMnx polygon feature query
- Download buffer: {DOWNLOAD_BUFFER_METERS:,} meters
- Successful Overpass endpoint: {ox.settings.overpass_url}
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Downloaded Parking Data

- Unique parking feature count: {len(parking):,}
- Point or multipoint count: {point_count:,}
- Polygon or multipolygon count: {polygon_count:,}
- Features with known numeric capacity: {known_capacity_count:,}
- Total known capacity: {parking["capacity_numeric"].sum():,.0f}
- Total mapped polygon area: {parking["parking_area_m2"].sum():,.2f} m2

## Grid Accessibility Results

- Grid cell count: {len(features):,}
- Cells containing a parking representative point: {cells_with_local_parking:,}
- Cells with parking within 500 meters: {cells_with_parking_within_500m:,}
- Cells with parking within 1,000 meters: {cells_with_parking_within_1000m:,}
- Mean distance to nearest parking: {features["distance_to_nearest_parking_m"].mean():,.2f} m
- Median distance to nearest parking: {features["distance_to_nearest_parking_m"].median():,.2f} m
- Maximum distance to nearest parking: {features["distance_to_nearest_parking_m"].max():,.2f} m
- Mean parking count within 500 meters: {features["parking_count_within_500m"].mean():,.2f}
- Mean parking count within 1,000 meters: {features["parking_count_within_1000m"].mean():,.2f}

## Generated Features

- `parking_count`
- `parking_area_m2`
- `parking_area_ratio`
- `distance_to_nearest_parking_m`
- `parking_count_within_500m`
- `parking_count_within_1000m`
- `known_parking_capacity`
- `parking_capacity_record_count`

## Generated Outputs

- `data/interim/cankaya_parking_features.gpkg`
- `data/processed/cankaya_grid_parking_features.gpkg`
- `data/processed/cankaya_grid_parking_features.geojson`
- `data/processed/cankaya_grid_parking_features.csv`
- `docs/cankaya_parking_features_preview.png`

## Method

Parking features were downloaded from OpenStreetMap for Cankaya and
an additional one-kilometer buffer around the district.

Each parking feature was represented by one internal point for local
cell assignment and radius-based counts. Polygon parking geometries
were unioned before grid intersection so overlapping mapped areas
would not be double-counted.

Distance to the nearest parking feature was calculated from every
grid-cell centroid in the projected meter-based coordinate system.

## Data Limitation

OpenStreetMap parking coverage and capacity attributes can be
incomplete. These variables represent mapped parking accessibility,
not a complete official parking inventory.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "Parking summary saved: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


def print_feature_statistics(
    features: gpd.GeoDataFrame,
    parking: gpd.GeoDataFrame,
) -> None:
    """Print parking statistics to the terminal."""

    print("-" * 70)

    print(
        "Prepared parking feature count: "
        f"{len(parking):,}"
    )

    print(
        "Grid cells containing parking: "
        f"{int((features['parking_count'] > 0).sum()):,}"
    )

    print(
        "Grid cells with parking within 500 m: "
        f"{int((features['parking_count_within_500m'] > 0).sum()):,}"
    )

    print(
        "Grid cells with parking within 1,000 m: "
        f"{int((features['parking_count_within_1000m'] > 0).sum()):,}"
    )

    print(
        "Median distance to nearest parking: "
        f"{features['distance_to_nearest_parking_m'].median():,.2f} m"
    )

    print(
        "Maximum distance to nearest parking: "
        f"{features['distance_to_nearest_parking_m'].max():,.2f} m"
    )

    print(
        "Mean parking count within 1,000 m: "
        f"{features['parking_count_within_1000m'].mean():,.2f}"
    )


def main() -> None:
    """Run the Cankaya parking feature pipeline."""

    print("=" * 70)

    print(
        "VoltSight - "
        "Cankaya Parking Feature Pipeline"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_files()
    configure_osmnx()

    boundary, base_features = (
        load_study_data()
    )

    download_polygon = (
        create_download_polygon(
            boundary
        )
    )

    raw_parking = (
        download_parking_features(
            download_polygon
        )
    )

    parking = prepare_parking_features(
        raw_parking,
        base_features.crs,
    )

    save_parking_features(
        parking
    )

    features = (
        build_grid_parking_features(
            base_features,
            parking,
        )
    )

    validate_features(
        features,
        base_features,
    )

    save_feature_outputs(
        features
    )

    create_preview(
        features,
        parking,
    )

    create_summary(
        features,
        parking,
    )

    print_feature_statistics(
        features,
        parking,
    )

    print("=" * 70)

    print(
        "Parking feature pipeline "
        "completed successfully."
    )

    print(
        f"Feature row count: "
        f"{len(features):,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()