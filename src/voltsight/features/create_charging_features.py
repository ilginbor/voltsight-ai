from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
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
    / "cankaya_grid_parking_features.gpkg"
)

CHARGING_STATIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations.gpkg"
)


MERGED_CHARGING_STATIONS_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.gpkg"
)

CHARGING_SOURCE_MERGE_SCRIPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "voltsight"
    / "data"
    / "merge_charging_station_sources.py"
)

FEATURE_GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.gpkg"
)

FEATURE_GEOJSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.geojson"
)

FEATURE_CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.csv"
)

PREVIEW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_features_preview.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_features_summary.md"
)

CACHE_DIRECTORY = PROJECT_ROOT / "cache"

BASE_FEATURE_LAYER_NAME = "grid_parking_features"
CHARGING_STATION_LAYER_NAME = "charging_stations"
MERGED_CHARGING_STATION_LAYER_NAME = "charging_stations_merged"
FEATURE_LAYER_NAME = "grid_charging_features"

DOWNLOAD_BUFFER_METERS = 2_500
CHARGING_COUNT_RADII_METERS = (1_000, 2_000)

OVERPASS_REQUEST_TIMEOUT_SECONDS = 600
MAX_QUERY_AREA_SIZE_M2 = 100_000_000

OVERPASS_ENDPOINTS = (
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass-api.de/api",
)

CHARGING_TAGS = {
    "amenity": "charging_station",
}

SUPPORTED_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
}

OPTIONAL_OSM_COLUMNS = (
    "name",
    "operator",
    "brand",
    "network",
    "access",
    "fee",
    "capacity",
    "opening_hours",
    "authentication:membership_card",
    "authentication:app",
    "payment:credit_cards",
    "payment:debit_cards",
    "motorcar",
    "bicycle",
)

AC_SOCKET_KEYS = {
    "socket:type1",
    "socket:type2",
    "socket:type3a",
    "socket:type3c",
    "socket:schuko",
    "socket:cee_blue",
    "socket:cee_red_16a",
    "socket:cee_red_32a",
    "socket:cee_red_63a",
    "socket:tesla_destination",
}

DC_SOCKET_KEYS = {
    "socket:type1_combo",
    "socket:type2_combo",
    "socket:chademo",
    "socket:nacs",
    "socket:tesla_supercharger",
    "socket:gb_t_dc",
}

CHARGING_FEATURE_COLUMNS = (
    "charging_station_count",
    "has_existing_charging_station",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "known_charging_capacity",
    "charging_capacity_record_count",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
)


def create_output_directories() -> None:
    """Create directories required by the charging pipeline."""

    directories = (
        CHARGING_STATIONS_OUTPUT_PATH.parent,
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
    """Ensure that the boundary and parking-feature files exist."""

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
        "Run create_study_grid.py, create_road_features.py and "
        "create_parking_features.py first.\n"
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
    """Load the district boundary and accumulated grid features."""

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
            "The grid parking-feature dataset is empty."
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
            "The base feature dataset is missing required columns: "
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
                "Cankaya charging-station download area"
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


def download_charging_stations(
    download_polygon: Any,
) -> gpd.GeoDataFrame:
    """Download charging stations using endpoint fallbacks."""

    endpoint_errors: list[str] = []

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint

        print("-" * 70)

        print(
            f"Trying Overpass endpoint: "
            f"{endpoint}"
        )

        print(
            "Downloading EV charging stations "
            "from OpenStreetMap..."
        )

        try:
            charging_stations = (
                ox.features.features_from_polygon(
                    download_polygon,
                    tags=CHARGING_TAGS,
                )
            )

            if charging_stations.empty:
                raise RuntimeError(
                    "The query returned "
                    "no charging stations."
                )

            print(
                f"Successful endpoint: "
                f"{endpoint}"
            )

            print(
                "Downloaded OSM charging-station records: "
                f"{len(charging_stations):,}"
            )

            return charging_stations

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
        "Charging stations could not be downloaded "
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


def parse_non_negative_number(
    value: Any,
) -> float:
    """Parse a simple non-negative numeric OSM tag."""

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


def socket_tag_is_available(
    value: Any,
) -> bool:
    """Return whether an OSM socket tag describes availability."""

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    text_value = (
        str(value)
        .strip()
        .lower()
    )

    if text_value in {
        "",
        "no",
        "0",
        "false",
        "none",
    }:
        return False

    if text_value in {
        "yes",
        "true",
    }:
        return True

    for part in (
        text_value
        .replace(",", ".")
        .split(";")
    ):
        try:
            if float(
                part.strip()
            ) > 0:
                return True
        except ValueError:
            continue

    return False


def parse_known_socket_count(
    value: Any,
) -> float:
    """Parse numeric socket counts without treating yes as a count."""

    if value is None:
        return 0.0

    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    text_value = (
        str(value)
        .strip()
        .replace(",", ".")
    )

    if not text_value:
        return 0.0

    numeric_values: list[float] = []

    for part in text_value.split(";"):
        try:
            numeric_value = float(
                part.strip()
            )
        except ValueError:
            continue

        if (
            np.isfinite(numeric_value)
            and numeric_value > 0
        ):
            numeric_values.append(
                numeric_value
            )

    if not numeric_values:
        return 0.0

    return float(
        sum(numeric_values)
    )


def prepare_charging_stations(
    raw_charging_stations: gpd.GeoDataFrame,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Clean, normalize and project OSM charging stations."""

    if raw_charging_stations.crs is None:
        raise ValueError(
            "Downloaded charging stations have no CRS."
        )

    charging_stations = (
        raw_charging_stations
        .reset_index()
        .copy()
    )

    element_column = find_column(
        charging_stations.columns,
        (
            "element",
            "element_type",
            "type",
        ),
        "OSM element type",
    )

    osm_id_column = find_column(
        charging_stations.columns,
        (
            "id",
            "osmid",
            "osm_id",
        ),
        "OSM identifier",
    )

    charging_stations.rename(
        columns={
            element_column: (
                "osm_element_type"
            ),
            osm_id_column: "osm_id",
        },
        inplace=True,
    )

    charging_stations = (
        charging_stations[
            charging_stations
            .geometry
            .notna()
        ]
        .copy()
    )

    charging_stations = (
        charging_stations[
            ~charging_stations
            .geometry
            .is_empty
        ]
        .copy()
    )

    charging_stations = (
        charging_stations[
            charging_stations
            .geometry
            .geom_type
            .isin(
                SUPPORTED_GEOMETRY_TYPES
            )
        ]
        .copy()
    )

    if charging_stations.empty:
        raise ValueError(
            "No supported charging-station geometries "
            "remained after cleaning."
        )

    charging_stations[
        "osm_element_type"
    ] = charging_stations[
        "osm_element_type"
    ].astype(str)

    charging_stations[
        "osm_id"
    ] = charging_stations[
        "osm_id"
    ].astype(str)

    charging_stations[
        "station_id"
    ] = (
        charging_stations[
            "osm_element_type"
        ]
        + "_"
        + charging_stations[
            "osm_id"
        ]
    )

    charging_stations.drop_duplicates(
        subset=["station_id"],
        keep="first",
        inplace=True,
    )

    for column in OPTIONAL_OSM_COLUMNS:
        if column not in charging_stations.columns:
            charging_stations[column] = ""

        charging_stations[column] = (
            charging_stations[
                column
            ].apply(
                stringify_tag_value
            )
        )

    charging_stations[
        "capacity_numeric"
    ] = charging_stations[
        "capacity"
    ].apply(
        parse_non_negative_number
    )

    socket_columns = sorted(
        column
        for column in charging_stations.columns
        if str(column).startswith("socket:")
    )

    connector_type_values: list[str] = []
    mapped_socket_type_counts: list[int] = []
    known_socket_counts: list[float] = []
    has_ac_values: list[bool] = []
    has_dc_values: list[bool] = []

    for _, row in charging_stations.iterrows():
        available_socket_keys = [
            column
            for column in socket_columns
            if socket_tag_is_available(
                row[column]
            )
        ]

        connector_type_values.append(
            ";".join(
                column.removeprefix(
                    "socket:"
                )
                for column in (
                    available_socket_keys
                )
            )
        )

        mapped_socket_type_counts.append(
            len(
                available_socket_keys
            )
        )

        known_socket_counts.append(
            sum(
                parse_known_socket_count(
                    row[column]
                )
                for column in (
                    available_socket_keys
                )
            )
        )

        has_ac_values.append(
            any(
                column in AC_SOCKET_KEYS
                for column in (
                    available_socket_keys
                )
            )
        )

        has_dc_values.append(
            any(
                column in DC_SOCKET_KEYS
                for column in (
                    available_socket_keys
                )
            )
        )

    charging_stations[
        "connector_types"
    ] = connector_type_values

    charging_stations[
        "mapped_socket_type_count"
    ] = mapped_socket_type_counts

    charging_stations[
        "known_socket_count"
    ] = (
        np.array(
            known_socket_counts,
            dtype=float,
        )
        .round(2)
    )

    charging_stations[
        "has_ac_connector"
    ] = has_ac_values

    charging_stations[
        "has_dc_connector"
    ] = has_dc_values

    charging_stations = (
        charging_stations.to_crs(
            target_crs
        )
    )

    charging_stations[
        "geometry_type"
    ] = (
        charging_stations
        .geometry
        .geom_type
    )

    selected_columns = [
        "station_id",
        "osm_element_type",
        "osm_id",
        "name",
        "operator",
        "brand",
        "network",
        "access",
        "fee",
        "capacity",
        "capacity_numeric",
        "opening_hours",
        "authentication:membership_card",
        "authentication:app",
        "payment:credit_cards",
        "payment:debit_cards",
        "motorcar",
        "bicycle",
        "connector_types",
        "mapped_socket_type_count",
        "known_socket_count",
        "has_ac_connector",
        "has_dc_connector",
        "geometry_type",
        "geometry",
    ]

    charging_stations = (
        charging_stations[
            selected_columns
        ]
        .copy()
    )

    charging_stations.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        "Prepared unique charging stations: "
        f"{len(charging_stations):,}"
    )

    print(
        "Stations with known capacity: "
        f"{int(charging_stations['capacity_numeric'].notna().sum()):,}"
    )

    print(
        "Stations with mapped AC connector: "
        f"{int(charging_stations['has_ac_connector'].sum()):,}"
    )

    print(
        "Stations with mapped DC connector: "
        f"{int(charging_stations['has_dc_connector'].sum()):,}"
    )

    return gpd.GeoDataFrame(
        charging_stations,
        geometry="geometry",
        crs=target_crs,
    )


def save_charging_stations(
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Save the cleaned charging-station dataset."""

    if CHARGING_STATIONS_OUTPUT_PATH.exists():
        CHARGING_STATIONS_OUTPUT_PATH.unlink()

    charging_stations.to_file(
        CHARGING_STATIONS_OUTPUT_PATH,
        layer=CHARGING_STATION_LAYER_NAME,
        driver="GPKG",
    )

    print(
        "Charging stations saved: "
        f"{CHARGING_STATIONS_OUTPUT_PATH}"
    )


def refresh_merged_charging_stations() -> None:
    """Rebuild the merged OSM and verified EPDK station inventory."""

    if not CHARGING_SOURCE_MERGE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "Charging-station source merge script was not found:\n"
            f"{CHARGING_SOURCE_MERGE_SCRIPT_PATH}"
        )

    print("-" * 70)

    print(
        "Refreshing merged OSM and EPDK "
        "charging-station inventory..."
    )

    subprocess.run(
        [
            sys.executable,
            str(
                CHARGING_SOURCE_MERGE_SCRIPT_PATH
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not MERGED_CHARGING_STATIONS_INPUT_PATH.exists():
        raise FileNotFoundError(
            "Merged charging-station output was not created:\n"
            f"{MERGED_CHARGING_STATIONS_INPUT_PATH}"
        )


def load_feature_charging_stations(
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Load the merged station inventory used for feature creation."""

    required_columns = {
        "station_id",
        "capacity_numeric",
        "has_ac_connector",
        "has_dc_connector",
        "data_source",
        "source_osm",
        "source_epdk",
        "geometry",
    }

    charging_stations = gpd.read_file(
        MERGED_CHARGING_STATIONS_INPUT_PATH,
        layer=MERGED_CHARGING_STATION_LAYER_NAME,
    )

    if charging_stations.empty:
        raise ValueError(
            "The merged charging-station dataset is empty."
        )

    if charging_stations.crs is None:
        raise ValueError(
            "The merged charging-station dataset has no CRS."
        )

    missing_columns = (
        required_columns
        - set(charging_stations.columns)
    )

    if missing_columns:
        raise ValueError(
            "The merged charging-station dataset is missing "
            f"columns: {sorted(missing_columns)}"
        )

    charging_stations[
        "station_id"
    ] = (
        charging_stations["station_id"]
        .astype(str)
        .str.strip()
    )

    if (
        charging_stations["station_id"]
        .eq("")
        .any()
    ):
        raise ValueError(
            "The merged dataset contains an empty station ID."
        )

    if (
        charging_stations["station_id"]
        .duplicated()
        .any()
    ):
        duplicate_ids = (
            charging_stations.loc[
                charging_stations[
                    "station_id"
                ].duplicated(
                    keep=False
                ),
                "station_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate merged station IDs were found: "
            f"{duplicate_ids}"
        )

    if (
        charging_stations.geometry
        .isna()
        .any()
    ):
        raise ValueError(
            "The merged dataset contains missing geometries."
        )

    charging_stations = (
        charging_stations
        .to_crs(target_crs)
        .copy()
    )

    charging_stations[
        "capacity_numeric"
    ] = pd.to_numeric(
        charging_stations[
            "capacity_numeric"
        ],
        errors="coerce",
    )

    charging_stations.loc[
        charging_stations[
            "capacity_numeric"
        ].lt(0),
        "capacity_numeric",
    ] = np.nan

    flag_columns = (
        "has_ac_connector",
        "has_dc_connector",
        "source_osm",
        "source_epdk",
    )

    for column in flag_columns:
        charging_stations[column] = (
            pd.to_numeric(
                charging_stations[column],
                errors="coerce",
            )
            .fillna(0)
        )

    connector_flag_columns = (
        "has_ac_connector",
        "has_dc_connector",
    )

    for column in connector_flag_columns:
        charging_stations[column] = (
            charging_stations[column]
            .gt(0)
        )

    source_flag_columns = (
        "source_osm",
        "source_epdk",
    )

    for column in source_flag_columns:
        charging_stations[column] = (
            charging_stations[column]
            .gt(0)
            .astype(int)
        )

    print(
        "Loaded merged charging-station count: "
        f"{len(charging_stations):,}"
    )

    source_counts = (
        charging_stations[
            "data_source"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    print(
        "Merged charging-station source counts:"
    )

    for source, count in source_counts.items():
        print(
            f"  {source}: {int(count):,}"
        )

    return charging_stations


def create_station_points(
    charging_stations: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create one representative point per charging station."""

    station_points = charging_stations[
        [
            "station_id",
            "capacity_numeric",
            "has_ac_connector",
            "has_dc_connector",
            "geometry",
        ]
    ].copy()

    station_points[
        "geometry"
    ] = (
        station_points
        .geometry
        .representative_point()
    )

    return gpd.GeoDataFrame(
        station_points,
        geometry="geometry",
        crs=charging_stations.crs,
    )


def calculate_local_station_features(
    base_features: gpd.GeoDataFrame,
    station_points: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign every charging station to one grid cell."""

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

    assignments.sort_values(
        by=[
            "station_id",
            "grid_id",
        ],
        inplace=True,
    )

    assignments.drop_duplicates(
        subset=["station_id"],
        keep="first",
        inplace=True,
    )

    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "charging_station_count",
                "known_charging_capacity",
                "charging_capacity_record_count",
            ]
        )

    return (
        assignments
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            charging_station_count=(
                "station_id",
                "nunique",
            ),
            known_charging_capacity=(
                "capacity_numeric",
                "sum",
            ),
            charging_capacity_record_count=(
                "capacity_numeric",
                "count",
            ),
        )
    )


def calculate_nearest_station_features(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate distance to the nearest charging station."""

    grid_centers = base_features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_centers[
        "geometry"
    ] = (
        grid_centers
        .geometry
        .centroid
    )

    nearest_matches = (
        gpd.sjoin_nearest(
            grid_centers,
            charging_stations[
                [
                    "station_id",
                    "geometry",
                ]
            ],
            how="left",
            distance_col=(
                "distance_to_nearest_charging_station_m"
            ),
        )
    )

    nearest_matches.sort_values(
        by=[
            "grid_id",
            "distance_to_nearest_charging_station_m",
            "station_id",
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
            "distance_to_nearest_charging_station_m",
        ]
    ].copy()


def calculate_radius_count_features(
    base_features: gpd.GeoDataFrame,
    station_points: gpd.GeoDataFrame,
    radius_meters: int,
    output_column: str,
) -> pd.DataFrame:
    """Count unique charging stations around grid centers."""

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
        station_points[
            [
                "station_id",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    valid_joined = (
        joined[
            joined[
                "station_id"
            ].notna()
        ]
        .copy()
    )

    if valid_joined.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                output_column,
            ]
        )

    return (
        valid_joined
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            **{
                output_column: (
                    "station_id",
                    "nunique",
                )
            }
        )
    )


def build_grid_charging_features(
    base_features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build and merge all charging-station features."""

    station_points = (
        create_station_points(
            charging_stations
        )
    )

    local_features = (
        calculate_local_station_features(
            base_features,
            station_points,
        )
    )

    nearest_features = (
        calculate_nearest_station_features(
            base_features,
            charging_stations,
        )
    )

    features = base_features.merge(
        local_features,
        on="grid_id",
        how="left",
    )

    features = features.merge(
        nearest_features,
        on="grid_id",
        how="left",
    )

    for radius_meters in (
        CHARGING_COUNT_RADII_METERS
    ):
        output_column = (
            f"charging_station_count_within_"
            f"{radius_meters}m"
        )

        radius_features = (
            calculate_radius_count_features(
                base_features,
                station_points,
                radius_meters,
                output_column,
            )
        )

        features = features.merge(
            radius_features,
            on="grid_id",
            how="left",
        )

    ac_station_points = (
        station_points[
            station_points[
                "has_ac_connector"
            ]
        ]
        .copy()
    )

    dc_station_points = (
        station_points[
            station_points[
                "has_dc_connector"
            ]
        ]
        .copy()
    )

    ac_features = (
        calculate_radius_count_features(
            base_features,
            ac_station_points,
            1_000,
            "ac_station_count_within_1000m",
        )
    )

    dc_features = (
        calculate_radius_count_features(
            base_features,
            dc_station_points,
            1_000,
            "dc_station_count_within_1000m",
        )
    )

    features = features.merge(
        ac_features,
        on="grid_id",
        how="left",
    )

    features = features.merge(
        dc_features,
        on="grid_id",
        how="left",
    )

    zero_fill_columns = [
        "charging_station_count",
        "known_charging_capacity",
        "charging_capacity_record_count",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    ]

    features[
        zero_fill_columns
    ] = features[
        zero_fill_columns
    ].fillna(0)

    integer_columns = [
        "charging_station_count",
        "charging_capacity_record_count",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    ]

    for column in integer_columns:
        features[column] = (
            features[column]
            .astype(int)
        )

    features[
        "has_existing_charging_station"
    ] = (
        features[
            "charging_station_count"
        ]
        > 0
    ).astype(int)

    features[
        "known_charging_capacity"
    ] = features[
        "known_charging_capacity"
    ].round(2)

    features[
        "distance_to_nearest_charging_station_m"
    ] = features[
        "distance_to_nearest_charging_station_m"
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
    """Validate generated charging-station features."""

    required_columns = {
        "grid_id",
        *CHARGING_FEATURE_COLUMNS,
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required charging-feature columns "
            "are missing: "
            f"{sorted(missing_columns)}"
        )

    if features.empty:
        raise ValueError(
            "The charging-feature dataset is empty."
        )

    if len(features) != len(
        base_features
    ):
        raise ValueError(
            "Charging feature row count does not "
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
        CHARGING_FEATURE_COLUMNS
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
                "Column contains non-finite values: "
                f"{column}"
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                "Column contains negative values: "
                f"{column}"
            )

    if (
        features[
            "charging_station_count_within_1000m"
        ]
        > features[
            "charging_station_count_within_2000m"
        ]
    ).any():
        raise ValueError(
            "A 1,000 meter charging-station count "
            "exceeds its 2,000 meter count."
        )

    expected_target = (
        features[
            "charging_station_count"
        ]
        > 0
    ).astype(int)

    if not expected_target.equals(
        features[
            "has_existing_charging_station"
        ].astype(int)
    ):
        raise ValueError(
            "The existing-station target does not "
            "match the local station count."
        )

    if not (
        features
        .geometry
        .is_valid
        .all()
    ):
        raise ValueError(
            "Invalid geometries were found in "
            "the charging-feature dataset."
        )

    print(
        "Charging feature validation "
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
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Create charging-station accessibility preview."""

    station_points = (
        create_station_points(
            charging_stations
        )
    )

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column=(
            "distance_to_nearest_charging_station_m"
        ),
        legend=True,
        linewidth=0,
        alpha=0.9,
        legend_kwds={
            "label": (
                "Distance to nearest mapped "
                "charging station (m)"
            ),
            "shrink": 0.65,
        },
    )

    station_points.plot(
        ax=axis,
        markersize=8,
        alpha=0.8,
    )

    axis.set_title(
        "VoltSight - "
        "Cankaya EV Charging Accessibility"
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
        "Charging preview saved: "
        f"{PREVIEW_OUTPUT_PATH}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Create Markdown charging-station summary."""

    cells_with_station = int(
        features[
            "has_existing_charging_station"
        ].sum()
    )

    cells_with_station_within_1000m = int(
        (
            features[
                "charging_station_count_within_1000m"
            ]
            > 0
        ).sum()
    )

    cells_with_station_within_2000m = int(
        (
            features[
                "charging_station_count_within_2000m"
            ]
            > 0
        ).sum()
    )

    known_capacity_count = int(
        charging_stations[
            "capacity_numeric"
        ].notna().sum()
    )

    ac_station_count = int(
        charging_stations[
            "has_ac_connector"
        ].sum()
    )

    dc_station_count = int(
        charging_stations[
            "has_dc_connector"
        ].sum()
    )

    summary = f"""# Cankaya Charging Station Feature Summary

## Source

- Charging-station data source: OpenStreetMap
- OSM tag query: `amenity=charging_station`
- Download method: OSMnx polygon feature query
- Download buffer: {DOWNLOAD_BUFFER_METERS:,} meters
- Successful Overpass endpoint: {ox.settings.overpass_url}
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Downloaded Charging Stations

- Unique mapped charging stations: {len(charging_stations):,}
- Stations with known numeric capacity: {known_capacity_count:,}
- Total known capacity: {charging_stations["capacity_numeric"].sum():,.0f}
- Stations with a mapped AC connector: {ac_station_count:,}
- Stations with a mapped DC connector: {dc_station_count:,}

## Grid Results

- Grid cell count: {len(features):,}
- Cells containing a mapped charging station: {cells_with_station:,}
- Cells with a mapped station within 1,000 meters: {cells_with_station_within_1000m:,}
- Cells with a mapped station within 2,000 meters: {cells_with_station_within_2000m:,}
- Mean distance to nearest mapped station: {features["distance_to_nearest_charging_station_m"].mean():,.2f} m
- Median distance to nearest mapped station: {features["distance_to_nearest_charging_station_m"].median():,.2f} m
- Maximum distance to nearest mapped station: {features["distance_to_nearest_charging_station_m"].max():,.2f} m

## Generated Columns

- `charging_station_count`
- `has_existing_charging_station`
- `distance_to_nearest_charging_station_m`
- `charging_station_count_within_1000m`
- `charging_station_count_within_2000m`
- `known_charging_capacity`
- `charging_capacity_record_count`
- `ac_station_count_within_1000m`
- `dc_station_count_within_1000m`

## Generated Outputs

- `data/interim/cankaya_charging_stations.gpkg`
- `data/processed/cankaya_grid_charging_features.gpkg`
- `data/processed/cankaya_grid_charging_features.geojson`
- `data/processed/cankaya_grid_charging_features.csv`
- `docs/cankaya_charging_features_preview.png`

## Method

Charging stations were downloaded from OpenStreetMap for Cankaya and
an additional 2.5-kilometer buffer around the district.

Each mapped station was represented by one internal point for local
cell assignment and radius-based accessibility counts. Distances were
calculated from grid-cell centroids in the projected meter-based
coordinate system.

## Scientific Use Warning

`has_existing_charging_station` and `charging_station_count` are target
or descriptive columns. They must not be included as predictor inputs
when training a model to reproduce the current station distribution.

Distance and neighborhood-count columns also require leakage-aware
feature design before model training.

## Data Limitation

OpenStreetMap coverage, station capacity and connector tags may be
incomplete. These values describe mapped infrastructure rather than a
complete official charging-station inventory.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "Charging summary saved: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


def print_feature_statistics(
    features: gpd.GeoDataFrame,
    charging_stations: gpd.GeoDataFrame,
) -> None:
    """Print charging-station statistics to the terminal."""

    print("-" * 70)

    print(
        "Prepared charging-station count: "
        f"{len(charging_stations):,}"
    )

    print(
        "Grid cells containing a charging station: "
        f"{int(features['has_existing_charging_station'].sum()):,}"
    )

    print(
        "Grid cells with a station within 1,000 m: "
        f"{int((features['charging_station_count_within_1000m'] > 0).sum()):,}"
    )

    print(
        "Grid cells with a station within 2,000 m: "
        f"{int((features['charging_station_count_within_2000m'] > 0).sum()):,}"
    )

    print(
        "Median distance to nearest charging station: "
        f"{features['distance_to_nearest_charging_station_m'].median():,.2f} m"
    )

    print(
        "Maximum distance to nearest charging station: "
        f"{features['distance_to_nearest_charging_station_m'].max():,.2f} m"
    )

    print(
        "Stations with mapped AC connector: "
        f"{int(charging_stations['has_ac_connector'].sum()):,}"
    )

    print(
        "Stations with mapped DC connector: "
        f"{int(charging_stations['has_dc_connector'].sum()):,}"
    )


def main() -> None:
    """Run the Cankaya charging-station feature pipeline."""

    print("=" * 70)

    print(
        "VoltSight - "
        "Cankaya Charging Station Feature Pipeline"
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

    raw_charging_stations = (
        download_charging_stations(
            download_polygon
        )
    )

    charging_stations = (
        prepare_charging_stations(
            raw_charging_stations,
            base_features.crs,
        )
    )

    save_charging_stations(
        charging_stations
    )

    refresh_merged_charging_stations()

    feature_charging_stations = (
        load_feature_charging_stations(
            base_features.crs
        )
    )

    features = (
        build_grid_charging_features(
            base_features,
            feature_charging_stations,
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
        feature_charging_stations,
    )

    create_summary(
        features,
        feature_charging_stations,
    )

    print_feature_statistics(
        features,
        feature_charging_stations,
    )

    print("=" * 70)

    print(
        "Charging feature pipeline "
        "completed successfully."
    )

    print(
        f"Feature row count: "
        f"{len(features):,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()