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

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_250m.gpkg"
)

ROADS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_drive_roads.gpkg"
)

FEATURE_GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.gpkg"
)

FEATURE_GEOJSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.geojson"
)

FEATURE_CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_road_features.csv"
)

PREVIEW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_road_features_preview.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_road_features_summary.md"
)

CACHE_DIRECTORY = PROJECT_ROOT / "cache"

GRID_LAYER_NAME = "cankaya_grid_250m"
ROADS_LAYER_NAME = "drive_roads"
FEATURE_LAYER_NAME = "grid_road_features"

NETWORK_TYPE = "drive"
DOWNLOAD_BUFFER_METERS = 1_000
OVERPASS_REQUEST_TIMEOUT_SECONDS = 600

# 100 square kilometers.
# OSMnx divides larger polygons into multiple Overpass requests.
MAX_QUERY_AREA_SIZE_M2 = 100_000_000

# OSMnx automatically adds /interpreter and /status to these base URLs.
# If one server cannot be reached, the next server will be attempted.
OVERPASS_ENDPOINTS = (
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass-api.de/api",
)

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


def create_output_directories() -> None:
    """Create directories required by the feature pipeline."""

    directories = (
        ROADS_OUTPUT_PATH.parent,
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
    """Ensure that the boundary and study-grid files exist."""

    required_paths = (
        BOUNDARY_PATH,
        GRID_PATH,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if not missing_paths:
        return

    missing_path_text = "\n".join(
        f"- {path}"
        for path in missing_paths
    )

    raise FileNotFoundError(
        "Required input files are missing.\n"
        "Run create_study_grid.py before this pipeline.\n"
        f"{missing_path_text}"
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

    # Public alternative endpoints do not always support
    # the same slot-management response as overpass-api.de.
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
        f"{ox.settings.max_query_area_size / 1_000_000:,.0f} km²"
    )


def load_study_data() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load and validate the Çankaya boundary and grid."""

    boundary = gpd.read_file(
        BOUNDARY_PATH
    )

    grid = gpd.read_file(
        GRID_PATH,
        layer=GRID_LAYER_NAME,
    )

    if boundary.empty:
        raise ValueError(
            "The Çankaya boundary dataset is empty."
        )

    if grid.empty:
        raise ValueError(
            "The Çankaya study grid is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "The boundary coordinate system is missing."
        )

    if grid.crs is None:
        raise ValueError(
            "The grid coordinate system is missing."
        )

    if not grid.crs.is_projected:
        raise ValueError(
            "The grid must use a projected "
            "meter-based coordinate system."
        )

    required_grid_columns = {
        "grid_id",
        "cell_area_m2",
        "geometry",
    }

    missing_columns = (
        required_grid_columns
        - set(grid.columns)
    )

    if missing_columns:
        raise ValueError(
            "The study grid is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    boundary = boundary.to_crs(
        grid.crs
    )

    print(
        f"Loaded grid cell count: {len(grid):,}"
    )

    print(
        f"Grid coordinate system: {grid.crs}"
    )

    return boundary, grid


def create_download_polygon(
    boundary_projected: gpd.GeoDataFrame,
) -> Any:
    """
    Create a buffered WGS84 polygon for the road download.

    The buffer allows border cells to find nearby roads that are
    located immediately outside the administrative boundary.
    """

    district_geometry = (
        boundary_projected.geometry.union_all()
    )

    buffered_geometry = (
        district_geometry.buffer(
            DOWNLOAD_BUFFER_METERS
        )
    )

    buffered_boundary = gpd.GeoDataFrame(
        {
            "name": [
                "Çankaya road download area"
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


def download_road_network(
    download_polygon: Any,
    target_crs: Any,
):
    """
    Download the road graph using multiple Overpass endpoints.

    If a server times out or cannot be reached, the next endpoint
    is attempted automatically.
    """

    endpoint_errors: list[str] = []

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint

        print("-" * 70)

        print(
            f"Trying Overpass endpoint: {endpoint}"
        )

        print(
            "Downloading the drivable road network "
            "from OpenStreetMap..."
        )

        try:
            directed_graph = (
                ox.graph.graph_from_polygon(
                    download_polygon,
                    network_type=NETWORK_TYPE,
                    simplify=True,
                    retain_all=True,
                    truncate_by_edge=True,
                )
            )

            if (
                directed_graph.number_of_edges()
                == 0
            ):
                raise RuntimeError(
                    "The downloaded graph "
                    "contains no road edges."
                )

            projected_graph = (
                ox.projection.project_graph(
                    directed_graph,
                    to_crs=target_crs,
                )
            )

            undirected_graph = (
                ox.convert.to_undirected(
                    projected_graph
                )
            )

            print(
                f"Successful endpoint: {endpoint}"
            )

            print(
                "Downloaded graph nodes: "
                f"{undirected_graph.number_of_nodes():,}"
            )

            print(
                "Downloaded physical road edges: "
                f"{undirected_graph.number_of_edges():,}"
            )

            return undirected_graph

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
                f"Endpoint failed: {error_message}"
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
        "The OpenStreetMap road network could not "
        "be downloaded from any configured "
        "Overpass endpoint.\n"
        "Check the internet connection and try "
        "again later.\n"
        f"Attempt details:\n{error_details}"
    )


def normalize_tag_values(
    value: Any,
) -> list[str]:
    """
    Convert scalar or list-based OpenStreetMap tags
    into a clean list of strings.
    """

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

    normalized_values = (
        normalize_tag_values(value)
    )

    return ";".join(
        normalized_values
    )


def prepare_road_edges(
    road_graph,
) -> gpd.GeoDataFrame:
    """
    Convert the downloaded graph into clean physical road edges.

    The graph is already undirected at this stage, so reciprocal
    driving directions are not counted as separate physical roads.
    """

    roads = ox.convert.graph_to_gdfs(
        road_graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )

    roads = roads.reset_index()

    roads = roads[
        roads.geometry.notna()
    ].copy()

    roads = roads[
        ~roads.geometry.is_empty
    ].copy()

    roads = roads[
        roads.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()

    roads.reset_index(
        drop=True,
        inplace=True,
    )

    roads.insert(
        0,
        "road_id",
        [
            f"ROAD_{index:06d}"
            for index in range(
                1,
                len(roads) + 1,
            )
        ],
    )

    optional_columns = (
        "highway",
        "name",
        "osmid",
        "oneway",
        "lanes",
        "maxspeed",
    )

    for column in optional_columns:
        if column not in roads.columns:
            roads[column] = ""

    highway_value_lists = (
        roads["highway"].apply(
            normalize_tag_values
        )
    )

    roads["highway"] = (
        highway_value_lists.apply(
            lambda values: ";".join(
                values
            )
        )
    )

    roads["is_main_road"] = (
        highway_value_lists.apply(
            lambda values: any(
                value in MAIN_ROAD_TYPES
                for value in values
            )
        )
    )

    roads["osm_id"] = (
        roads["osmid"].apply(
            stringify_tag_value
        )
    )

    roads["name"] = (
        roads["name"].apply(
            stringify_tag_value
        )
    )

    roads["oneway"] = (
        roads["oneway"].apply(
            stringify_tag_value
        )
    )

    roads["lanes"] = (
        roads["lanes"].apply(
            stringify_tag_value
        )
    )

    roads["maxspeed"] = (
        roads["maxspeed"].apply(
            stringify_tag_value
        )
    )

    roads["edge_length_m"] = (
        roads.geometry.length.round(2)
    )

    selected_columns = [
        "road_id",
        "u",
        "v",
        "key",
        "osm_id",
        "name",
        "highway",
        "is_main_road",
        "oneway",
        "lanes",
        "maxspeed",
        "edge_length_m",
        "geometry",
    ]

    roads = roads[
        selected_columns
    ].copy()

    print(
        "Prepared road edge count: "
        f"{len(roads):,}"
    )

    print(
        "Prepared main-road edge count: "
        f"{int(roads['is_main_road'].sum()):,}"
    )

    return roads


def save_road_edges(
    roads: gpd.GeoDataFrame,
) -> None:
    """Save the projected road network as GeoPackage."""

    if ROADS_OUTPUT_PATH.exists():
        ROADS_OUTPUT_PATH.unlink()

    roads.to_file(
        ROADS_OUTPUT_PATH,
        layer=ROADS_LAYER_NAME,
        driver="GPKG",
    )

    print(
        "Road network saved: "
        f"{ROADS_OUTPUT_PATH}"
    )


def intersect_roads_with_grid(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Intersect road geometries with grid cells.

    Only the road portion located inside a cell contributes
    to that cell's road-length feature.
    """

    grid_for_overlay = grid[
        [
            "grid_id",
            "cell_area_m2",
            "geometry",
        ]
    ].copy()

    roads_for_overlay = roads[
        [
            "road_id",
            "highway",
            "is_main_road",
            "geometry",
        ]
    ].copy()

    print(
        "Intersecting road edges "
        "with study-grid cells..."
    )

    intersections = gpd.overlay(
        grid_for_overlay,
        roads_for_overlay,
        how="intersection",
        keep_geom_type=False,
        make_valid=True,
    )

    intersections = intersections[
        intersections.geometry.notna()
    ].copy()

    intersections = intersections[
        ~intersections.geometry.is_empty
    ].copy()

    intersections = intersections[
        intersections.geometry.geom_type.isin(
            [
                "LineString",
                "MultiLineString",
            ]
        )
    ].copy()

    intersections[
        "clipped_length_m"
    ] = intersections.geometry.length

    intersections = intersections[
        intersections["clipped_length_m"]
        > 0
    ].copy()

    print(
        "Generated grid-road intersections: "
        f"{len(intersections):,}"
    )

    return intersections


def aggregate_road_features(
    grid: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Calculate road length, road segment count and density
    for every grid cell.
    """

    total_road_features = (
        intersections.groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            road_length_m=(
                "clipped_length_m",
                "sum",
            ),
            road_segment_count=(
                "road_id",
                "nunique",
            ),
        )
    )

    main_road_intersections = (
        intersections[
            intersections["is_main_road"]
        ].copy()
    )

    if main_road_intersections.empty:
        main_road_features = (
            pd.DataFrame(
                columns=[
                    "grid_id",
                    "main_road_length_m",
                    "main_road_segment_count",
                ]
            )
        )
    else:
        main_road_features = (
            main_road_intersections
            .groupby(
                "grid_id",
                as_index=False,
            )
            .agg(
                main_road_length_m=(
                    "clipped_length_m",
                    "sum",
                ),
                main_road_segment_count=(
                    "road_id",
                    "nunique",
                ),
            )
        )

    features = grid.merge(
        total_road_features,
        on="grid_id",
        how="left",
    )

    features = features.merge(
        main_road_features,
        on="grid_id",
        how="left",
    )

    fill_columns = [
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
    ]

    features[fill_columns] = (
        features[fill_columns].fillna(0)
    )

    features[
        "road_segment_count"
    ] = features[
        "road_segment_count"
    ].astype(int)

    features[
        "main_road_segment_count"
    ] = features[
        "main_road_segment_count"
    ].astype(int)

    features[
        "road_length_m"
    ] = features[
        "road_length_m"
    ].round(2)

    features[
        "main_road_length_m"
    ] = features[
        "main_road_length_m"
    ].round(2)

    grid_area_km2 = (
        features["cell_area_m2"]
        / 1_000_000
    )

    features[
        "road_density_km_per_km2"
    ] = (
        (
            features["road_length_m"]
            / 1_000
        )
        / grid_area_km2
    ).round(4)

    return gpd.GeoDataFrame(
        features,
        geometry="geometry",
        crs=grid.crs,
    )


def add_main_road_distance_features(
    features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Calculate the distance from every grid centroid
    to the nearest main road.
    """

    main_roads = roads[
        roads["is_main_road"]
    ].copy()

    if main_roads.empty:
        raise RuntimeError(
            "No main roads were identified "
            "in the downloaded road network."
        )

    grid_centers = features[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    grid_centers["geometry"] = (
        grid_centers.geometry.centroid
    )

    nearest_matches = (
        gpd.sjoin_nearest(
            grid_centers,
            main_roads[
                [
                    "road_id",
                    "highway",
                    "geometry",
                ]
            ],
            how="left",
            distance_col=(
                "distance_to_main_road_m"
            ),
        )
    )

    nearest_matches = (
        nearest_matches.sort_values(
            by=[
                "grid_id",
                "distance_to_main_road_m",
                "road_id",
            ],
            na_position="last",
        )
    )

    nearest_matches = (
        nearest_matches.drop_duplicates(
            subset=["grid_id"],
            keep="first",
        )
    )

    nearest_features = (
        nearest_matches[
            [
                "grid_id",
                "distance_to_main_road_m",
                "highway",
            ]
        ]
        .rename(
            columns={
                "highway": (
                    "nearest_main_road_type"
                ),
            }
        )
    )

    result = features.merge(
        nearest_features,
        on="grid_id",
        how="left",
    )

    result[
        "distance_to_main_road_m"
    ] = result[
        "distance_to_main_road_m"
    ].round(2)

    missing_distance_count = int(
        result[
            "distance_to_main_road_m"
        ].isna().sum()
    )

    if missing_distance_count > 0:
        raise RuntimeError(
            "Some grid cells could not be "
            "matched to a main road. "
            f"Missing count: "
            f"{missing_distance_count}"
        )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=features.crs,
    )


def validate_features(
    features: gpd.GeoDataFrame,
) -> None:
    """Validate the generated road-feature dataset."""

    required_columns = {
        "grid_id",
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
        "road_density_km_per_km2",
        "distance_to_main_road_m",
        "nearest_main_road_type",
        "geometry",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required road feature columns "
            "are missing: "
            f"{sorted(missing_columns)}"
        )

    if features.empty:
        raise ValueError(
            "The road feature dataset is empty."
        )

    if (
        features["grid_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate grid identifiers "
            "were found."
        )

    numeric_columns = [
        "road_length_m",
        "road_segment_count",
        "main_road_length_m",
        "main_road_segment_count",
        "road_density_km_per_km2",
        "distance_to_main_road_m",
    ]

    for column in numeric_columns:
        values = (
            features[column].to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Column contains non-finite "
                f"values: {column}"
            )

        if (values < 0).any():
            raise ValueError(
                "Column contains negative "
                f"values: {column}"
            )

    if not features.geometry.is_valid.all():
        raise ValueError(
            "Invalid geometries were found "
            "in the road feature dataset."
        )

    print(
        "Road feature validation "
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
    roads: gpd.GeoDataFrame,
) -> None:
    """Create a road-density preview image."""

    figure, axis = plt.subplots(
        figsize=(12, 11)
    )

    features.plot(
        ax=axis,
        column=(
            "road_density_km_per_km2"
        ),
        legend=True,
        linewidth=0,
        alpha=0.9,
        legend_kwds={
            "label": (
                "Road density "
                "(km / km²)"
            ),
            "shrink": 0.65,
        },
    )

    roads.plot(
        ax=axis,
        linewidth=0.12,
        alpha=0.25,
    )

    main_roads = roads[
        roads["is_main_road"]
    ]

    main_roads.plot(
        ax=axis,
        linewidth=0.45,
        alpha=0.65,
    )

    axis.set_title(
        "VoltSight - "
        "Çankaya Grid Road Density"
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
        "Road feature preview saved: "
        f"{PREVIEW_OUTPUT_PATH}"
    )


def create_summary(
    features: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> None:
    """Create a Markdown road-feature summary."""

    total_road_length_km = (
        roads["edge_length_m"].sum()
        / 1_000
    )

    main_road_length_km = (
        roads.loc[
            roads["is_main_road"],
            "edge_length_m",
        ].sum()
        / 1_000
    )

    road_density = features[
        "road_density_km_per_km2"
    ]

    main_road_distance = features[
        "distance_to_main_road_m"
    ]

    zero_road_cell_count = int(
        (
            features["road_length_m"]
            == 0
        ).sum()
    )

    main_road_type_list = "\n".join(
        f"- `{road_type}`"
        for road_type in sorted(
            MAIN_ROAD_TYPES
        )
    )

    summary = f"""# Çankaya Road Feature Summary

## Source

- Road data source: OpenStreetMap
- Download method: OSMnx polygon road-network query
- Network type: `{NETWORK_TYPE}`
- Download buffer: {DOWNLOAD_BUFFER_METERS:,} meters
- Successful Overpass endpoint: {ox.settings.overpass_url}
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Road Network

- Physical road edge count: {len(roads):,}
- Main-road edge count: {int(roads["is_main_road"].sum()):,}
- Total downloaded road length: {total_road_length_km:,.2f} km
- Total downloaded main-road length: {main_road_length_km:,.2f} km

## Grid Features

- Grid cell count: {len(features):,}
- Cells without a road segment: {zero_road_cell_count:,}
- Mean road density: {road_density.mean():,.2f} km/km²
- Median road density: {road_density.median():,.2f} km/km²
- Maximum road density: {road_density.max():,.2f} km/km²
- Mean distance to a main road: {main_road_distance.mean():,.2f} m
- Median distance to a main road: {main_road_distance.median():,.2f} m
- Maximum distance to a main road: {main_road_distance.max():,.2f} m

## Main-Road Classification

{main_road_type_list}

## Generated Features

- `road_length_m`
- `road_segment_count`
- `main_road_length_m`
- `main_road_segment_count`
- `road_density_km_per_km2`
- `distance_to_main_road_m`
- `nearest_main_road_type`

## Generated Outputs

- `data/interim/cankaya_drive_roads.gpkg`
- `data/processed/cankaya_grid_road_features.gpkg`
- `data/processed/cankaya_grid_road_features.geojson`
- `data/processed/cankaya_grid_road_features.csv`
- `docs/cankaya_road_features_preview.png`

## Method

The OpenStreetMap drive network was downloaded for the Çankaya
administrative boundary with an additional one-kilometer buffer.

The directed graph was converted to an undirected physical road
graph to avoid counting reciprocal travel directions as separate
physical roads.

Road geometries were intersected with each 250 x 250 meter grid
cell. Only the road length inside the corresponding grid cell was
included in that cell's feature value.

Distance to a main road was calculated from every grid-cell
centroid in the projected meter-based coordinate system.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "Road feature summary saved: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


def print_feature_statistics(
    features: gpd.GeoDataFrame,
) -> None:
    """Print important output statistics."""

    print("-" * 70)

    cells_with_roads = int(
        (
            features["road_length_m"]
            > 0
        ).sum()
    )

    cells_without_roads = int(
        (
            features["road_length_m"]
            == 0
        ).sum()
    )

    mean_road_density = (
        features[
            "road_density_km_per_km2"
        ].mean()
    )

    median_main_road_distance = (
        features[
            "distance_to_main_road_m"
        ].median()
    )

    maximum_main_road_distance = (
        features[
            "distance_to_main_road_m"
        ].max()
    )

    print(
        "Grid cells with road data: "
        f"{cells_with_roads:,}"
    )

    print(
        "Grid cells without road data: "
        f"{cells_without_roads:,}"
    )

    print(
        "Mean road density: "
        f"{mean_road_density:,.2f} "
        "km/km²"
    )

    print(
        "Median distance to main road: "
        f"{median_main_road_distance:,.2f} m"
    )

    print(
        "Maximum distance to main road: "
        f"{maximum_main_road_distance:,.2f} m"
    )


def main() -> None:
    """Run the Çankaya road-feature pipeline."""

    print("=" * 70)

    print(
        "VoltSight - "
        "Çankaya Road Feature Pipeline"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_files()
    configure_osmnx()

    boundary, grid = load_study_data()

    download_polygon = (
        create_download_polygon(
            boundary
        )
    )

    road_graph = (
        download_road_network(
            download_polygon,
            grid.crs,
        )
    )

    roads = prepare_road_edges(
        road_graph
    )

    save_road_edges(
        roads
    )

    intersections = (
        intersect_roads_with_grid(
            grid,
            roads,
        )
    )

    features = (
        aggregate_road_features(
            grid,
            intersections,
        )
    )

    features = (
        add_main_road_distance_features(
            features,
            roads,
        )
    )

    validate_features(
        features
    )

    save_feature_outputs(
        features
    )

    create_preview(
        features,
        roads,
    )

    create_summary(
        features,
        roads,
    )

    print_feature_statistics(
        features
    )

    print("=" * 70)

    print(
        "Road feature pipeline "
        "completed successfully."
    )

    print(
        f"Feature row count: "
        f"{len(features):,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main() 