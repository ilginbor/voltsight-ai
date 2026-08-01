from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )

from voltsight.core.study_areas import (  # noqa: E402
    SUPPORTED_GRID_SIZES_METERS,
    StudyAreaConfig,
    StudyAreaPaths,
    build_study_area_paths,
    get_study_area,
    list_study_area_keys,
)


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


def resolve_pipeline_paths(
    study_area: StudyAreaConfig,
    requested_grid_size_m: int | None = None,
) -> StudyAreaPaths:
    """
    Resolve deterministic paths while preserving legacy Çankaya outputs.

    The original project generated:

    - docs/cankaya_grid_preview.png
    - docs/cankaya_grid_summary.md

    These names remain unchanged for the default 250-metre Çankaya run.
    """

    paths = build_study_area_paths(
        PROJECT_ROOT,
        study_area,
        requested_grid_size_m,
    )

    if (
        study_area.key == "cankaya"
        and paths.grid_size_m == 250
    ):
        return StudyAreaPaths(
            boundary_geojson=paths.boundary_geojson,
            grid_gpkg=paths.grid_gpkg,
            grid_geojson=paths.grid_geojson,
            grid_preview_png=(
                DOCS_DIR
                / "cankaya_grid_preview.png"
            ),
            grid_summary_md=(
                DOCS_DIR
                / "cankaya_grid_summary.md"
            ),
            cache_directory=paths.cache_directory,
            grid_layer_name=paths.grid_layer_name,
            grid_size_m=paths.grid_size_m,
        )

    return paths


CANKAYA_CONFIG = get_study_area(
    "cankaya"
)

CANKAYA_PATHS = resolve_pipeline_paths(
    CANKAYA_CONFIG
)

BOUNDARY_OUTPUT_PATH = (
    CANKAYA_PATHS.boundary_geojson
)

GRID_GPKG_OUTPUT_PATH = (
    CANKAYA_PATHS.grid_gpkg
)

GRID_GEOJSON_OUTPUT_PATH = (
    CANKAYA_PATHS.grid_geojson
)

PREVIEW_OUTPUT_PATH = (
    CANKAYA_PATHS.grid_preview_png
)

SUMMARY_OUTPUT_PATH = (
    CANKAYA_PATHS.grid_summary_md
)

GRID_SIZE_METERS = (
    CANKAYA_PATHS.grid_size_m
)

PLACE_QUERIES: list[str] = list(
    CANKAYA_CONFIG.osm_place_queries
)


def create_output_directories(
    paths: StudyAreaPaths | None = None,
) -> None:
    """Create directories required by the grid pipeline."""

    selected_paths = (
        paths
        if paths is not None
        else CANKAYA_PATHS
    )

    directories = {
        selected_paths.boundary_geojson.parent,
        selected_paths.grid_gpkg.parent,
        selected_paths.grid_geojson.parent,
        selected_paths.grid_preview_png.parent,
        selected_paths.grid_summary_md.parent,
        selected_paths.cache_directory,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def configure_osmnx(
    cache_directory: Path | None = None,
) -> None:
    """Configure OSMnx caching and console logging."""

    selected_cache = (
        cache_directory
        if cache_directory is not None
        else PROJECT_ROOT / "cache"
    )

    selected_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.cache_folder = str(
        selected_cache
    )

    ox.settings.http_user_agent = (
        "VoltSight/0.2 "
        "educational-research-project"
    )


def get_polygon_result(
    result: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep valid Polygon and MultiPolygon geometries."""

    if result.empty:
        return result.copy()

    polygon_result = result.loc[
        result.geometry.notna()
    ].copy()

    if polygon_result.empty:
        return polygon_result

    polygon_result = polygon_result.loc[
        ~polygon_result.geometry.is_empty
    ].copy()

    return polygon_result.loc[
        polygon_result.geometry.geom_type.isin(
            [
                "Polygon",
                "MultiPolygon",
            ]
        )
    ].copy()


def calculate_boundary_area_km2(
    boundary: gpd.GeoDataFrame,
    projected_crs: str,
) -> float:
    """Calculate unioned boundary area in square kilometres."""

    if boundary.empty:
        raise ValueError(
            "Boundary dataset is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "Boundary dataset has no CRS."
        )

    projected = boundary.to_crs(
        projected_crs
    )

    unioned_geometry = (
        projected.geometry.union_all()
    )

    if unioned_geometry.is_empty:
        raise ValueError(
            "Boundary union geometry is empty."
        )

    return float(
        unioned_geometry.area
        / 1_000_000.0
    )


def validate_boundary(
    boundary: gpd.GeoDataFrame,
    study_area: StudyAreaConfig,
    source_description: str,
) -> float:
    """Validate geometry and expected administrative area."""

    if boundary.empty:
        raise ValueError(
            "Boundary dataset is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "Boundary dataset has no coordinate system."
        )

    polygon_result = get_polygon_result(
        boundary
    )

    if polygon_result.empty:
        raise ValueError(
            "Boundary dataset contains no polygon geometry."
        )

    area_km2 = calculate_boundary_area_km2(
        polygon_result,
        study_area.projected_crs,
    )

    if not (
        study_area.minimum_expected_area_km2
        <= area_km2
        <= study_area.maximum_expected_area_km2
    ):
        raise ValueError(
            f"{source_description} produced an unexpected "
            f"boundary area of {area_km2:,.2f} km². "
            f"Expected between "
            f"{study_area.minimum_expected_area_km2:,.2f} "
            f"and "
            f"{study_area.maximum_expected_area_km2:,.2f} km²."
        )

    return area_km2


def fetch_study_area_boundary(
    study_area: StudyAreaConfig,
) -> tuple[gpd.GeoDataFrame, str]:
    """
    Download and validate one administrative boundary.

    Multiple place queries are attempted. Area validation prevents a city
    boundary from being accepted accidentally when the requested area is the
    full Ankara province.
    """

    errors: list[str] = []

    for query in study_area.osm_place_queries:
        try:
            print(
                f"Trying boundary query: {query}"
            )

            result = ox.geocode_to_gdf(
                query
            )

            polygon_result = get_polygon_result(
                result
            )

            if polygon_result.empty:
                errors.append(
                    "Query returned no polygon geometry: "
                    f"{query}"
                )
                continue

            boundary_geometry = (
                polygon_result.geometry.union_all()
            )

            selected_row = (
                polygon_result.iloc[0]
            )

            display_name = str(
                selected_row.get(
                    "display_name",
                    query,
                )
            )

            boundary = gpd.GeoDataFrame(
                {
                    "name": [
                        study_area.display_name
                    ],
                    "study_area_key": [
                        study_area.key
                    ],
                    "boundary_scope": [
                        study_area.boundary_scope
                    ],
                    "source": [
                        "OpenStreetMap / Nominatim"
                    ],
                    "display_name": [
                        display_name
                    ],
                    "osm_id": [
                        selected_row.get(
                            "osm_id"
                        )
                    ],
                    "osm_type": [
                        selected_row.get(
                            "osm_type"
                        )
                    ],
                    "boundary_type": [
                        selected_row.get(
                            "type"
                        )
                    ],
                    "boundary_class": [
                        selected_row.get(
                            "class"
                        )
                    ],
                    "query_used": [
                        str(query)
                    ],
                    "retrieved_at_utc": [
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ],
                },
                geometry=[
                    boundary_geometry
                ],
                crs=polygon_result.crs,
            )

            area_km2 = validate_boundary(
                boundary,
                study_area,
                source_description=(
                    f"Boundary query {query!r}"
                ),
            )

            print(
                f"Boundary selected: {display_name}"
            )

            print(
                "Validated boundary area: "
                f"{area_km2:,.2f} km²"
            )

            return boundary, display_name

        except Exception as error:
            message = (
                f"{query}: {error}"
            )

            errors.append(
                message
            )

            print(
                f"Query failed: {error}"
            )

    error_details = "\n".join(
        f"- {message}"
        for message in errors
    )

    raise RuntimeError(
        f"{study_area.display_name} boundary "
        "could not be downloaded.\n"
        f"Attempt details:\n{error_details}"
    )


def fetch_cankaya_boundary() -> tuple[
    gpd.GeoDataFrame,
    str,
]:
    """Compatibility wrapper for the original Çankaya function."""

    return fetch_study_area_boundary(
        CANKAYA_CONFIG
    )


def load_saved_boundary(
    path: Path,
    study_area: StudyAreaConfig,
) -> tuple[gpd.GeoDataFrame, str]:
    """Load and validate a previously saved boundary."""

    if not path.exists():
        raise FileNotFoundError(
            f"Saved boundary was not found: {path}"
        )

    boundary = gpd.read_file(
        path
    )

    area_km2 = validate_boundary(
        boundary,
        study_area,
        source_description=(
            f"Saved boundary {path.name!r}"
        ),
    )

    display_name = (
        str(
            boundary[
                "display_name"
            ].iloc[0]
        )
        if (
            "display_name"
            in boundary.columns
            and boundary[
                "display_name"
            ].notna().any()
        )
        else study_area.display_name
    )

    print(
        f"Existing boundary loaded: {path}"
    )

    print(
        "Validated boundary area: "
        f"{area_km2:,.2f} km²"
    )

    return boundary, display_name


def save_boundary(
    boundary: gpd.GeoDataFrame,
    output_path: Path | None = None,
) -> None:
    """Save a boundary in WGS84 GeoJSON format."""

    selected_output_path = (
        output_path
        if output_path is not None
        else BOUNDARY_OUTPUT_PATH
    )

    selected_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    boundary_wgs84 = boundary.to_crs(
        epsg=4326
    )

    boundary_wgs84.to_file(
        selected_output_path,
        driver="GeoJSON",
    )

    print(
        f"Boundary saved: {selected_output_path}"
    )


def load_or_fetch_boundary(
    study_area: StudyAreaConfig,
    paths: StudyAreaPaths,
    *,
    reuse_boundary: bool,
    force_download: bool,
) -> tuple[gpd.GeoDataFrame, str]:
    """Reuse a validated boundary or download a fresh copy."""

    if reuse_boundary and force_download:
        raise ValueError(
            "--reuse-boundary and --force-download "
            "cannot be used together."
        )

    if force_download:
        print(
            "Force-download requested. "
            "Ignoring any existing boundary."
        )

        boundary, display_name = (
            fetch_study_area_boundary(
                study_area
            )
        )

        save_boundary(
            boundary,
            paths.boundary_geojson,
        )

        return boundary, display_name

    if paths.boundary_geojson.exists():
        try:
            return load_saved_boundary(
                paths.boundary_geojson,
                study_area,
            )
        except Exception:
            if reuse_boundary:
                raise

            print(
                "Existing boundary failed validation. "
                "A new boundary will be downloaded."
            )

    elif reuse_boundary:
        raise FileNotFoundError(
            "--reuse-boundary was requested, but "
            "the boundary file does not exist:\n"
            f"{paths.boundary_geojson}"
        )

    boundary, display_name = (
        fetch_study_area_boundary(
            study_area
        )
    )

    save_boundary(
        boundary,
        paths.boundary_geojson,
    )

    return boundary, display_name


def project_boundary(
    boundary: gpd.GeoDataFrame,
    study_area: StudyAreaConfig | None = None,
) -> tuple[gpd.GeoDataFrame, Any]:
    """Project the boundary into its configured metric CRS."""

    selected_study_area = (
        study_area
        if study_area is not None
        else CANKAYA_CONFIG
    )

    projected_crs = (
        selected_study_area.projected_crs
    )

    boundary_projected = boundary.to_crs(
        projected_crs
    )

    if boundary_projected.empty:
        raise RuntimeError(
            "Projected boundary is empty."
        )

    if (
        boundary_projected.geometry.isna().any()
        or boundary_projected.geometry.is_empty.any()
    ):
        raise RuntimeError(
            "Projected boundary contains invalid geometry."
        )

    print(
        f"Analysis CRS selected: {projected_crs}"
    )

    return (
        boundary_projected,
        projected_crs,
    )


def generate_square_grid(
    boundary_projected: gpd.GeoDataFrame,
    projected_crs: Any,
    grid_size: int,
    grid_prefix: str = "CKY",
    district_name: str | None = "Çankaya",
    city_name: str = "Ankara",
) -> gpd.GeoDataFrame:
    """
    Generate fixed-size square cells.

    A cell is retained when its centroid falls inside or touches the
    administrative boundary. Every retained cell therefore keeps the exact
    requested square dimensions.
    """

    if grid_size <= 0:
        raise ValueError(
            "Grid size must be positive."
        )

    if boundary_projected.empty:
        raise ValueError(
            "Projected boundary is empty."
        )

    if boundary_projected.crs is None:
        raise ValueError(
            "Projected boundary has no CRS."
        )

    min_x, min_y, max_x, max_y = (
        boundary_projected.total_bounds
    )

    start_x = int(
        math.floor(
            min_x / grid_size
        )
        * grid_size
    )

    start_y = int(
        math.floor(
            min_y / grid_size
        )
        * grid_size
    )

    end_x = int(
        math.ceil(
            max_x / grid_size
        )
        * grid_size
    )

    end_y = int(
        math.ceil(
            max_y / grid_size
        )
        * grid_size
    )

    cells = [
        box(
            x_coordinate,
            y_coordinate,
            x_coordinate + grid_size,
            y_coordinate + grid_size,
        )
        for x_coordinate in range(
            start_x,
            end_x,
            grid_size,
        )
        for y_coordinate in range(
            start_y,
            end_y,
            grid_size,
        )
    ]

    grid = gpd.GeoDataFrame(
        geometry=cells,
        crs=projected_crs,
    )

    study_geometry = (
        boundary_projected.geometry.union_all()
    )

    cell_centers = (
        grid.geometry.centroid
    )

    center_inside_boundary = (
        cell_centers.within(
            study_geometry
        )
        | cell_centers.touches(
            study_geometry
        )
    )

    grid = grid.loc[
        center_inside_boundary
    ].copy()

    grid.reset_index(
        drop=True,
        inplace=True,
    )

    minimum_id_width = (
        6
        if grid_prefix == "ANK"
        else 5
    )

    id_width = max(
        minimum_id_width,
        len(
            str(
                max(
                    len(grid),
                    1,
                )
            )
        ),
    )

    grid.insert(
        0,
        "grid_id",
        [
            (
                f"{grid_prefix}_"
                f"{index:0{id_width}d}"
            )
            for index in range(
                1,
                len(grid) + 1,
            )
        ],
    )

    grid["district"] = (
        district_name
        if district_name is not None
        else "Province-wide"
    )

    grid["city"] = city_name
    grid["grid_size_m"] = grid_size

    grid["cell_area_m2"] = (
        grid.geometry.area.round(2)
    )

    grid_centers_wgs84 = gpd.GeoSeries(
        grid.geometry.centroid,
        crs=projected_crs,
    ).to_crs(
        epsg=4326
    )

    grid["center_longitude"] = (
        grid_centers_wgs84.x.round(6)
    )

    grid["center_latitude"] = (
        grid_centers_wgs84.y.round(6)
    )

    print(
        f"Generated grid cell count: {len(grid):,}"
    )

    return grid


def save_grid(
    grid_projected: gpd.GeoDataFrame,
    paths: StudyAreaPaths | None = None,
) -> None:
    """Save projected and web-compatible grid copies."""

    selected_paths = (
        paths
        if paths is not None
        else CANKAYA_PATHS
    )

    if selected_paths.grid_gpkg.exists():
        selected_paths.grid_gpkg.unlink()

    grid_projected.to_file(
        selected_paths.grid_gpkg,
        layer=(
            selected_paths.grid_layer_name
        ),
        driver="GPKG",
    )

    grid_wgs84 = grid_projected.to_crs(
        epsg=4326
    )

    grid_wgs84.to_file(
        selected_paths.grid_geojson,
        driver="GeoJSON",
    )

    print(
        "Projected grid saved: "
        f"{selected_paths.grid_gpkg}"
    )

    print(
        "Web grid saved: "
        f"{selected_paths.grid_geojson}"
    )


def create_preview(
    boundary_projected: gpd.GeoDataFrame,
    grid_projected: gpd.GeoDataFrame,
    study_area: StudyAreaConfig | None = None,
    paths: StudyAreaPaths | None = None,
) -> None:
    """Create a preview image for documentation and GitHub."""

    selected_study_area = (
        study_area
        if study_area is not None
        else CANKAYA_CONFIG
    )

    selected_paths = (
        paths
        if paths is not None
        else CANKAYA_PATHS
    )

    figure, axis = plt.subplots(
        figsize=(11, 11)
    )

    grid_projected.boundary.plot(
        ax=axis,
        linewidth=0.05,
        alpha=0.50,
    )

    boundary_projected.boundary.plot(
        ax=axis,
        linewidth=1.5,
    )

    grid_size = (
        selected_paths.grid_size_m
    )

    axis.set_title(
        "VoltSight - "
        f"{selected_study_area.display_name} "
        f"{grid_size} x {grid_size} Meter Study Grid"
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        selected_paths.grid_preview_png,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Preview image saved: "
        f"{selected_paths.grid_preview_png}"
    )


def project_relative_path(
    path: Path,
) -> str:
    """Return a platform-independent project-relative path."""

    try:
        relative = path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative = path

    return relative.as_posix()


def create_summary(
    boundary_projected: gpd.GeoDataFrame,
    grid_projected: gpd.GeoDataFrame,
    projected_crs: Any,
    display_name: str,
    study_area: StudyAreaConfig | None = None,
    paths: StudyAreaPaths | None = None,
) -> None:
    """Create a Markdown summary for project documentation."""

    selected_study_area = (
        study_area
        if study_area is not None
        else CANKAYA_CONFIG
    )

    selected_paths = (
        paths
        if paths is not None
        else CANKAYA_PATHS
    )

    grid_size = (
        selected_paths.grid_size_m
    )

    boundary_geometry = (
        boundary_projected.geometry.union_all()
    )

    boundary_area_km2 = (
        boundary_geometry.area
        / 1_000_000.0
    )

    grid_coverage_km2 = (
        grid_projected.geometry.area.sum()
        / 1_000_000.0
    )

    if selected_study_area.district_name:
        study_area_text = (
            f"{selected_study_area.district_name}, "
            f"{selected_study_area.city_name}, Türkiye"
        )

        area_label = (
            "Approximate district area"
        )
    else:
        study_area_text = (
            f"{selected_study_area.display_name} "
            "Province, Türkiye"
        )

        area_label = (
            "Approximate province area"
        )

    summary = f"""# {selected_study_area.display_name} Study Grid Summary

## Data Source

- Boundary source: OpenStreetMap / Nominatim
- Selected result: {display_name}
- Study-area key: `{selected_study_area.key}`
- Boundary scope: {selected_study_area.boundary_scope}
- Study area: {study_area_text}

## Grid Configuration

- Grid type: Fixed square grid
- Grid prefix: `{selected_study_area.grid_prefix}`
- Cell width: {grid_size:,} meters
- Cell height: {grid_size:,} meters
- Individual cell area: {grid_size * grid_size:,} square meters
- Generated cell count: {len(grid_projected):,}

## Coordinate Systems

- Download and web CRS: EPSG:4326
- Analysis CRS: {projected_crs}

## Area Information

- {area_label}: {boundary_area_km2:,.2f} square kilometers
- Total retained grid area: {grid_coverage_km2:,.2f} square kilometers

## Generated Outputs

- `{project_relative_path(selected_paths.boundary_geojson)}`
- `{project_relative_path(selected_paths.grid_gpkg)}`
- `{project_relative_path(selected_paths.grid_geojson)}`
- `{project_relative_path(selected_paths.grid_preview_png)}`

## Method

A square grid was generated over the administrative boundary bounding
box. A cell was retained when its centroid fell inside or touched the
selected boundary. This preserves a consistent
{grid_size} x {grid_size} meter square for every retained analysis
cell.

The boundary area was validated against the expected range configured
for `{selected_study_area.key}`. This prevents an Ankara city boundary
from being mistaken for the complete Ankara province boundary.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    selected_paths.grid_summary_md.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "Summary saved: "
        f"{selected_paths.grid_summary_md}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a validated administrative boundary "
            "and fixed-size study grid."
        )
    )

    parser.add_argument(
        "--study-area",
        choices=list_study_area_keys(),
        default="cankaya",
        help=(
            "Configured study area. "
            "Default: cankaya."
        ),
    )

    parser.add_argument(
        "--grid-size-m",
        type=int,
        choices=SUPPORTED_GRID_SIZES_METERS,
        default=None,
        help=(
            "Grid size in metres. "
            "Defaults to the selected study area's setting."
        ),
    )

    parser.add_argument(
        "--reuse-boundary",
        action="store_true",
        help=(
            "Require and reuse the existing validated "
            "boundary file without downloading."
        ),
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Ignore any existing boundary file and "
            "download the boundary again."
        ),
    )

    return parser


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = build_argument_parser()

    arguments = parser.parse_args(
        argv
    )

    if (
        arguments.reuse_boundary
        and arguments.force_download
    ):
        parser.error(
            "--reuse-boundary and --force-download "
            "cannot be used together."
        )

    return arguments


def run_pipeline(
    arguments: argparse.Namespace,
) -> gpd.GeoDataFrame:
    """Run one configured boundary and grid pipeline."""

    study_area = get_study_area(
        arguments.study_area
    )

    paths = resolve_pipeline_paths(
        study_area,
        arguments.grid_size_m,
    )

    print("=" * 70)

    print(
        "VoltSight - "
        f"{study_area.display_name} "
        "Study Grid Pipeline"
    )

    print("=" * 70)

    print(
        f"Study area key: {study_area.key}"
    )

    print(
        f"Boundary scope: {study_area.boundary_scope}"
    )

    print(
        f"Grid size: {paths.grid_size_m:,} m"
    )

    print(
        f"Grid prefix: {study_area.grid_prefix}"
    )

    print(
        f"Analysis CRS: {study_area.projected_crs}"
    )

    create_output_directories(
        paths
    )

    configure_osmnx(
        paths.cache_directory
    )

    boundary, display_name = (
        load_or_fetch_boundary(
            study_area,
            paths,
            reuse_boundary=(
                arguments.reuse_boundary
            ),
            force_download=(
                arguments.force_download
            ),
        )
    )

    (
        boundary_projected,
        projected_crs,
    ) = project_boundary(
        boundary,
        study_area,
    )

    grid_projected = generate_square_grid(
        boundary_projected=(
            boundary_projected
        ),
        projected_crs=(
            projected_crs
        ),
        grid_size=(
            paths.grid_size_m
        ),
        grid_prefix=(
            study_area.grid_prefix
        ),
        district_name=(
            study_area.district_name
        ),
        city_name=(
            study_area.city_name
        ),
    )

    if grid_projected.empty:
        raise RuntimeError(
            "The study grid is empty. "
            "Check the selected boundary."
        )

    save_grid(
        grid_projected,
        paths,
    )

    create_preview(
        boundary_projected=(
            boundary_projected
        ),
        grid_projected=(
            grid_projected
        ),
        study_area=study_area,
        paths=paths,
    )

    create_summary(
        boundary_projected=(
            boundary_projected
        ),
        grid_projected=(
            grid_projected
        ),
        projected_crs=(
            projected_crs
        ),
        display_name=(
            display_name
        ),
        study_area=study_area,
        paths=paths,
    )

    print("=" * 70)

    print(
        "Pipeline completed successfully."
    )

    print(
        f"Grid cells: {len(grid_projected):,}"
    )

    print(
        f"Analysis CRS: {projected_crs}"
    )

    print(
        f"GeoPackage: {paths.grid_gpkg}"
    )

    print("=" * 70)

    return grid_projected


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run a configured study-grid pipeline."""

    arguments = parse_arguments(
        argv
    )

    run_pipeline(
        arguments
    )


if __name__ == "__main__":
    main()
