from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

BOUNDARY_OUTPUT_PATH = RAW_DATA_DIR / "cankaya_boundary_osm.geojson"
GRID_GPKG_OUTPUT_PATH = PROCESSED_DATA_DIR / "cankaya_grid_250m.gpkg"
GRID_GEOJSON_OUTPUT_PATH = PROCESSED_DATA_DIR / "cankaya_grid_250m.geojson"
PREVIEW_OUTPUT_PATH = DOCS_DIR / "cankaya_grid_preview.png"
SUMMARY_OUTPUT_PATH = DOCS_DIR / "cankaya_grid_summary.md"

GRID_SIZE_METERS = 250

PLACE_QUERIES: list[str | dict[str, str]] = [
    {
        "county": "Çankaya",
        "state": "Ankara",
        "country": "Türkiye",
    },
    "Çankaya district, Ankara, Türkiye",
    "Çankaya, Ankara, Türkiye",
]


def create_output_directories() -> None:
    """Create all directories required by the data pipeline."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def configure_osmnx() -> None:
    """Configure OSMnx caching and console logging."""

    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.http_user_agent = "VoltSight/0.1 educational-research-project"


def get_polygon_result(
    result: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep valid Polygon and MultiPolygon geometries."""

    if result.empty:
        return result

    result = result[result.geometry.notna()].copy()

    return result[
        result.geometry.geom_type.isin(
            [
                "Polygon",
                "MultiPolygon",
            ]
        )
    ].copy()


def fetch_cankaya_boundary() -> tuple[gpd.GeoDataFrame, str]:
    """
    Download the Çankaya administrative boundary.

    Several queries are attempted because geocoding services may interpret
    administrative place names differently.
    """

    errors: list[str] = []

    for query in PLACE_QUERIES:
        try:
            print(f"Trying boundary query: {query}")

            result = ox.geocode_to_gdf(query)
            polygon_result = get_polygon_result(result)

            if polygon_result.empty:
                errors.append(
                    f"Query returned no polygon geometry: {query}"
                )
                continue

            boundary_geometry = polygon_result.geometry.union_all()

            selected_row = polygon_result.iloc[0]

            display_name = str(
                selected_row.get(
                    "display_name",
                    query,
                )
            )

            osm_id = selected_row.get("osm_id")
            osm_type = selected_row.get("osm_type")
            boundary_type = selected_row.get("type")
            boundary_class = selected_row.get("class")

            boundary = gpd.GeoDataFrame(
                {
                    "name": ["Çankaya"],
                    "source": ["OpenStreetMap / Nominatim"],
                    "display_name": [display_name],
                    "osm_id": [osm_id],
                    "osm_type": [osm_type],
                    "boundary_type": [boundary_type],
                    "boundary_class": [boundary_class],
                    "query_used": [str(query)],
                    "retrieved_at_utc": [
                        datetime.now(timezone.utc).isoformat()
                    ],
                },
                geometry=[boundary_geometry],
                crs=polygon_result.crs,
            )

            print(f"Boundary selected: {display_name}")

            return boundary, display_name

        except Exception as error:
            errors.append(f"{query}: {error}")
            print(f"Query failed: {error}")

    error_details = "\n".join(errors)

    raise RuntimeError(
        "Çankaya boundary could not be downloaded.\n"
        f"Attempt details:\n{error_details}"
    )


def save_boundary(boundary: gpd.GeoDataFrame) -> None:
    """Save the boundary in WGS84 GeoJSON format."""

    boundary_wgs84 = boundary.to_crs(epsg=4326)

    boundary_wgs84.to_file(
        BOUNDARY_OUTPUT_PATH,
        driver="GeoJSON",
    )

    print(f"Boundary saved: {BOUNDARY_OUTPUT_PATH}")


def project_boundary(
    boundary: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Any]:
    """
    Project the district boundary into a local UTM coordinate system.

    Geographic coordinates use degrees. Grid generation requires a projected
    CRS whose coordinate units are meters.
    """

    projected_crs = boundary.estimate_utm_crs()

    if projected_crs is None:
        raise RuntimeError(
            "A suitable projected coordinate system could not be estimated."
        )

    boundary_projected = boundary.to_crs(projected_crs)

    print(f"Analysis CRS selected: {projected_crs}")

    return boundary_projected, projected_crs


def generate_square_grid(
    boundary_projected: gpd.GeoDataFrame,
    projected_crs: Any,
    grid_size: int,
) -> gpd.GeoDataFrame:
    """
    Generate fixed-size square cells.

    Cells are retained when their center point falls inside the Çankaya
    boundary. This keeps every retained analysis cell exactly square.
    """

    min_x, min_y, max_x, max_y = boundary_projected.total_bounds

    start_x = int(math.floor(min_x / grid_size) * grid_size)
    start_y = int(math.floor(min_y / grid_size) * grid_size)

    end_x = int(math.ceil(max_x / grid_size) * grid_size)
    end_y = int(math.ceil(max_y / grid_size) * grid_size)

    cells = [
        box(
            x_coordinate,
            y_coordinate,
            x_coordinate + grid_size,
            y_coordinate + grid_size,
        )
        for x_coordinate in range(start_x, end_x, grid_size)
        for y_coordinate in range(start_y, end_y, grid_size)
    ]

    grid = gpd.GeoDataFrame(
        geometry=cells,
        crs=projected_crs,
    )

    district_geometry = boundary_projected.geometry.iloc[0]
    cell_centers = grid.geometry.centroid

    center_inside_boundary = cell_centers.within(
        district_geometry
    ) | cell_centers.touches(district_geometry)

    grid = grid.loc[center_inside_boundary].copy()
    grid.reset_index(drop=True, inplace=True)

    grid.insert(
        0,
        "grid_id",
        [
            f"CKY_{index:05d}"
            for index in range(1, len(grid) + 1)
        ],
    )

    grid["district"] = "Çankaya"
    grid["city"] = "Ankara"
    grid["grid_size_m"] = grid_size
    grid["cell_area_m2"] = grid.geometry.area.round(2)

    grid_centers_wgs84 = gpd.GeoSeries(
        grid.geometry.centroid,
        crs=projected_crs,
    ).to_crs(epsg=4326)

    grid["center_longitude"] = grid_centers_wgs84.x.round(6)
    grid["center_latitude"] = grid_centers_wgs84.y.round(6)

    print(f"Generated grid cell count: {len(grid):,}")

    return grid


def save_grid(grid_projected: gpd.GeoDataFrame) -> None:
    """Save projected and web-compatible copies of the study grid."""

    if GRID_GPKG_OUTPUT_PATH.exists():
        GRID_GPKG_OUTPUT_PATH.unlink()

    grid_projected.to_file(
        GRID_GPKG_OUTPUT_PATH,
        layer="cankaya_grid_250m",
        driver="GPKG",
    )

    grid_wgs84 = grid_projected.to_crs(epsg=4326)

    grid_wgs84.to_file(
        GRID_GEOJSON_OUTPUT_PATH,
        driver="GeoJSON",
    )

    print(f"Projected grid saved: {GRID_GPKG_OUTPUT_PATH}")
    print(f"Web grid saved: {GRID_GEOJSON_OUTPUT_PATH}")


def create_preview(
    boundary_projected: gpd.GeoDataFrame,
    grid_projected: gpd.GeoDataFrame,
) -> None:
    """Create a preview image for documentation and GitHub."""

    figure, axis = plt.subplots(figsize=(11, 11))

    grid_projected.boundary.plot(
        ax=axis,
        linewidth=0.08,
        alpha=0.55,
    )

    boundary_projected.boundary.plot(
        ax=axis,
        linewidth=1.5,
    )

    axis.set_title(
        "VoltSight - Çankaya 250 x 250 Meter Study Grid"
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

    print(f"Preview image saved: {PREVIEW_OUTPUT_PATH}")


def create_summary(
    boundary_projected: gpd.GeoDataFrame,
    grid_projected: gpd.GeoDataFrame,
    projected_crs: Any,
    display_name: str,
) -> None:
    """Create a Markdown summary for project documentation."""

    district_area_km2 = (
        boundary_projected.geometry.area.sum() / 1_000_000
    )

    grid_coverage_km2 = (
        grid_projected.geometry.area.sum() / 1_000_000
    )

    summary = f"""# Çankaya Study Grid Summary

## Data Source

- Boundary source: OpenStreetMap / Nominatim
- Selected result: {display_name}
- Study area: Çankaya, Ankara, Türkiye

## Grid Configuration

- Grid type: Fixed square grid
- Cell width: {GRID_SIZE_METERS} meters
- Cell height: {GRID_SIZE_METERS} meters
- Individual cell area: {GRID_SIZE_METERS * GRID_SIZE_METERS:,} square meters
- Generated cell count: {len(grid_projected):,}

## Coordinate Systems

- Download and web CRS: EPSG:4326
- Analysis CRS: {projected_crs}

## Area Information

- Approximate district area: {district_area_km2:,.2f} square kilometers
- Total retained grid area: {grid_coverage_km2:,.2f} square kilometers

## Generated Outputs

- `data/raw/cankaya_boundary_osm.geojson`
- `data/processed/cankaya_grid_250m.gpkg`
- `data/processed/cankaya_grid_250m.geojson`
- `docs/cankaya_grid_preview.png`

## Method

A square grid was generated over the district bounding box. A grid
cell was retained when its center point fell inside the Çankaya
administrative boundary. This approach preserves a consistent
250 x 250 meter shape for every analysis cell.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(f"Summary saved: {SUMMARY_OUTPUT_PATH}")


def main() -> None:
    """Run the Çankaya boundary and grid generation pipeline."""

    print("=" * 70)
    print("VoltSight - Çankaya Study Grid Pipeline")
    print("=" * 70)

    create_output_directories()
    configure_osmnx()

    boundary, display_name = fetch_cankaya_boundary()

    save_boundary(boundary)

    boundary_projected, projected_crs = project_boundary(
        boundary
    )

    grid_projected = generate_square_grid(
        boundary_projected=boundary_projected,
        projected_crs=projected_crs,
        grid_size=GRID_SIZE_METERS,
    )

    if grid_projected.empty:
        raise RuntimeError(
            "The study grid is empty. Check the downloaded boundary."
        )

    save_grid(grid_projected)

    create_preview(
        boundary_projected=boundary_projected,
        grid_projected=grid_projected,
    )

    create_summary(
        boundary_projected=boundary_projected,
        grid_projected=grid_projected,
        projected_crs=projected_crs,
        display_name=display_name,
    )

    print("=" * 70)
    print("Pipeline completed successfully.")
    print(f"Grid cells: {len(grid_projected):,}")
    print(f"Analysis CRS: {projected_crs}")
    print("=" * 70)


if __name__ == "__main__":
    main()