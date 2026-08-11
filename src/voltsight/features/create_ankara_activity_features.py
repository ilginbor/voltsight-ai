from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_charging_features.gpkg"
)

POI_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_activity_pois_osm.gpkg"
)

OUTPUT_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_activity_features.gpkg"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_activity_features.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_activity_features_summary.md"
)

GRID_LAYER_NAME = "grid_charging_features"
POI_LAYER_NAME = "activity_pois"
OUTPUT_LAYER_NAME = "grid_activity_features"

EXPECTED_CRS = "EPSG:32636"

GRID_CELL_SIZE_M = 500.0

NEIGHBORHOOD_RADII_M = (
    1_000.0,
    2_000.0,
)

CATEGORY_COLUMNS = {
    "retail_commercial": (
        "is_retail_commercial"
    ),
    "education": (
        "is_education"
    ),
    "healthcare": (
        "is_healthcare"
    ),
    "transport_activity": (
        "is_transport_activity"
    ),
}

OUTPUT_FEATURE_COLUMNS = (
    "poi_count",
    "retail_commercial_count",
    "education_count",
    "healthcare_count",
    "transport_activity_count",
    "poi_count_within_1000m",
    "poi_count_within_2000m",
    "retail_commercial_within_1000m",
    "education_within_1000m",
    "healthcare_within_1000m",
    "transport_activity_within_1000m",
)


def coerce_boolean_series(
    values: pd.Series,
) -> pd.Series:
    """Coerce common GeoPackage boolean representations safely."""

    if pd.api.types.is_bool_dtype(
        values.dtype
    ):
        return values.astype(
            bool
        )

    if pd.api.types.is_numeric_dtype(
        values.dtype
    ):
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )

        if numeric.isna().any():
            raise ValueError(
                "Boolean category column contains invalid numeric values."
            )

        if not numeric.isin(
            [
                0,
                1,
            ]
        ).all():
            raise ValueError(
                "Boolean category column must contain only 0/1 values."
            )

        return numeric.astype(
            bool
        )

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    converted = normalized.map(
        mapping
    )

    if converted.isna().any():
        bad_values = sorted(
            set(
                normalized.loc[
                    converted.isna()
                ]
            )
        )

        raise ValueError(
            "Unrecognized boolean category values: "
            f"{bad_values}"
        )

    return converted.astype(
        bool
    )


def load_grid() -> gpd.GeoDataFrame:
    """Load and validate the Ankara 500-m analysis grid."""

    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Ankara grid not found: {GRID_PATH}"
        )

    grid = gpd.read_file(
        GRID_PATH,
        layer=GRID_LAYER_NAME,
    )[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    if grid.empty:
        raise ValueError(
            "Ankara grid is empty."
        )

    if grid.crs is None:
        raise ValueError(
            "Ankara grid CRS is missing."
        )

    if (
        grid.crs.to_string().upper()
        != EXPECTED_CRS
    ):
        raise ValueError(
            "Unexpected Ankara grid CRS: "
            f"{grid.crs}"
        )

    grid[
        "grid_id"
    ] = grid[
        "grid_id"
    ].astype(str)

    if grid[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Ankara grid IDs were found."
        )

    if grid.geometry.isna().any():
        raise ValueError(
            "Missing Ankara grid geometry was found."
        )

    if (
        ~grid.geometry.is_valid
    ).any():
        raise ValueError(
            "Invalid Ankara grid geometry was found."
        )

    bounds = (
        grid.geometry.bounds
    )

    widths = (
        bounds[
            "maxx"
        ]
        - bounds[
            "minx"
        ]
    ).to_numpy(
        dtype=float
    )

    heights = (
        bounds[
            "maxy"
        ]
        - bounds[
            "miny"
        ]
    ).to_numpy(
        dtype=float
    )

    if not np.allclose(
        widths,
        GRID_CELL_SIZE_M,
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError(
            "Grid cells are not consistently 500 m wide."
        )

    if not np.allclose(
        heights,
        GRID_CELL_SIZE_M,
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError(
            "Grid cells are not consistently 500 m high."
        )

    return gpd.GeoDataFrame(
        grid,
        geometry="geometry",
        crs=grid.crs,
    )


def load_pois() -> gpd.GeoDataFrame:
    """Load and validate the downloaded OSM activity inventory."""

    if not POI_PATH.exists():
        raise FileNotFoundError(
            f"Ankara activity POIs not found: {POI_PATH}"
        )

    pois = gpd.read_file(
        POI_PATH,
        layer=POI_LAYER_NAME,
    )

    if pois.empty:
        raise ValueError(
            "Ankara activity POI inventory is empty."
        )

    if pois.crs is None:
        raise ValueError(
            "Ankara activity POI CRS is missing."
        )

    if (
        pois.crs.to_string().upper()
        != EXPECTED_CRS
    ):
        pois = pois.to_crs(
            EXPECTED_CRS
        )

    required = {
        "osm_uid",
        *CATEGORY_COLUMNS.values(),
        "geometry",
    }

    missing = (
        required
        - set(
            pois.columns
        )
    )

    if missing:
        raise ValueError(
            "Activity POI columns are missing: "
            f"{sorted(missing)}"
        )

    pois[
        "osm_uid"
    ] = pois[
        "osm_uid"
    ].astype(str)

    if pois[
        "osm_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate activity POI IDs were found."
        )

    if pois.geometry.isna().any():
        raise ValueError(
            "Missing activity POI geometry was found."
        )

    if pois.geometry.is_empty.any():
        raise ValueError(
            "Empty activity POI geometry was found."
        )

    if not pois.geometry.is_valid.all():
        raise ValueError(
            "Invalid activity POI geometry was found."
        )

    for column in CATEGORY_COLUMNS.values():
        pois[
            column
        ] = coerce_boolean_series(
            pois[
                column
            ]
        )

    if not pois[
        list(
            CATEGORY_COLUMNS.values()
        )
    ].any(
        axis=1
    ).all():
        raise ValueError(
            "At least one activity POI has no category."
        )

    return gpd.GeoDataFrame(
        pois,
        geometry="geometry",
        crs=pois.crs,
    )


def count_points_within_radius(
    centers_xy: np.ndarray,
    poi_xy: np.ndarray,
    *,
    radius_m: float,
) -> np.ndarray:
    """Count POI points within a Euclidean radius of every grid center."""

    centers = np.asarray(
        centers_xy,
        dtype=float,
    )

    points = np.asarray(
        poi_xy,
        dtype=float,
    )

    if (
        centers.ndim != 2
        or centers.shape[
            1
        ]
        != 2
    ):
        raise ValueError(
            "centers_xy must have shape (n, 2)."
        )

    if (
        points.ndim != 2
        or points.shape[
            1
        ]
        != 2
    ):
        raise ValueError(
            "poi_xy must have shape (m, 2)."
        )

    if radius_m < 0:
        raise ValueError(
            "Neighborhood radius cannot be negative."
        )

    if not np.isfinite(
        centers
    ).all():
        raise ValueError(
            "Grid center coordinates contain non-finite values."
        )

    if not np.isfinite(
        points
    ).all():
        raise ValueError(
            "POI coordinates contain non-finite values."
        )

    if len(
        points
    ) == 0:
        return np.zeros(
            len(
                centers
            ),
            dtype=np.int64,
        )

    tree = cKDTree(
        points
    )

    counts = tree.query_ball_point(
        centers,
        r=float(
            radius_m
        ),
        return_length=True,
    )

    return np.asarray(
        counts,
        dtype=np.int64,
    )


def assign_local_grid_ids(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Assign each POI to at most one Ankara grid cell for local counts.

    `intersects` safely includes a point exactly on a polygon edge. If such a
    point intersects more than one adjacent grid cell, the lexicographically
    smallest grid ID is selected deterministically so the POI is counted once.
    POIs in the 2.5-km download buffer but outside the Ankara grid remain
    available for neighborhood counts but receive no local grid ID.
    """

    joined = gpd.sjoin(
        pois[
            [
                "osm_uid",
                *CATEGORY_COLUMNS.values(),
                "geometry",
            ]
        ],
        grid[
            [
                "grid_id",
                "geometry",
            ]
        ],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return pd.DataFrame(
            columns=[
                "osm_uid",
                "grid_id",
                *CATEGORY_COLUMNS.values(),
            ]
        )

    joined[
        "grid_id"
    ] = joined[
        "grid_id"
    ].astype(str)

    joined = (
        joined.sort_values(
            [
                "osm_uid",
                "grid_id",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "osm_uid",
            ],
            keep="first",
        )
    )

    return pd.DataFrame(
        joined[
            [
                "osm_uid",
                "grid_id",
                *CATEGORY_COLUMNS.values(),
            ]
        ]
    )


def calculate_local_counts(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate unique total and category POI counts inside each grid cell."""

    assignments = (
        assign_local_grid_ids(
            grid,
            pois,
        )
    )

    result = pd.DataFrame(
        {
            "grid_id": (
                grid[
                    "grid_id"
                ].astype(str)
            ),
        }
    )

    result[
        "poi_count"
    ] = 0

    for category_name in (
        CATEGORY_COLUMNS
    ):
        result[
            f"{category_name}_count"
        ] = 0

    if assignments.empty:
        return result

    total_counts = (
        assignments.groupby(
            "grid_id"
        )[
            "osm_uid"
        ]
        .nunique()
        .rename(
            "poi_count"
        )
    )

    result = (
        result.drop(
            columns=[
                "poi_count",
            ]
        )
        .merge(
            total_counts,
            on="grid_id",
            how="left",
        )
    )

    result[
        "poi_count"
    ] = (
        result[
            "poi_count"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    for (
        category_name,
        flag_column,
    ) in CATEGORY_COLUMNS.items():
        category_counts = (
            assignments.loc[
                assignments[
                    flag_column
                ].astype(
                    bool
                )
            ]
            .groupby(
                "grid_id"
            )[
                "osm_uid"
            ]
            .nunique()
            .rename(
                f"{category_name}_count"
            )
        )

        result = (
            result.drop(
                columns=[
                    f"{category_name}_count",
                ]
            )
            .merge(
                category_counts,
                on="grid_id",
                how="left",
            )
        )

        result[
            f"{category_name}_count"
        ] = (
            result[
                f"{category_name}_count"
            ]
            .fillna(
                0
            )
            .astype(
                int
            )
        )

    return result


def calculate_neighborhood_counts(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate 1-km and 2-km representative-point activity counts."""

    grid_points = (
        grid.geometry
        .representative_point()
    )

    centers_xy = np.column_stack(
        [
            grid_points.x.to_numpy(
                dtype=float
            ),
            grid_points.y.to_numpy(
                dtype=float
            ),
        ]
    )

    poi_xy = np.column_stack(
        [
            pois.geometry.x.to_numpy(
                dtype=float
            ),
            pois.geometry.y.to_numpy(
                dtype=float
            ),
        ]
    )

    result = pd.DataFrame(
        {
            "grid_id": (
                grid[
                    "grid_id"
                ].astype(str)
            ),
        }
    )

    result[
        "poi_count_within_1000m"
    ] = count_points_within_radius(
        centers_xy,
        poi_xy,
        radius_m=1_000.0,
    )

    result[
        "poi_count_within_2000m"
    ] = count_points_within_radius(
        centers_xy,
        poi_xy,
        radius_m=2_000.0,
    )

    for (
        category_name,
        flag_column,
    ) in CATEGORY_COLUMNS.items():
        mask = (
            pois[
                flag_column
            ]
            .astype(
                bool
            )
            .to_numpy()
        )

        category_xy = poi_xy[
            mask
        ]

        result[
            f"{category_name}_within_1000m"
        ] = count_points_within_radius(
            centers_xy,
            category_xy,
            radius_m=1_000.0,
        )

    return result


def build_activity_features(
    grid: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach local and neighborhood OSM activity features to Ankara cells."""

    local = calculate_local_counts(
        grid,
        pois,
    )

    neighborhood = (
        calculate_neighborhood_counts(
            grid,
            pois,
        )
    )

    features = local.merge(
        neighborhood,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    result = grid[
        [
            "grid_id",
            "geometry",
        ]
    ].merge(
        features,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=grid.crs,
    )


def validate_activity_features(
    result: gpd.GeoDataFrame,
    *,
    expected_rows: int,
) -> None:
    """Validate generated activity features before writing outputs."""

    if len(
        result
    ) != expected_rows:
        raise ValueError(
            "Activity feature row count does not match the Ankara grid."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in activity features."
        )

    for column in OUTPUT_FEATURE_COLUMNS:
        if column not in result.columns:
            raise ValueError(
                f"Missing activity feature column: {column}"
            )

        values = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"{column} contains negative values."
            )

        if not np.allclose(
            values,
            np.rint(
                values
            ),
            atol=0,
            rtol=0,
        ):
            raise ValueError(
                f"{column} must contain integer counts."
            )

    local_total = result[
        "poi_count"
    ].to_numpy(
        dtype=float
    )

    within_1km = result[
        "poi_count_within_1000m"
    ].to_numpy(
        dtype=float
    )

    within_2km = result[
        "poi_count_within_2000m"
    ].to_numpy(
        dtype=float
    )

    if (
        within_1km
        < local_total
    ).any():
        raise ValueError(
            "1-km total POI count cannot be below local POI count."
        )

    if (
        within_2km
        < within_1km
    ).any():
        raise ValueError(
            "2-km total POI count cannot be below 1-km POI count."
        )

    for category_name in CATEGORY_COLUMNS:
        local = result[
            f"{category_name}_count"
        ].to_numpy(
            dtype=float
        )

        neighborhood = result[
            f"{category_name}_within_1000m"
        ].to_numpy(
            dtype=float
        )

        if (
            local
            > local_total
        ).any():
            raise ValueError(
                f"{category_name} local count exceeds total local POI count."
            )

        if (
            neighborhood
            > within_1km
        ).any():
            raise ValueError(
                f"{category_name} 1-km count exceeds total 1-km POI count."
            )

        if (
            neighborhood
            < local
        ).any():
            raise ValueError(
                f"{category_name} 1-km count is below its local count."
            )


def save_outputs(
    result: gpd.GeoDataFrame,
) -> None:
    """Save spatial and tabular Ankara activity features."""

    OUTPUT_GPKG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = (
        result.sort_values(
            "grid_id",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if OUTPUT_GPKG_PATH.exists():
        OUTPUT_GPKG_PATH.unlink()

    result.to_file(
        OUTPUT_GPKG_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    pd.DataFrame(
        result.drop(
            columns="geometry"
        )
    ).to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    result: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> None:
    """Document methodology, coverage, caveats, and feature distributions."""

    category_source_counts = {
        category_name: int(
            pois[
                flag_column
            ].sum()
        )
        for (
            category_name,
            flag_column,
        ) in CATEGORY_COLUMNS.items()
    }

    local_cells_with_poi = int(
        (
            result[
                "poi_count"
            ]
            > 0
        ).sum()
    )

    summary_lines = []

    for column in OUTPUT_FEATURE_COLUMNS:
        series = result[
            column
        ]

        summary_lines.append(
            f"- `{column}`: "
            f"mean {series.mean():,.2f}, "
            f"median {series.median():,.2f}, "
            f"p95 {series.quantile(0.95):,.2f}, "
            f"p99 {series.quantile(0.99):,.2f}, "
            f"max {series.max():,.0f}"
        )

    category_lines = [
        (
            f"- {category_name}: "
            f"{category_source_counts[category_name]:,}"
        )
        for category_name in CATEGORY_COLUMNS
    ]

    summary = f"""# Ankara OSM Activity Features

## Purpose

These features provide an OpenStreetMap-based urban-activity proxy for the
Ankara 500-m analysis grid.

They are not direct observations of EV demand, traffic volume, trips, or
economic activity.

## Source Inventory

- Unique buffered OSM activity POIs: {len(pois):,}
- Grid rows: {len(result):,}
- Grid cells with at least one local activity POI: {local_cells_with_poi:,}
- Grid cells with zero local activity POIs: {len(result) - local_cells_with_poi:,}

Category source counts:

{chr(10).join(category_lines)}

One OSM element may carry tags from more than one activity family. Total POI
counts deduplicate by OSM element identity; category counts are separate
descriptive flags and therefore are not required to sum to the total.

## Activity Taxonomy

### Retail / Commercial

The category includes:

- `shop=*` except explicit vacant/closed values
- `office=*` except explicit vacant/closed values
- selected commercial amenities such as marketplace, restaurants, cafes,
  fast-food/food-court venues, bars/pubs, banks and ATMs

### Education

Selected `amenity=*` values include school, college, university, kindergarten,
childcare, language school, music school and driving school.

### Healthcare

The category includes any non-empty `healthcare=*` plus selected healthcare
amenities: hospital, clinic, doctors, dentist and pharmacy.

### Transport Activity

The category includes:

- `public_transport=*`
- `highway=bus_stop`
- railway station/halt/tram-stop/subway-entrance features
- bus station, ferry terminal and taxi amenities
- aerodrome/terminal features

Charging stations and parking facilities are intentionally not part of this
activity taxonomy because VoltSight already models those feature families
separately.

## Spatial Method

Local counts assign every OSM POI to at most one Ankara 500-m grid cell.

The downloader keeps a 2.5-km buffer around Ankara. This allows 1-km and 2-km
neighborhood features for boundary cells to include nearby mapped activity just
outside the administrative boundary.

Neighborhood counts use exact Euclidean distance from each 500-m grid cell's
representative point to POI representative points in `{EXPECTED_CRS}`.

Ways and relations are represented by the center returned by Overpass rather
than their full footprint. The resulting counts should therefore be interpreted
as mapped activity-presence proxies, not precise floor-area or capacity
measures.

## Output Features

- `poi_count`
- `retail_commercial_count`
- `education_count`
- `healthcare_count`
- `transport_activity_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`
- `retail_commercial_within_1000m`
- `education_within_1000m`
- `healthcare_within_1000m`
- `transport_activity_within_1000m`

## Feature Distribution

{chr(10).join(summary_lines)}

## Interpretation Policy

OpenStreetMap completeness is spatially heterogeneous. A low POI count can mean
low mapped activity, incomplete mapping, or both.

These features must therefore be treated as urban-activity proxies rather than
ground-truth demand.

Before any canonical scoring change, the feature family should be audited for
coverage/redundancy and evaluated incrementally under the existing 5-km spatial
cross-validation design.

## Outputs

- `data/processed/{OUTPUT_GPKG_PATH.name}`
- `data/processed/{OUTPUT_CSV_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    result: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
) -> None:
    """Print the main Ankara activity-feature diagnostics."""

    print(
        "-"
        * 70
    )

    print(
        "Buffered unique OSM activity POIs:",
        f"{len(pois):,}",
    )

    print(
        "Grid rows:",
        f"{len(result):,}",
    )

    print(
        "Cells with local POIs:",
        f"{int((result['poi_count'] > 0).sum()):,}",
    )

    summary = result[
        list(
            OUTPUT_FEATURE_COLUMNS
        )
    ].describe().T[
        [
            "mean",
            "50%",
            "max",
        ]
    ]

    print()

    print(
        summary.to_string(
            float_format=lambda value: (
                f"{value:,.2f}"
            )
        )
    )


def main() -> None:
    """Create Ankara OSM urban-activity grid features."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara OSM Activity Features"
    )

    print(
        "="
        * 70
    )

    grid = load_grid()

    pois = load_pois()

    result = build_activity_features(
        grid,
        pois,
    )

    validate_activity_features(
        result,
        expected_rows=len(
            grid
        ),
    )

    save_outputs(
        result
    )

    create_summary(
        result,
        pois,
    )

    print_results(
        result,
        pois,
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara OSM activity features completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
