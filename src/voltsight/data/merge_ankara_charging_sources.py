from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from voltsight.data.merge_charging_station_sources import (  # noqa: E402
    DUPLICATE_DISTANCE_THRESHOLD_METERS,
    build_epdk_geodataframe,
    create_merged_dataset,
    load_verified_epdk_stations,
)


OSM_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_charging_stations_osm.gpkg"
)

OUTPUT_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_charging_stations_merged.gpkg"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_charging_stations_merged.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_charging_station_merge_summary.md"
)

OSM_LAYER_NAME = "charging_stations"
MERGED_LAYER_NAME = "charging_stations_merged"


def load_ankara_osm() -> gpd.GeoDataFrame:
    """Load the fast Ankara-wide OSM charging inventory."""

    if not OSM_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Ankara OSM inventory: {OSM_INPUT_PATH}"
        )

    osm = gpd.read_file(
        OSM_INPUT_PATH,
        layer=OSM_LAYER_NAME,
    )

    if osm.empty:
        raise ValueError(
            "Ankara OSM charging inventory is empty."
        )

    if osm.crs is None:
        raise ValueError(
            "Ankara OSM charging inventory has no CRS."
        )

    if "station_id" not in osm.columns:
        raise ValueError(
            "Ankara OSM inventory has no station_id."
        )

    if osm["station_id"].duplicated().any():
        raise ValueError(
            "Duplicate Ankara OSM station IDs were found."
        )

    print(
        "Loaded Ankara OSM stations:",
        len(osm),
    )

    return osm


def save_outputs(
    merged: gpd.GeoDataFrame,
    comparison: pd.DataFrame,
    osm_count: int,
    epdk_count: int,
) -> None:
    """Save the Ankara analysis inventory and provenance summary."""

    OUTPUT_GPKG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        OUTPUT_GPKG_PATH,
        OUTPUT_CSV_PATH,
    ):
        if path.exists():
            path.unlink()

    merged.to_file(
        OUTPUT_GPKG_PATH,
        layer=MERGED_LAYER_NAME,
        driver="GPKG",
    )

    pd.DataFrame(
        merged.drop(columns="geometry")
    ).to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    duplicate_count = int(
        comparison[
            "treated_as_duplicate"
        ].sum()
    )

    epdk_added = (
        epdk_count
        - duplicate_count
    )

    source_counts = (
        merged["data_source"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    source_lines = "\n".join(
        f"- {source}: {int(count):,}"
        for source, count in source_counts.items()
    )

    comparison_lines = "\n".join(
        (
            f"- {row.epdk_station_no}: "
            f"nearest OSM "
            f"{row.nearest_osm_distance_m:,.2f} m; "
            f"duplicate={bool(row.treated_as_duplicate)}"
        )
        for row in comparison.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Charging Station Merge Summary

## Analysis Inventory

- Ankara-wide OSM charging stations: {osm_count:,}
- Supplemental verified-coordinate EPDK records: {epdk_count:,}
- EPDK records matched to existing OSM stations: {duplicate_count:,}
- EPDK-only records added: {epdk_added:,}
- Final analysis station count: {len(merged):,}
- Duplicate-distance threshold: {DUPLICATE_DISTANCE_THRESHOLD_METERS:,.0f} m

## Source Counts

{source_lines}

## EPDK / OSM Comparison

{comparison_lines}

## Important Scope Note

The OpenStreetMap inventory covers the Ankara study area.

The EPDK component is only the previously reviewed supplemental
coordinate dataset used in the Çankaya pilot. It must not be
interpreted as a complete province-wide spatial EPDK inventory.

The accepted EPDK coordinate is retained with its provenance and
coordinate-confidence fields.

## Outputs

- `data/interim/ankara_charging_stations_merged.gpkg`
- `data/interim/ankara_charging_stations_merged.csv`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def main() -> None:
    """Build the Ankara charging analysis inventory."""

    print("=" * 70)
    print("VoltSight - Ankara Charging Source Merge")
    print("=" * 70)

    osm = load_ankara_osm()

    verified_epdk = (
        load_verified_epdk_stations()
    )

    epdk_gdf, comparison = (
        build_epdk_geodataframe(
            verified_epdk,
            osm,
        )
    )

    merged = create_merged_dataset(
        osm,
        epdk_gdf,
    )

    if merged["station_id"].duplicated().any():
        raise ValueError(
            "Duplicate station IDs remained after merge."
        )

    if merged.geometry.isna().any():
        raise ValueError(
            "Merged inventory contains missing geometry."
        )

    save_outputs(
        merged=merged,
        comparison=comparison,
        osm_count=len(osm),
        epdk_count=len(verified_epdk),
    )

    print("-" * 70)
    print("OSM station count:", len(osm))
    print(
        "Supplemental EPDK count:",
        len(verified_epdk),
    )
    print(
        "EPDK duplicates:",
        int(
            comparison[
                "treated_as_duplicate"
            ].sum()
        ),
    )
    print(
        "Final analysis station count:",
        len(merged),
    )
    print(
        "AC stations:",
        int(
            merged[
                "has_ac_connector"
            ].sum()
        ),
    )
    print(
        "DC stations:",
        int(
            merged[
                "has_dc_connector"
            ].sum()
        ),
    )
    print("=" * 70)
    print(
        "Ankara charging source merge completed successfully."
    )


if __name__ == "__main__":
    main()
