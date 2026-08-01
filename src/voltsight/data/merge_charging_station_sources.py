from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OSM_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations.gpkg"
)

EPDK_STATION_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_stations.csv"
)

EPDK_COORDINATE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_verified_coordinates.csv"
)

MERGED_GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.gpkg"
)

MERGED_CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_station_source_merge_summary.md"
)

OSM_LAYER_NAME = "charging_stations"
MERGED_LAYER_NAME = "charging_stations_merged"

DUPLICATE_DISTANCE_THRESHOLD_METERS = 100.0

REQUIRED_EPDK_STATION_COLUMNS = {
    "station_no",
    "station_name",
    "service_type",
    "brand",
    "network_operator",
    "station_operator",
    "address",
    "socket_count",
    "ac_socket_count",
    "dc_socket_count",
    "total_socket_power_kw",
    "connector_types",
}

REQUIRED_COORDINATE_COLUMNS = {
    "station_no",
    "final_latitude",
    "final_longitude",
    "review_status",
    "coordinate_confidence",
    "coordinate_is_official_epdk",
    "coordinate_source",
}


def clean_text(value: Any) -> str:
    """Convert an arbitrary value into normalized text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return " ".join(
        str(value).split()
    )


def extract_station_id(value: Any) -> str:
    """Extract the numeric part of an EPDK station number."""

    text = clean_text(value)

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    return digits


def parse_integer(
    value: Any,
    default: int = 0,
) -> int:
    """Convert a value to a non-negative integer."""

    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return default

    return max(
        int(round(float(number))),
        0,
    )


def parse_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""

    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return default

    number = float(number)

    if not np.isfinite(number):
        return default

    return number


def validate_input_paths() -> None:
    """Ensure that all required input files exist."""

    required_paths = (
        OSM_INPUT_PATH,
        EPDK_STATION_INPUT_PATH,
        EPDK_COORDINATE_INPUT_PATH,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"- {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required input files are missing:\n"
            f"{formatted}"
        )


def load_osm_stations() -> gpd.GeoDataFrame:
    """Load the existing OpenStreetMap charging inventory."""

    osm = gpd.read_file(
        OSM_INPUT_PATH,
        layer=OSM_LAYER_NAME,
    )

    if osm.empty:
        raise ValueError(
            "The OSM charging-station dataset is empty."
        )

    if osm.crs is None:
        raise ValueError(
            "The OSM charging-station dataset has no CRS."
        )

    if "station_id" not in osm.columns:
        raise ValueError(
            "The OSM dataset does not contain station_id."
        )

    if osm["station_id"].duplicated().any():
        raise ValueError(
            "Duplicate OSM station IDs were found."
        )

    if osm.geometry.isna().any():
        raise ValueError(
            "The OSM dataset contains missing geometries."
        )

    print(
        "Loaded OSM station count:",
        len(osm),
    )

    print(
        "OSM CRS:",
        osm.crs,
    )

    return osm


def load_verified_epdk_stations() -> pd.DataFrame:
    """Join EPDK station metadata with accepted coordinates."""

    stations = pd.read_csv(
        EPDK_STATION_INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")

    coordinates = pd.read_csv(
        EPDK_COORDINATE_INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")

    missing_station_columns = (
        REQUIRED_EPDK_STATION_COLUMNS
        - set(stations.columns)
    )

    if missing_station_columns:
        raise ValueError(
            "Missing EPDK station columns: "
            f"{sorted(missing_station_columns)}"
        )

    missing_coordinate_columns = (
        REQUIRED_COORDINATE_COLUMNS
        - set(coordinates.columns)
    )

    if missing_coordinate_columns:
        raise ValueError(
            "Missing verified-coordinate columns: "
            f"{sorted(missing_coordinate_columns)}"
        )

    stations["_station_id"] = (
        stations["station_no"]
        .apply(extract_station_id)
    )

    coordinates["_station_id"] = (
        coordinates["station_no"]
        .apply(extract_station_id)
    )

    coordinates = coordinates[
        coordinates["review_status"]
        .str.startswith("accepted_")
    ].copy()

    coordinates[
        "final_latitude"
    ] = pd.to_numeric(
        coordinates["final_latitude"],
        errors="coerce",
    )

    coordinates[
        "final_longitude"
    ] = pd.to_numeric(
        coordinates["final_longitude"],
        errors="coerce",
    )

    coordinates.dropna(
        subset=[
            "final_latitude",
            "final_longitude",
        ],
        inplace=True,
    )

    coordinate_columns = [
        "_station_id",
        "final_latitude",
        "final_longitude",
        "review_status",
        "coordinate_confidence",
        "coordinate_is_official_epdk",
        "coordinate_source",
        "review_notes",
    ]

    available_coordinate_columns = [
        column
        for column in coordinate_columns
        if column in coordinates.columns
    ]

    verified = stations.merge(
        coordinates[
            available_coordinate_columns
        ],
        on="_station_id",
        how="inner",
        validate="one_to_one",
    )

    if verified.empty:
        raise ValueError(
            "No accepted EPDK coordinate was found."
        )

    if not verified[
        "final_latitude"
    ].between(-90, 90).all():
        raise ValueError(
            "Invalid EPDK latitude values were found."
        )

    if not verified[
        "final_longitude"
    ].between(-180, 180).all():
        raise ValueError(
            "Invalid EPDK longitude values were found."
        )

    print(
        "Loaded verified EPDK station count:",
        len(verified),
    )

    return verified


def add_osm_provenance(
    osm: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add common source fields to OSM records."""

    result = osm.copy()

    result["data_source"] = "OSM"
    result["source_osm"] = 1
    result["source_epdk"] = 0
    result["epdk_station_no"] = ""
    result["epdk_service_type"] = ""
    result["epdk_address"] = ""
    result["epdk_socket_count"] = 0
    result["epdk_ac_socket_count"] = 0
    result["epdk_dc_socket_count"] = 0
    result["epdk_total_socket_power_kw"] = 0.0
    result["coordinate_confidence"] = ""
    result["coordinate_is_official_epdk"] = 0
    result["coordinate_source"] = ""
    result["nearest_osm_distance_m"] = np.nan

    return result


def build_epdk_geodataframe(
    verified: pd.DataFrame,
    osm: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create EPDK spatial records and detect nearby OSM stations."""

    epdk_wgs84 = gpd.GeoDataFrame(
        verified.copy(),
        geometry=gpd.points_from_xy(
            verified["final_longitude"],
            verified["final_latitude"],
        ),
        crs="EPSG:4326",
    )

    epdk_projected = epdk_wgs84.to_crs(
        osm.crs
    )

    epdk_records: list[
        dict[str, Any]
    ] = []

    comparison_records: list[
        dict[str, Any]
    ] = []

    for _, row in epdk_projected.iterrows():
        distances = osm.geometry.distance(
            row.geometry
        )

        nearest_index = distances.idxmin()

        nearest_distance = float(
            distances.loc[nearest_index]
        )

        is_duplicate = (
            nearest_distance
            <= DUPLICATE_DISTANCE_THRESHOLD_METERS
        )

        comparison_records.append(
            {
                "epdk_station_no": clean_text(
                    row["station_no"]
                ),
                "epdk_station_name": clean_text(
                    row["station_name"]
                ),
                "nearest_osm_station_id": clean_text(
                    osm.loc[
                        nearest_index,
                        "station_id",
                    ]
                ),
                "nearest_osm_name": clean_text(
                    osm.loc[
                        nearest_index,
                        "name",
                    ]
                    if "name" in osm.columns
                    else ""
                ),
                "nearest_osm_distance_m": round(
                    nearest_distance,
                    2,
                ),
                "treated_as_duplicate": int(
                    is_duplicate
                ),
            }
        )

        if is_duplicate:
            continue

        socket_count = parse_integer(
            row["socket_count"]
        )

        ac_socket_count = parse_integer(
            row["ac_socket_count"]
        )

        dc_socket_count = parse_integer(
            row["dc_socket_count"]
        )

        connector_types = clean_text(
            row["connector_types"]
        )

        connector_type_count = len(
            [
                connector_type
                for connector_type in (
                    connector_types.split(";")
                )
                if connector_type.strip()
            ]
        )

        record = {
            column: None
            for column in osm.columns
            if column != "geometry"
        }

        numeric_station_id = (
            extract_station_id(
                row["station_no"]
            )
        )

        record.update(
            {
                "station_id": (
                    f"epdk_srj_{numeric_station_id}"
                ),
                "osm_element_type": "epdk",
                "osm_id": "",
                "name": clean_text(
                    row["station_name"]
                ),
                "operator": clean_text(
                    row["station_operator"]
                ),
                "brand": clean_text(
                    row["brand"]
                ),
                "network": clean_text(
                    row["network_operator"]
                ),
                "access": (
                    "yes"
                    if "HALKA_ACIK" in clean_text(
                        row["service_type"]
                    ).upper()
                    else ""
                ),
                "capacity": str(
                    socket_count
                ),
                "capacity_numeric": (
                    socket_count
                ),
                "connector_types": (
                    connector_types
                ),
                "mapped_socket_type_count": (
                    connector_type_count
                ),
                "known_socket_count": (
                    socket_count
                ),
                "has_ac_connector": int(
                    ac_socket_count > 0
                ),
                "has_dc_connector": int(
                    dc_socket_count > 0
                ),
                "geometry_type": "Point",
                "data_source": "EPDK",
                "source_osm": 0,
                "source_epdk": 1,
                "epdk_station_no": clean_text(
                    row["station_no"]
                ),
                "epdk_service_type": clean_text(
                    row["service_type"]
                ),
                "epdk_address": clean_text(
                    row["address"]
                ),
                "epdk_socket_count": (
                    socket_count
                ),
                "epdk_ac_socket_count": (
                    ac_socket_count
                ),
                "epdk_dc_socket_count": (
                    dc_socket_count
                ),
                "epdk_total_socket_power_kw": (
                    parse_float(
                        row[
                            "total_socket_power_kw"
                        ]
                    )
                ),
                "coordinate_confidence": clean_text(
                    row[
                        "coordinate_confidence"
                    ]
                ),
                "coordinate_is_official_epdk": (
                    parse_integer(
                        row[
                            "coordinate_is_official_epdk"
                        ]
                    )
                ),
                "coordinate_source": clean_text(
                    row["coordinate_source"]
                ),
                "nearest_osm_distance_m": round(
                    nearest_distance,
                    2,
                ),
                "geometry": row.geometry,
            }
        )

        epdk_records.append(
            record
        )

    epdk_columns = list(
        add_osm_provenance(
            osm.head(0)
        ).columns
    )

    epdk_gdf = gpd.GeoDataFrame(
        epdk_records,
        columns=epdk_columns,
        geometry="geometry",
        crs=osm.crs,
    )

    comparisons = pd.DataFrame(
        comparison_records
    )

    return epdk_gdf, comparisons


def create_merged_dataset(
    osm: gpd.GeoDataFrame,
    epdk: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Combine OSM and non-duplicate EPDK station records."""

    osm_with_source = add_osm_provenance(
        osm
    )

    merged = gpd.GeoDataFrame(
        pd.concat(
            [
                osm_with_source,
                epdk,
            ],
            ignore_index=True,
            sort=False,
        ),
        geometry="geometry",
        crs=osm.crs,
    )

    if merged["station_id"].duplicated().any():
        duplicates = merged.loc[
            merged["station_id"].duplicated(
                keep=False
            ),
            "station_id",
        ].tolist()

        raise ValueError(
            "Duplicate merged station IDs found: "
            f"{duplicates}"
        )

    if merged.geometry.isna().any():
        raise ValueError(
            "The merged dataset contains missing geometries."
        )

    return merged


def save_outputs(
    merged: gpd.GeoDataFrame,
    comparisons: pd.DataFrame,
    osm_count: int,
    verified_epdk_count: int,
) -> None:
    """Save merged spatial data and a reproducibility summary."""

    MERGED_GPKG_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MERGED_GPKG_OUTPUT_PATH.exists():
        MERGED_GPKG_OUTPUT_PATH.unlink()

    merged.to_file(
        MERGED_GPKG_OUTPUT_PATH,
        layer=MERGED_LAYER_NAME,
        driver="GPKG",
    )

    csv_output = merged.drop(
        columns="geometry"
    ).copy()

    csv_output["geometry_wkt"] = (
        merged.geometry.to_wkt()
    )

    csv_output.to_csv(
        MERGED_CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_count = int(
        comparisons[
            "treated_as_duplicate"
        ].sum()
    )

    epdk_added_count = (
        verified_epdk_count
        - duplicate_count
    )

    comparison_lines = "\n".join(
        (
            f"- `{row.epdk_station_no}` "
            f"{row.epdk_station_name}: "
            f"nearest OSM distance "
            f"{row.nearest_osm_distance_m:,.2f} m; "
            f"duplicate={bool(row.treated_as_duplicate)}"
        )
        for row in comparisons.itertuples()
    )

    summary = f"""# Çankaya Charging-Station Source Merge

## Sources

- OpenStreetMap station records: {osm_count:,}
- Verified EPDK station records: {verified_epdk_count:,}
- Duplicate-distance threshold: {DUPLICATE_DISTANCE_THRESHOLD_METERS:,.0f} m
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Result

- EPDK records treated as existing OSM stations: {duplicate_count:,}
- New EPDK-only stations added: {epdk_added_count:,}
- Final merged station count: {len(merged):,}

## EPDK and OSM Distance Review

{comparison_lines}

## Generated Files

- `data/interim/cankaya_charging_stations_merged.gpkg`
- `data/interim/cankaya_charging_stations_merged.csv`

## Provenance

Every merged record contains `data_source`, `source_osm` and
`source_epdk` fields.

The EPDK coordinate currently used is not an official coordinate
published in the downloaded EPDK report. Its confidence and source
are retained in the merged dataset.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("Merged GeoPackage saved:")
    print(MERGED_GPKG_OUTPUT_PATH)

    print()
    print("Merged CSV saved:")
    print(MERGED_CSV_OUTPUT_PATH)

    print()
    print("Merge summary saved:")
    print(SUMMARY_OUTPUT_PATH)


def main() -> None:
    """Merge OSM and verified EPDK charging stations."""

    print("=" * 70)
    print("VoltSight - Charging Station Source Merge")
    print("=" * 70)

    validate_input_paths()

    osm = load_osm_stations()

    verified_epdk = (
        load_verified_epdk_stations()
    )

    epdk_gdf, comparisons = (
        build_epdk_geodataframe(
            verified_epdk,
            osm,
        )
    )

    merged = create_merged_dataset(
        osm,
        epdk_gdf,
    )

    expected_count = (
        len(osm)
        + len(epdk_gdf)
    )

    if len(merged) != expected_count:
        raise ValueError(
            "Unexpected merged station count."
        )

    save_outputs(
        merged,
        comparisons,
        osm_count=len(osm),
        verified_epdk_count=len(
            verified_epdk
        ),
    )

    print()
    print("OSM station count:", len(osm))

    print(
        "Verified EPDK station count:",
        len(verified_epdk),
    )

    print(
        "EPDK stations added:",
        len(epdk_gdf),
    )

    print(
        "Final merged station count:",
        len(merged),
    )

    print()
    print("Nearest-source comparison:")

    print(
        comparisons.to_string(
            index=False
        )
    )

    print("=" * 70)
    print(
        "Charging station source merge "
        "completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
