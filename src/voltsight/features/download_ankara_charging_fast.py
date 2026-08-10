from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from voltsight.features.create_charging_features import (  # noqa: E402
    prepare_charging_stations,
)


BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ankara_boundary_osm.geojson"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_charging_stations_osm.gpkg"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_charging_stations_osm_metadata.json"
)

RAW_RESPONSE_PATH = (
    PROJECT_ROOT
    / "cache"
    / "ankara"
    / "charging_fast"
    / "ankara_charging_overpass.json"
)

CANKAYA_REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations.gpkg"
)

OUTPUT_LAYER_NAME = "charging_stations"

TARGET_CRS = "EPSG:32636"

DOWNLOAD_BUFFER_METERS = 2_500
REQUEST_TIMEOUT_SECONDS = 600

OVERPASS_ENDPOINTS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def create_directories() -> None:
    """Create all output directories."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_RESPONSE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_download_footprint() -> tuple[
    Any,
    tuple[float, float, float, float],
]:
    """
    Load Ankara and create the exact 2.5-km analysis buffer.

    Returns the projected footprint and its WGS84 bounding box
    as south, west, north, east.
    """

    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            "Ankara boundary was not found:\n"
            f"{BOUNDARY_PATH}"
        )

    boundary = gpd.read_file(
        BOUNDARY_PATH
    )

    if boundary.empty:
        raise ValueError(
            "Ankara boundary is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "Ankara boundary has no CRS."
        )

    boundary_projected = boundary.to_crs(
        TARGET_CRS
    )

    footprint = (
        boundary_projected
        .geometry
        .union_all()
        .buffer(
            DOWNLOAD_BUFFER_METERS
        )
    )

    footprint_frame = gpd.GeoDataFrame(
        geometry=[footprint],
        crs=TARGET_CRS,
    ).to_crs(
        epsg=4326
    )

    min_x, min_y, max_x, max_y = (
        footprint_frame.total_bounds
    )

    bbox = (
        float(min_y),
        float(min_x),
        float(max_y),
        float(max_x),
    )

    print(
        "Buffered Ankara area: "
        f"{footprint.area / 1_000_000:,.2f} km2"
    )

    print(
        "WGS84 query bbox:"
    )

    print(
        "  south="
        f"{bbox[0]:.6f}, "
        "west="
        f"{bbox[1]:.6f}, "
        "north="
        f"{bbox[2]:.6f}, "
        "east="
        f"{bbox[3]:.6f}"
    )

    return footprint, bbox


def build_overpass_query(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> str:
    """Build one Ankara-wide charging-station query."""

    south, west, north, east = bbox

    return f"""
[out:json][timeout:{REQUEST_TIMEOUT_SECONDS}];
nwr
  ["amenity"="charging_station"]
  ({south:.7f},{west:.7f},{north:.7f},{east:.7f});
out tags center;
""".strip()


def download_overpass_json(
    query: str,
) -> tuple[
    dict[str, Any],
    str,
]:
    """Run one sparse Overpass query with endpoint fallback."""

    errors: list[str] = []

    headers = {
        "User-Agent": (
            "VoltSight/0.2 "
            "educational-geospatial-research-project"
        ),
        "Referer": (
            "https://github.com/ilginbor/voltsight-ai"
        ),
    }

    for endpoint in OVERPASS_ENDPOINTS:
        print("-" * 70)

        print(
            f"Trying endpoint: {endpoint}"
        )

        try:
            response = requests.post(
                endpoint,
                data={
                    "data": query,
                },
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            payload = response.json()

            elements = payload.get(
                "elements",
                [],
            )

            print(
                "Returned OSM elements: "
                f"{len(elements):,}"
            )

            RAW_RESPONSE_PATH.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                "Successful endpoint: "
                f"{endpoint}"
            )

            return payload, endpoint

        except Exception as error:
            message = (
                f"{endpoint}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            errors.append(
                message
            )

            print(
                f"Endpoint failed: {message}"
            )

    details = "\n".join(
        f"- {message}"
        for message in errors
    )

    raise RuntimeError(
        "All Overpass endpoints failed.\n"
        f"{details}"
    )


def element_coordinates(
    element: dict[str, Any],
) -> tuple[
    float,
    float,
] | None:
    """
    Return longitude/latitude for an OSM element.

    Nodes use their direct coordinate. Ways and relations use the
    center returned by Overpass.
    """

    element_type = str(
        element.get(
            "type",
            "",
        )
    )

    if element_type == "node":
        latitude = element.get(
            "lat"
        )

        longitude = element.get(
            "lon"
        )

    else:
        center = element.get(
            "center",
            {},
        )

        latitude = center.get(
            "lat"
        )

        longitude = center.get(
            "lon"
        )

    if (
        latitude is None
        or longitude is None
    ):
        return None

    return (
        float(longitude),
        float(latitude),
    )


def overpass_json_to_gdf(
    payload: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Convert the sparse Overpass response to an OSM-style GDF."""

    rows: list[
        dict[str, Any]
    ] = []

    for element in payload.get(
        "elements",
        [],
    ):
        coordinates = (
            element_coordinates(
                element
            )
        )

        if coordinates is None:
            continue

        tags = dict(
            element.get(
                "tags",
                {},
            )
        )

        row: dict[str, Any] = tags

        row["element"] = str(
            element.get(
                "type",
                "",
            )
        )

        row["id"] = str(
            element.get(
                "id",
                "",
            )
        )

        row["geometry"] = Point(
            coordinates
        )

        rows.append(
            row
        )

    if not rows:
        raise RuntimeError(
            "The Ankara-wide Overpass query "
            "returned no usable charging stations."
        )

    frame = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    if (
        frame["id"]
        .astype(str)
        .eq("")
        .any()
    ):
        raise ValueError(
            "An Overpass result has no OSM ID."
        )

    print(
        "Usable raw station records: "
        f"{len(frame):,}"
    )

    return frame


def filter_to_exact_footprint(
    stations: gpd.GeoDataFrame,
    footprint: Any,
) -> gpd.GeoDataFrame:
    """Remove bbox corner records outside Ankara's 2.5-km buffer."""

    if stations.crs != TARGET_CRS:
        stations = stations.to_crs(
            TARGET_CRS
        )

    result = stations.loc[
        stations.geometry.intersects(
            footprint
        )
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        "Stations inside exact Ankara buffer: "
        f"{len(result):,}"
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=TARGET_CRS,
    )


def validate_stations(
    stations: gpd.GeoDataFrame,
) -> None:
    """Validate the final Ankara OSM station inventory."""

    if stations.empty:
        raise ValueError(
            "Final station inventory is empty."
        )

    if stations[
        "station_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate station IDs were found."
        )

    if stations.geometry.isna().any():
        raise ValueError(
            "Missing station geometry was found."
        )

    if stations.geometry.is_empty.any():
        raise ValueError(
            "Empty station geometry was found."
        )

    if not stations.geometry.is_valid.all():
        raise ValueError(
            "Invalid station geometry was found."
        )

    print(
        "Ankara charging-station validation "
        "completed successfully."
    )


def compare_cankaya_reference(
    stations: gpd.GeoDataFrame,
) -> None:
    """
    Compare with the previous Cankaya OSM inventory when available.

    This is a diagnostic, not a hard validation because OSM can
    legitimately change over time.
    """

    if not CANKAYA_REFERENCE_PATH.exists():
        return

    reference = gpd.read_file(
        CANKAYA_REFERENCE_PATH,
        layer="charging_stations",
    )

    if (
        reference.empty
        or "station_id"
        not in reference.columns
    ):
        return

    reference_ids = set(
        reference[
            "station_id"
        ].astype(str)
    )

    current_ids = set(
        stations[
            "station_id"
        ].astype(str)
    )

    recovered = (
        reference_ids
        & current_ids
    )

    missing = (
        reference_ids
        - current_ids
    )

    print("-" * 70)

    print(
        "Cankaya reference OSM stations recovered: "
        f"{len(recovered):,}/{len(reference_ids):,}"
    )

    if missing:
        print(
            "Reference IDs not present in current OSM result:"
        )

        for station_id in sorted(
            missing
        ):
            print(
                f"  {station_id}"
            )


def save_outputs(
    stations: gpd.GeoDataFrame,
    endpoint: str,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> None:
    """Save station inventory and metadata."""

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    stations.to_file(
        OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    metadata = {
        "generated_at_utc": utc_now(),
        "method": (
            "single_overpass_bbox_query"
        ),
        "osm_tag": (
            "amenity=charging_station"
        ),
        "download_buffer_m": (
            DOWNLOAD_BUFFER_METERS
        ),
        "query_bbox_wgs84": {
            "south": bbox[0],
            "west": bbox[1],
            "north": bbox[2],
            "east": bbox[3],
        },
        "successful_endpoint": endpoint,
        "station_count": int(
            len(stations)
        ),
        "known_capacity_count": int(
            stations[
                "capacity_numeric"
            ].notna().sum()
        ),
        "ac_station_count": int(
            stations[
                "has_ac_connector"
            ].sum()
        ),
        "dc_station_count": int(
            stations[
                "has_dc_connector"
            ].sum()
        ),
        "output_path": (
            OUTPUT_PATH.as_posix()
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 70)

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


def main() -> None:
    """Run the Ankara-wide fast charging download."""

    print("=" * 70)

    print(
        "VoltSight - Fast Ankara Charging Download"
    )

    print("=" * 70)

    create_directories()

    footprint, bbox = (
        load_download_footprint()
    )

    query = build_overpass_query(
        bbox
    )

    print("-" * 70)

    print(
        "Running ONE Ankara-wide sparse "
        "charging-station query..."
    )

    payload, endpoint = (
        download_overpass_json(
            query
        )
    )

    raw_stations = (
        overpass_json_to_gdf(
            payload
        )
    )

    stations = prepare_charging_stations(
        raw_stations,
        TARGET_CRS,
    )

    stations = (
        filter_to_exact_footprint(
            stations,
            footprint,
        )
    )

    validate_stations(
        stations
    )

    compare_cankaya_reference(
        stations
    )

    save_outputs(
        stations,
        endpoint,
        bbox,
    )

    print("=" * 70)

    print(
        "Fast Ankara charging download completed."
    )

    print(
        "Unique OSM charging stations: "
        f"{len(stations):,}"
    )

    print(
        "Known-capacity stations: "
        f"{int(stations['capacity_numeric'].notna().sum()):,}"
    )

    print(
        "Stations with mapped AC: "
        f"{int(stations['has_ac_connector'].sum()):,}"
    )

    print(
        "Stations with mapped DC: "
        f"{int(stations['has_dc_connector'].sum()):,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
