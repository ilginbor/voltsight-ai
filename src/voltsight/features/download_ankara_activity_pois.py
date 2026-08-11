from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "ankara_activity_pois_osm.gpkg"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "ankara_activity_pois_osm_metadata.json"
)

RAW_RESPONSE_DIRECTORY = (
    PROJECT_ROOT
    / "cache"
    / "ankara"
    / "activity_pois"
)

OUTPUT_LAYER_NAME = "activity_pois"

TARGET_CRS = "EPSG:32636"

# The feature pipeline evaluates neighborhoods out to 2 km.
# A 2.5-km download buffer reduces edge truncation for cells near Ankara's
# administrative boundary.
DOWNLOAD_BUFFER_METERS = 2_500.0

REQUEST_TIMEOUT_SECONDS = 900

# Large province-wide tag families can time out on public Overpass instances.
# Failed query groups are therefore subdivided recursively into 2x2 bbox tiles.
# Depth 5 means a worst-case leaf is 1/1024 of the original bbox area.
MAX_QUERY_SPLIT_DEPTH = 5

# Be gentle with public Overpass services. Successful cached tiles make reruns
# cheap, while a small delay reduces 429/rate-limit failures during long runs.
REQUEST_DELAY_SECONDS = 2.0
TRANSIENT_RETRY_DELAYS_SECONDS = (
    5.0,
    15.0,
    30.0,
)

# These families are broad enough that Ankara-wide queries commonly time out.
# Start them tiled immediately instead of spending one full-province request.
FORCE_TILED_QUERY_NAMES = {
    "commercial_office",
    "education",
    "transport_activity",
}

# The mail.ru endpoint was the most reliable for this Ankara workload in prior
# runs, so try it first. The order is operational only and does not affect the
# data semantics.
OVERPASS_ENDPOINTS = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

RETAIL_COMMERCIAL_AMENITIES = {
    "marketplace",
    "restaurant",
    "cafe",
    "fast_food",
    "food_court",
    "bar",
    "pub",
    "biergarten",
    "bank",
    "atm",
}

EDUCATION_AMENITIES = {
    "school",
    "college",
    "university",
    "kindergarten",
    "childcare",
    "language_school",
    "music_school",
    "driving_school",
}

HEALTHCARE_AMENITIES = {
    "hospital",
    "clinic",
    "doctors",
    "dentist",
    "pharmacy",
}

TRANSPORT_AMENITIES = {
    "bus_station",
    "ferry_terminal",
    "taxi",
}

TRANSPORT_RAILWAY_VALUES = {
    "station",
    "halt",
    "tram_stop",
    "subway_entrance",
}

TRANSPORT_AEROWAY_VALUES = {
    "terminal",
    "aerodrome",
}

# Large tag families are separated into several sparse queries so one very
# large Overpass response does not make the full Ankara download all-or-nothing.
QUERY_SPECS: dict[str, tuple[str, ...]] = {
    "retail_shop": (
        '["shop"]',
    ),
    "commercial_office": (
        '["office"]',
    ),
    "retail_food_finance": (
        '["amenity"~"^(marketplace|restaurant|cafe|fast_food|food_court|bar|pub|biergarten|bank|atm)$"]',
    ),
    "education": (
        '["amenity"~"^(school|college|university|kindergarten|childcare|language_school|music_school|driving_school)$"]',
    ),
    "healthcare": (
        '["healthcare"]',
        '["amenity"~"^(hospital|clinic|doctors|dentist|pharmacy)$"]',
    ),
    "transport_activity": (
        '["public_transport"]',
        '["highway"="bus_stop"]',
        '["railway"~"^(station|halt|tram_stop|subway_entrance)$"]',
        '["amenity"~"^(bus_station|ferry_terminal|taxi)$"]',
        '["aeroway"~"^(terminal|aerodrome)$"]',
    ),
}

SELECTED_TAG_COLUMNS = (
    "name",
    "amenity",
    "shop",
    "office",
    "healthcare",
    "public_transport",
    "highway",
    "railway",
    "aeroway",
)


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def create_directories() -> None:
    """Create output and cache directories."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_RESPONSE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_download_footprint() -> tuple[
    Any,
    tuple[
        float,
        float,
        float,
        float,
    ],
]:
    """
    Load Ankara and create a 2.5-km projected download footprint.

    Returns the projected footprint plus a WGS84 bbox ordered as
    south, west, north, east.
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

    boundary_projected = (
        boundary.to_crs(
            TARGET_CRS
        )
    )

    footprint = (
        boundary_projected.geometry
        .union_all()
        .buffer(
            DOWNLOAD_BUFFER_METERS
        )
    )

    footprint_wgs84 = gpd.GeoDataFrame(
        geometry=[
            footprint,
        ],
        crs=TARGET_CRS,
    ).to_crs(
        epsg=4326
    )

    min_x, min_y, max_x, max_y = (
        footprint_wgs84.total_bounds
    )

    bbox = (
        float(
            min_y
        ),
        float(
            min_x
        ),
        float(
            max_y
        ),
        float(
            max_x
        ),
    )

    print(
        "Buffered Ankara activity area: "
        f"{footprint.area / 1_000_000:,.2f} km2"
    )

    print(
        "WGS84 query bbox: "
        f"south={bbox[0]:.6f}, "
        f"west={bbox[1]:.6f}, "
        f"north={bbox[2]:.6f}, "
        f"east={bbox[3]:.6f}"
    )

    return (
        footprint,
        bbox,
    )


def build_overpass_query(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    selectors: tuple[
        str,
        ...,
    ],
) -> str:
    """Build one sparse Overpass query from a set of selectors."""

    if not selectors:
        raise ValueError(
            "At least one Overpass selector is required."
        )

    south, west, north, east = bbox

    clauses = "\n".join(
        (
            f"  nwr{selector}"
            f"({south:.7f},{west:.7f},"
            f"{north:.7f},{east:.7f});"
        )
        for selector in selectors
    )

    return (
        f"[out:json][timeout:{REQUEST_TIMEOUT_SECONDS}];\n"
        "(\n"
        f"{clauses}\n"
        ");\n"
        "out tags center;"
    )


def cache_path_for_query(
    cache_key: str,
) -> Path:
    """Return the deterministic raw-response cache path."""

    safe_key = (
        str(
            cache_key
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )

    return (
        RAW_RESPONSE_DIRECTORY
        / f"{safe_key}.json"
    )


def load_cached_payload(
    cache_key: str,
) -> dict[
    str,
    Any,
] | None:
    """Load one cached Overpass payload when present."""

    cache_path = cache_path_for_query(
        cache_key
    )

    if not cache_path.exists():
        return None

    payload = json.loads(
        cache_path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Cached payload is not a JSON object: {cache_path}"
        )

    print(
        f"Using cached query: {cache_key}"
    )

    return payload


def save_cached_payload(
    cache_key: str,
    payload: dict[
        str,
        Any,
    ],
) -> None:
    """Persist one Overpass payload under a deterministic cache key."""

    cache_path = cache_path_for_query(
        cache_key
    )

    cache_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _retry_after_seconds(
    response: requests.Response,
) -> float | None:
    """Parse a numeric Retry-After header when one is supplied."""

    raw_value = response.headers.get(
        "Retry-After"
    )

    if raw_value is None:
        return None

    try:
        value = float(
            raw_value
        )
    except ValueError:
        return None

    return max(
        0.0,
        value,
    )


def _is_transient_http_status(
    status_code: int,
) -> bool:
    """Return whether an HTTP status should receive a delayed retry."""

    return status_code in {
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    }


def request_overpass_json(
    display_name: str,
    query: str,
    *,
    cache_key: str,
    refresh: bool,
) -> tuple[
    dict[
        str,
        Any,
    ],
    str,
]:
    """
    Run one concrete Overpass request with endpoint fallback and backoff.

    Public Overpass instances may return temporary 429/5xx responses. Those
    statuses are retried with conservative delays before moving to the next
    endpoint. Connection-level failures move on immediately.
    """

    if not refresh:
        cached = load_cached_payload(
            cache_key
        )

        if cached is not None:
            return (
                cached,
                "cache",
            )

    errors: list[
        str
    ] = []

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
        attempts = (
            1
            + len(
                TRANSIENT_RETRY_DELAYS_SECONDS
            )
        )

        for attempt_index in range(
            attempts
        ):
            print(
                "-"
                * 70
            )

            print(
                f"Query {display_name}: trying {endpoint} "
                f"(attempt {attempt_index + 1}/{attempts})"
            )

            if REQUEST_DELAY_SECONDS > 0:
                time.sleep(
                    REQUEST_DELAY_SECONDS
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

                if not response.ok:
                    status_code = int(
                        response.status_code
                    )

                    message = (
                        f"{endpoint}: HTTP {status_code} "
                        f"{response.reason}"
                    )

                    if (
                        _is_transient_http_status(
                            status_code
                        )
                        and attempt_index
                        < (
                            attempts
                            - 1
                        )
                    ):
                        retry_after = (
                            _retry_after_seconds(
                                response
                            )
                        )

                        fallback_delay = (
                            TRANSIENT_RETRY_DELAYS_SECONDS[
                                attempt_index
                            ]
                        )

                        delay = (
                            retry_after
                            if retry_after
                            is not None
                            else fallback_delay
                        )

                        print(
                            f"Transient endpoint response: {message}. "
                            f"Waiting {delay:.1f}s before retry."
                        )

                        time.sleep(
                            delay
                        )

                        continue

                    response.raise_for_status()

                payload = response.json()

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise ValueError(
                        "Overpass response is not a JSON object."
                    )

                elements = payload.get(
                    "elements",
                    [],
                )

                if not isinstance(
                    elements,
                    list,
                ):
                    raise ValueError(
                        "Overpass elements field is not a list."
                    )

                save_cached_payload(
                    cache_key,
                    payload,
                )

                print(
                    f"Query {display_name}: "
                    f"{len(elements):,} elements"
                )

                return (
                    payload,
                    endpoint,
                )

            except requests.HTTPError as error:
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

                break

            except requests.RequestException as error:
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

                break

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

                break

    details = "\n".join(
        f"- {message}"
        for message in errors
    )

    raise RuntimeError(
        f"All Overpass endpoints failed for {display_name}.\n"
        f"{details}"
    )


def split_bbox_into_quadrants(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    tuple[
        float,
        float,
        float,
        float,
    ],
    ...,
]:
    """Split a south/west/north/east bbox into four deterministic quadrants."""

    south, west, north, east = bbox

    if not (
        south < north
        and west < east
    ):
        raise ValueError(
            f"Invalid bbox: {bbox}"
        )

    middle_latitude = (
        south
        + north
    ) / 2.0

    middle_longitude = (
        west
        + east
    ) / 2.0

    return (
        (
            south,
            west,
            middle_latitude,
            middle_longitude,
        ),
        (
            south,
            middle_longitude,
            middle_latitude,
            east,
        ),
        (
            middle_latitude,
            west,
            north,
            middle_longitude,
        ),
        (
            middle_latitude,
            middle_longitude,
            north,
            east,
        ),
    )


def merge_overpass_payloads(
    payloads: list[
        dict[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    Any,
]:
    """Merge tiled Overpass payloads and deduplicate elements by type and ID."""

    merged_elements: dict[
        tuple[
            str,
            str,
        ],
        dict[
            str,
            Any,
        ],
    ] = {}

    for payload in payloads:
        elements = payload.get(
            "elements",
            [],
        )

        if not isinstance(
            elements,
            list,
        ):
            raise ValueError(
                "Overpass elements field is not a list."
            )

        for element in elements:
            if not isinstance(
                element,
                dict,
            ):
                continue

            element_type = str(
                element.get(
                    "type",
                    "",
                )
            )

            element_id = str(
                element.get(
                    "id",
                    "",
                )
            )

            if not element_type or not element_id:
                continue

            key = (
                element_type,
                element_id,
            )

            if key not in merged_elements:
                merged_elements[
                    key
                ] = element

    return {
        "elements": [
            merged_elements[
                key
            ]
            for key in sorted(
                merged_elements
            )
        ],
    }


def download_tiled_query(
    query_name: str,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    selectors: tuple[
        str,
        ...,
    ],
    *,
    refresh: bool,
    depth: int,
    tile_key: str,
) -> tuple[
    dict[
        str,
        Any,
    ],
    list[
        str
    ],
]:
    """
    Download one bbox, recursively subdividing it after endpoint-wide failure.

    Successful leaf-tile responses are cached independently. A rerun therefore
    resumes from completed tiles rather than repeating the entire group.
    """

    cache_key = (
        f"{query_name}__{tile_key}"
    )

    query = build_overpass_query(
        bbox,
        selectors,
    )

    try:
        payload, endpoint = (
            request_overpass_json(
                (
                    f"{query_name} "
                    f"[{tile_key}]"
                ),
                query,
                cache_key=cache_key,
                refresh=refresh,
            )
        )

        return (
            payload,
            [
                endpoint,
            ],
        )

    except RuntimeError:
        if depth >= MAX_QUERY_SPLIT_DEPTH:
            raise

        print(
            "-"
            * 70
        )

        print(
            f"Query {query_name} [{tile_key}] failed on all endpoints; "
            f"splitting to depth {depth + 1}."
        )

        child_payloads: list[
            dict[
                str,
                Any,
            ]
        ] = []

        child_endpoints: list[
            str
        ] = []

        for child_index, child_bbox in enumerate(
            split_bbox_into_quadrants(
                bbox
            ),
            start=1,
        ):
            (
                child_payload,
                endpoints,
            ) = download_tiled_query(
                query_name,
                child_bbox,
                selectors,
                refresh=refresh,
                depth=depth + 1,
                tile_key=(
                    f"{tile_key}_{child_index}"
                ),
            )

            child_payloads.append(
                child_payload
            )

            child_endpoints.extend(
                endpoints
            )

        return (
            merge_overpass_payloads(
                child_payloads
            ),
            child_endpoints,
        )


def download_overpass_json(
    query_name: str,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    selectors: tuple[
        str,
        ...,
    ],
    *,
    refresh: bool,
) -> tuple[
    dict[
        str,
        Any,
    ],
    str,
]:
    """
    Download one logical query group with resumable adaptive bbox tiling.

    A completed group is stored as `<query_name>.json`. Large/failing groups
    are assembled from cached leaf tiles and then persisted under that same
    group-level cache key.
    """

    if not refresh:
        cached = load_cached_payload(
            query_name
        )

        if cached is not None:
            return (
                cached,
                "cache",
            )

    if query_name in FORCE_TILED_QUERY_NAMES:
        print(
            "-"
            * 70
        )

        print(
            f"Query {query_name}: starting with 2x2 tiles "
            "because this tag family is broad."
        )

        payloads: list[
            dict[
                str,
                Any,
            ]
        ] = []

        endpoints: list[
            str
        ] = []

        for child_index, child_bbox in enumerate(
            split_bbox_into_quadrants(
                bbox
            ),
            start=1,
        ):
            (
                child_payload,
                child_endpoints,
            ) = download_tiled_query(
                query_name,
                child_bbox,
                selectors,
                refresh=refresh,
                depth=1,
                tile_key=(
                    f"tile_{child_index}"
                ),
            )

            payloads.append(
                child_payload
            )

            endpoints.extend(
                child_endpoints
            )

        payload = merge_overpass_payloads(
            payloads
        )

    else:
        try:
            payload, endpoint = (
                request_overpass_json(
                    query_name,
                    build_overpass_query(
                        bbox,
                        selectors,
                    ),
                    cache_key=query_name,
                    refresh=refresh,
                )
            )

            return (
                payload,
                endpoint,
            )

        except RuntimeError:
            print(
                "-"
                * 70
            )

            print(
                f"Query {query_name} failed as one Ankara-wide request; "
                "falling back to 2x2 tiles."
            )

            payloads = []
            endpoints = []

            for child_index, child_bbox in enumerate(
                split_bbox_into_quadrants(
                    bbox
                ),
                start=1,
            ):
                (
                    child_payload,
                    child_endpoints,
                ) = download_tiled_query(
                    query_name,
                    child_bbox,
                    selectors,
                    refresh=refresh,
                    depth=1,
                    tile_key=(
                        f"tile_{child_index}"
                    ),
                )

                payloads.append(
                    child_payload
                )

                endpoints.extend(
                    child_endpoints
                )

            payload = merge_overpass_payloads(
                payloads
            )

    save_cached_payload(
        query_name,
        payload,
    )

    unique_endpoints = sorted(
        {
            endpoint
            for endpoint in endpoints
            if endpoint
            and endpoint
            != "cache"
        }
    )

    endpoint_summary = (
        "adaptive_tiles:"
        + (
            "|".join(
                unique_endpoints
            )
            if unique_endpoints
            else "cache"
        )
    )

    print(
        "-"
        * 70
    )

    print(
        f"Query {query_name}: assembled "
        f"{len(payload.get('elements', [])):,} unique elements "
        "from adaptive tiles."
    )

    return (
        payload,
        endpoint_summary,
    )


def element_coordinates(
    element: dict[
        str,
        Any,
    ],
) -> tuple[
    float,
    float,
] | None:
    """
    Return longitude/latitude for one Overpass element.

    Nodes use direct coordinates. Ways and relations use the Overpass center
    returned by `out center`.
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
        float(
            longitude
        ),
        float(
            latitude
        ),
    )


def _present_tag_value(
    tags: dict[
        str,
        Any,
    ],
    key: str,
) -> str:
    """Return a normalized non-empty OSM tag value."""

    value = str(
        tags.get(
            key,
            "",
        )
    ).strip()

    if value.lower() in {
        "",
        "no",
        "none",
        "null",
    }:
        return ""

    return value


def classify_tags(
    tags: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    bool,
]:
    """Classify one OSM feature into auditable urban-activity categories."""

    amenity = (
        _present_tag_value(
            tags,
            "amenity",
        )
        .lower()
    )

    shop = (
        _present_tag_value(
            tags,
            "shop",
        )
        .lower()
    )

    office = (
        _present_tag_value(
            tags,
            "office",
        )
        .lower()
    )

    healthcare = (
        _present_tag_value(
            tags,
            "healthcare",
        )
        .lower()
    )

    public_transport = (
        _present_tag_value(
            tags,
            "public_transport",
        )
        .lower()
    )

    highway = (
        _present_tag_value(
            tags,
            "highway",
        )
        .lower()
    )

    railway = (
        _present_tag_value(
            tags,
            "railway",
        )
        .lower()
    )

    aeroway = (
        _present_tag_value(
            tags,
            "aeroway",
        )
        .lower()
    )

    is_retail_commercial = (
        (
            bool(
                shop
            )
            and shop
            not in {
                "vacant",
                "closed",
            }
        )
        or (
            bool(
                office
            )
            and office
            not in {
                "vacant",
                "closed",
            }
        )
        or amenity
        in RETAIL_COMMERCIAL_AMENITIES
    )

    is_education = (
        amenity
        in EDUCATION_AMENITIES
    )

    is_healthcare = (
        bool(
            healthcare
        )
        or amenity
        in HEALTHCARE_AMENITIES
    )

    is_transport_activity = (
        bool(
            public_transport
        )
        or highway
        == "bus_stop"
        or railway
        in TRANSPORT_RAILWAY_VALUES
        or amenity
        in TRANSPORT_AMENITIES
        or aeroway
        in TRANSPORT_AEROWAY_VALUES
    )

    return {
        "is_retail_commercial": (
            is_retail_commercial
        ),
        "is_education": (
            is_education
        ),
        "is_healthcare": (
            is_healthcare
        ),
        "is_transport_activity": (
            is_transport_activity
        ),
    }


def combine_payloads(
    payloads: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> gpd.GeoDataFrame:
    """
    Merge all query payloads into one deduplicated activity inventory.

    `osm_uid` is based on OSM element type plus ID, so the same mapped feature
    returned by several category queries is counted once in total POI counts.
    Category flags may overlap when one OSM element genuinely carries tags
    from more than one activity family.
    """

    records: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for (
        query_name,
        payload,
    ) in payloads.items():
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

            element_type = str(
                element.get(
                    "type",
                    "",
                )
            )

            element_id = str(
                element.get(
                    "id",
                    "",
                )
            )

            if not element_type or not element_id:
                continue

            osm_uid = (
                f"{element_type}/{element_id}"
            )

            tags = dict(
                element.get(
                    "tags",
                    {},
                )
            )

            if osm_uid not in records:
                row: dict[
                    str,
                    Any,
                ] = {
                    "osm_uid": osm_uid,
                    "osm_type": element_type,
                    "osm_id": element_id,
                    "longitude": coordinates[0],
                    "latitude": coordinates[1],
                    "query_groups": set(),
                    "tags": tags,
                }

                records[
                    osm_uid
                ] = row

            records[
                osm_uid
            ][
                "query_groups"
            ].add(
                query_name
            )

            existing_tags = records[
                osm_uid
            ][
                "tags"
            ]

            for (
                key,
                value,
            ) in tags.items():
                if (
                    key
                    not in existing_tags
                    or not str(
                        existing_tags[
                            key
                        ]
                    ).strip()
                ):
                    existing_tags[
                        key
                    ] = value

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for osm_uid in sorted(
        records
    ):
        record = records[
            osm_uid
        ]

        tags = record[
            "tags"
        ]

        categories = (
            classify_tags(
                tags
            )
        )

        if not any(
            categories.values()
        ):
            continue

        row = {
            "osm_uid": record[
                "osm_uid"
            ],
            "osm_type": record[
                "osm_type"
            ],
            "osm_id": record[
                "osm_id"
            ],
            "query_groups": "|".join(
                sorted(
                    record[
                        "query_groups"
                    ]
                )
            ),
            **categories,
        }

        for tag_name in SELECTED_TAG_COLUMNS:
            tag_value = str(
                tags.get(
                    tag_name,
                    "",
                )
            ).strip()

            row[
                tag_name
            ] = (
                tag_value
                if tag_value
                else None
            )

        row[
            "osm_tags_json"
        ] = json.dumps(
            tags,
            ensure_ascii=False,
            sort_keys=True,
        )

        row[
            "geometry"
        ] = Point(
            float(
                record[
                    "longitude"
                ]
            ),
            float(
                record[
                    "latitude"
                ]
            ),
        )

        rows.append(
            row
        )

    if not rows:
        raise RuntimeError(
            "The Ankara activity queries returned no usable POIs."
        )

    frame = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    if frame[
        "osm_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate OSM activity IDs remained after merge."
        )

    return frame


def filter_to_exact_footprint(
    pois: gpd.GeoDataFrame,
    footprint: Any,
) -> gpd.GeoDataFrame:
    """Remove bbox-corner results outside the exact buffered footprint."""

    projected = (
        pois.to_crs(
            TARGET_CRS
        )
        if pois.crs != TARGET_CRS
        else pois.copy()
    )

    result = projected.loc[
        projected.geometry.intersects(
            footprint
        )
    ].copy()

    result.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        "Unique activity POIs inside exact buffer: "
        f"{len(result):,}"
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=TARGET_CRS,
    )


def validate_pois(
    pois: gpd.GeoDataFrame,
) -> None:
    """Validate the final activity inventory."""

    if pois.empty:
        raise ValueError(
            "Final activity inventory is empty."
        )

    if pois.crs is None:
        raise ValueError(
            "Activity inventory has no CRS."
        )

    if (
        pois.crs.to_string().upper()
        != TARGET_CRS
    ):
        raise ValueError(
            "Unexpected activity inventory CRS: "
            f"{pois.crs}"
        )

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

    category_columns = (
        "is_retail_commercial",
        "is_education",
        "is_healthcare",
        "is_transport_activity",
    )

    for column in category_columns:
        if column not in pois.columns:
            raise ValueError(
                f"Missing activity category column: {column}"
            )

        pois[
            column
        ] = pois[
            column
        ].astype(
            bool
        )

    category_matrix = pois[
        list(
            category_columns
        )
    ].to_numpy(
        dtype=bool
    )

    if not category_matrix.any(
        axis=1
    ).all():
        raise ValueError(
            "At least one POI has no activity category."
        )


def save_outputs(
    pois: gpd.GeoDataFrame,
    *,
    endpoints: dict[
        str,
        str,
    ],
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
) -> None:
    """Save the OSM activity inventory and metadata."""

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    pois.to_file(
        OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    category_columns = (
        "is_retail_commercial",
        "is_education",
        "is_healthcare",
        "is_transport_activity",
    )

    category_counts = {
        column: int(
            pois[
                column
            ].sum()
        )
        for column in category_columns
    }

    multi_category_count = int(
        (
            pois[
                list(
                    category_columns
                )
            ].sum(
                axis=1
            )
            > 1
        ).sum()
    )

    metadata = {
        "generated_at_utc": utc_now(),
        "method": (
            "split_sparse_overpass_bbox_queries_with_exact_buffer_filter"
        ),
        "download_buffer_m": DOWNLOAD_BUFFER_METERS,
        "query_bbox_wgs84": {
            "south": bbox[0],
            "west": bbox[1],
            "north": bbox[2],
            "east": bbox[3],
        },
        "query_specs": {
            name: list(
                selectors
            )
            for (
                name,
                selectors,
            ) in QUERY_SPECS.items()
        },
        "query_sources": endpoints,
        "unique_poi_count": int(
            len(
                pois
            )
        ),
        "category_counts": (
            category_counts
        ),
        "multi_category_poi_count": (
            multi_category_count
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

    print(
        "-"
        * 70
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


def print_results(
    pois: gpd.GeoDataFrame,
) -> None:
    """Print key activity-inventory diagnostics."""

    print(
        "-"
        * 70
    )

    print(
        "Unique activity POIs:",
        f"{len(pois):,}",
    )

    for (
        label,
        column,
    ) in (
        (
            "Retail/commercial",
            "is_retail_commercial",
        ),
        (
            "Education",
            "is_education",
        ),
        (
            "Healthcare",
            "is_healthcare",
        ),
        (
            "Transport activity",
            "is_transport_activity",
        ),
    ):
        print(
            f"{label}: "
            f"{int(pois[column].sum()):,}"
        )

    multi_category_count = int(
        (
            pois[
                [
                    "is_retail_commercial",
                    "is_education",
                    "is_healthcare",
                    "is_transport_activity",
                ]
            ].sum(
                axis=1
            )
            > 1
        ).sum()
    )

    print(
        "Multi-category POIs:",
        f"{multi_category_count:,}",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Download Ankara OSM urban-activity POIs."
        )
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignore cached Overpass JSON and redownload all query groups."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the Ankara urban-activity POI download."""

    arguments = parse_arguments()

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara OSM Activity POI Download"
    )

    print(
        "="
        * 70
    )

    create_directories()

    footprint, bbox = (
        load_download_footprint()
    )

    payloads: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    endpoints: dict[
        str,
        str,
    ] = {}

    for (
        query_name,
        selectors,
    ) in QUERY_SPECS.items():
        payload, endpoint = (
            download_overpass_json(
                query_name,
                bbox,
                selectors,
                refresh=arguments.refresh,
            )
        )

        payloads[
            query_name
        ] = payload

        endpoints[
            query_name
        ] = endpoint

    pois = combine_payloads(
        payloads
    )

    pois = (
        filter_to_exact_footprint(
            pois,
            footprint,
        )
    )

    validate_pois(
        pois
    )

    save_outputs(
        pois,
        endpoints=endpoints,
        bbox=bbox,
    )

    print_results(
        pois
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara OSM activity POI download completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
