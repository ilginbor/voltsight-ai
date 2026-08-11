from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from voltsight.features.download_ankara_activity_pois import (
    build_overpass_query,
    classify_tags,
    combine_payloads,
    element_coordinates,
    filter_to_exact_footprint,
    _is_transient_http_status,
    merge_overpass_payloads,
    split_bbox_into_quadrants,
)


def test_build_overpass_query_contains_all_selectors() -> None:
    query = build_overpass_query(
        (
            39.0,
            32.0,
            40.0,
            33.0,
        ),
        (
            '["shop"]',
            '["office"]',
        ),
    )

    assert 'nwr["shop"]' in query
    assert 'nwr["office"]' in query
    assert "out tags center;" in query


def test_element_coordinates_support_nodes_and_way_centers() -> None:
    node = {
        "type": "node",
        "id": 1,
        "lat": 39.9,
        "lon": 32.8,
    }

    way = {
        "type": "way",
        "id": 2,
        "center": {
            "lat": 39.8,
            "lon": 32.7,
        },
    }

    assert element_coordinates(
        node
    ) == (
        32.8,
        39.9,
    )

    assert element_coordinates(
        way
    ) == (
        32.7,
        39.8,
    )


def test_classify_tags_matches_expected_activity_families() -> None:
    retail = classify_tags(
        {
            "shop": "supermarket",
        }
    )

    education = classify_tags(
        {
            "amenity": "university",
        }
    )

    healthcare = classify_tags(
        {
            "healthcare": "physiotherapist",
        }
    )

    transport = classify_tags(
        {
            "highway": "bus_stop",
        }
    )

    assert retail[
        "is_retail_commercial"
    ]

    assert education[
        "is_education"
    ]

    assert healthcare[
        "is_healthcare"
    ]

    assert transport[
        "is_transport_activity"
    ]


def test_combine_payloads_deduplicates_same_osm_element() -> None:
    payloads = {
        "retail_shop": {
            "elements": [
                {
                    "type": "node",
                    "id": 10,
                    "lat": 39.9,
                    "lon": 32.8,
                    "tags": {
                        "shop": "convenience",
                        "public_transport": "platform",
                    },
                },
            ],
        },
        "transport_activity": {
            "elements": [
                {
                    "type": "node",
                    "id": 10,
                    "lat": 39.9,
                    "lon": 32.8,
                    "tags": {
                        "shop": "convenience",
                        "public_transport": "platform",
                    },
                },
            ],
        },
    }

    result = combine_payloads(
        payloads
    )

    assert len(
        result
    ) == 1

    row = result.iloc[
        0
    ]

    assert row[
        "osm_uid"
    ] == "node/10"

    assert row[
        "is_retail_commercial"
    ]

    assert row[
        "is_transport_activity"
    ]

    assert (
        row[
            "query_groups"
        ]
        == "retail_shop|transport_activity"
    )


def test_filter_to_exact_footprint_removes_bbox_corner_points() -> None:
    points = gpd.GeoDataFrame(
        {
            "osm_uid": [
                "node/1",
                "node/2",
            ],
        },
        geometry=gpd.points_from_xy(
            [
                500.0,
                5_000.0,
            ],
            [
                500.0,
                5_000.0,
            ],
        ),
        crs="EPSG:32636",
    )

    footprint = box(
        0,
        0,
        1_000,
        1_000,
    )

    result = filter_to_exact_footprint(
        points,
        footprint,
    )

    assert result[
        "osm_uid"
    ].tolist() == [
        "node/1",
    ]



def test_split_bbox_into_quadrants_covers_original_bbox() -> None:
    quadrants = split_bbox_into_quadrants(
        (
            0.0,
            0.0,
            2.0,
            4.0,
        )
    )

    assert quadrants == (
        (
            0.0,
            0.0,
            1.0,
            2.0,
        ),
        (
            0.0,
            2.0,
            1.0,
            4.0,
        ),
        (
            1.0,
            0.0,
            2.0,
            2.0,
        ),
        (
            1.0,
            2.0,
            2.0,
            4.0,
        ),
    )


def test_merge_overpass_payloads_deduplicates_tile_overlap() -> None:
    merged = merge_overpass_payloads(
        [
            {
                "elements": [
                    {
                        "type": "node",
                        "id": 1,
                        "lat": 39.9,
                        "lon": 32.8,
                        "tags": {
                            "office": "company",
                        },
                    },
                ],
            },
            {
                "elements": [
                    {
                        "type": "node",
                        "id": 1,
                        "lat": 39.9,
                        "lon": 32.8,
                        "tags": {
                            "office": "company",
                        },
                    },
                    {
                        "type": "node",
                        "id": 2,
                        "lat": 39.8,
                        "lon": 32.7,
                        "tags": {
                            "office": "government",
                        },
                    },
                ],
            },
        ]
    )

    assert [
        (
            element[
                "type"
            ],
            element[
                "id"
            ],
        )
        for element in merged[
            "elements"
        ]
    ] == [
        (
            "node",
            1,
        ),
        (
            "node",
            2,
        ),
    ]



def test_transient_http_status_classification() -> None:
    assert _is_transient_http_status(
        429
    )

    assert _is_transient_http_status(
        504
    )

    assert not _is_transient_http_status(
        400
    )

    assert not _is_transient_http_status(
        404
    )
