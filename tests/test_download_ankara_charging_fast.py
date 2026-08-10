from __future__ import annotations

from shapely.geometry import Point, box

import geopandas as gpd

from voltsight.features.download_ankara_charging_fast import (
    build_overpass_query,
    element_coordinates,
    filter_to_exact_footprint,
    overpass_json_to_gdf,
)


def test_query_uses_sparse_charging_tag() -> None:
    """One query must request all charging-station element types."""

    query = build_overpass_query(
        (
            39.0,
            32.0,
            41.0,
            34.0,
        )
    )

    assert 'amenity"="charging_station' in query
    assert "nwr" in query
    assert "out tags center" in query


def test_node_coordinates_are_read_directly() -> None:
    """Node coordinates must use lat/lon."""

    coordinates = element_coordinates(
        {
            "type": "node",
            "lat": 39.9,
            "lon": 32.8,
        }
    )

    assert coordinates == (
        32.8,
        39.9,
    )


def test_way_coordinates_use_overpass_center() -> None:
    """Ways must use the center returned by Overpass."""

    coordinates = element_coordinates(
        {
            "type": "way",
            "center": {
                "lat": 39.9,
                "lon": 32.8,
            },
        }
    )

    assert coordinates == (
        32.8,
        39.9,
    )


def test_overpass_json_becomes_geodataframe() -> None:
    """Tags and station identifiers must survive conversion."""

    payload = {
        "elements": [
            {
                "type": "node",
                "id": 123,
                "lat": 39.9,
                "lon": 32.8,
                "tags": {
                    "amenity": (
                        "charging_station"
                    ),
                    "name": "Test Station",
                    "socket:type2": "2",
                },
            }
        ]
    }

    result = overpass_json_to_gdf(
        payload
    )

    assert len(result) == 1
    assert result.iloc[0]["element"] == "node"
    assert result.iloc[0]["id"] == "123"
    assert result.iloc[0]["name"] == "Test Station"
    assert result.crs.to_epsg() == 4326


def test_exact_filter_removes_bbox_corner() -> None:
    """BBox extras must be removed using the real buffered footprint."""

    stations = gpd.GeoDataFrame(
        {
            "station_id": [
                "node_1",
                "node_2",
            ],
        },
        geometry=[
            Point(50, 50),
            Point(500, 500),
        ],
        crs="EPSG:32636",
    )

    result = filter_to_exact_footprint(
        stations,
        box(
            0,
            0,
            100,
            100,
        ),
    )

    assert result[
        "station_id"
    ].tolist() == [
        "node_1"
    ]
