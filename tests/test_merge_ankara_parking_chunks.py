from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from voltsight.features.merge_ankara_parking_chunks import (
    deduplicate_parking_features,
    distance_token,
    filter_to_relevant_footprint,
    geometry_measure,
    parse_arguments,
)


def create_parking_frame() -> gpd.GeoDataFrame:
    """Create synthetic duplicate parking records."""

    return gpd.GeoDataFrame(
        {
            "parking_id": [
                "way_100",
                "way_100",
                "node_200",
            ],
            "source_chunk_id": [
                "ANK_ROAD_0001",
                "ANK_ROAD_0002",
                "ANK_ROAD_0001",
            ],
            "source_chunk_order": [
                1,
                2,
                1,
            ],
            "capacity_numeric": [
                10.0,
                10.0,
                float("nan"),
            ],
            "parking_area_m2": [
                100.0,
                400.0,
                0.0,
            ],
        },
        geometry=[
            box(0, 0, 10, 10),
            box(0, 0, 20, 20),
            Point(5, 5),
        ],
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """The merge defaults to 8-km chunks and a 1-km buffer."""

    arguments = parse_arguments([])

    assert arguments.chunk_size_m == 8_000
    assert arguments.download_buffer_m == 1_000
    assert not arguments.skip_preview


def test_distance_token() -> None:
    """Distance tokens must remain deterministic."""

    assert distance_token(8_000) == "8km"
    assert distance_token(750) == "750m"


def test_geometry_measure_uses_polygon_area() -> None:
    """Polygon completeness must be measured by area."""

    assert geometry_measure(
        box(0, 0, 20, 20)
    ) == 400.0


def test_spatial_filter_removes_irrelevant_records() -> None:
    """Records outside the Ankara buffer must be removed."""

    parking = create_parking_frame()

    outside = parking.iloc[[2]].copy()

    outside["parking_id"] = [
        "node_outside"
    ]

    outside["geometry"] = [
        Point(10_000, 10_000)
    ]

    combined = gpd.GeoDataFrame(
        list(parking.to_dict("records"))
        + list(outside.to_dict("records")),
        geometry="geometry",
        crs=parking.crs,
    )

    filtered = filter_to_relevant_footprint(
        combined,
        box(-100, -100, 100, 100),
    )

    assert "node_outside" not in set(
        filtered["parking_id"]
    )


def test_deduplication_keeps_largest_geometry() -> None:
    """The most complete duplicate geometry must be retained."""

    result, duplicate_count, variant_count = (
        deduplicate_parking_features(
            create_parking_frame()
        )
    )

    assert len(result) == 2
    assert duplicate_count == 1
    assert variant_count == 1
    assert result["parking_id"].is_unique

    retained = result.loc[
        result["parking_id"] == "way_100"
    ].iloc[0]

    assert retained.geometry.area == 400.0

    assert (
        retained[
            "source_chunk_occurrence_count"
        ]
        == 2
    )

    assert retained[
        "geometry_variant_count"
    ] == 2
