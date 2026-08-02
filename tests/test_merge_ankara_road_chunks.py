from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, box

from voltsight.features.merge_ankara_road_chunks import (
    canonical_geometry_key,
    clip_roads_to_core,
    deduplicate_road_pieces,
    parse_arguments,
)


def create_road_frame(
    geometries: list[LineString],
) -> gpd.GeoDataFrame:
    """Create synthetic road rows."""

    count = len(
        geometries
    )

    return gpd.GeoDataFrame(
        {
            "road_id": [
                f"SOURCE_{index}"
                for index in range(
                    1,
                    count + 1,
                )
            ],
            "source_chunk_id": [
                "ANK_ROAD_0001"
            ] * count,
            "source_chunk_order": [
                1
            ] * count,
            "osm_id": [
                "123"
            ] * count,
            "highway": [
                "primary"
            ] * count,
            "is_main_road": [
                True
            ] * count,
            "edge_length_m": [
                geometry.length
                for geometry in geometries
            ],
        },
        geometry=geometries,
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """The merge must default to eight-kilometre chunks."""

    arguments = parse_arguments(
        []
    )

    assert arguments.chunk_size_m == 8_000
    assert not arguments.skip_preview


def test_geometry_key_ignores_line_direction() -> None:
    """Opposite coordinate order must produce the same key."""

    forward = LineString(
        [
            (0, 0),
            (10, 10),
        ]
    )

    reverse = LineString(
        [
            (10, 10),
            (0, 0),
        ]
    )

    assert canonical_geometry_key(
        forward
    ) == canonical_geometry_key(
        reverse
    )


def test_roads_are_clipped_to_core_geometry() -> None:
    """Buffered road portions outside the core must be removed."""

    roads = create_road_frame(
        [
            LineString(
                [
                    (-5, 5),
                    (15, 5),
                ]
            )
        ]
    )

    clipped = clip_roads_to_core(
        roads=roads,
        core_geometry=box(
            0,
            0,
            10,
            10,
        ),
        target_crs="EPSG:32636",
        chunk_id="ANK_ROAD_0001",
        chunk_order=1,
    )

    assert len(clipped) == 1

    assert float(
        clipped.iloc[0][
            "edge_length_m"
        ]
    ) == 10.0

    assert clipped.iloc[0][
        "core_chunk_id"
    ] == "ANK_ROAD_0001"


def test_exact_boundary_duplicates_are_removed() -> None:
    """Identical normalized road pieces must be deduplicated."""

    roads = create_road_frame(
        [
            LineString(
                [
                    (0, 0),
                    (10, 10),
                ]
            ),
            LineString(
                [
                    (10, 10),
                    (0, 0),
                ]
            ),
        ]
    )

    roads["core_chunk_id"] = [
        "ANK_ROAD_0001",
        "ANK_ROAD_0002",
    ]

    roads["core_chunk_order"] = [
        1,
        2,
    ]

    roads.loc[
        1,
        "source_chunk_id",
    ] = "ANK_ROAD_0002"

    roads.loc[
        1,
        "source_chunk_order",
    ] = 2

    result, duplicate_count = (
        deduplicate_road_pieces(
            roads
        )
    )

    assert len(result) == 1
    assert duplicate_count == 1
    assert result["road_id"].is_unique
    assert result.iloc[0][
        "road_id"
    ].startswith(
        "ANK_ROAD_"
    )
