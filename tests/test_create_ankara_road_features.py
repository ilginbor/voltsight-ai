from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from voltsight.features.create_ankara_road_features import (
    ROAD_METRIC_COLUMNS,
    add_nearest_main_road_features,
    calculate_intersection_features,
    compute_batch_features,
    load_cached_batch,
    normalize_main_road_column,
    parse_arguments,
    select_candidate_roads,
)


def create_grid() -> gpd.GeoDataFrame:
    """Create two synthetic 500-metre grid cells."""

    return gpd.GeoDataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
            ],
            "cell_area_m2": [
                250_000.0,
                250_000.0,
            ],
        },
        geometry=[
            box(
                500_000,
                4_400_000,
                500_500,
                4_400_500,
            ),
            box(
                500_500,
                4_400_000,
                501_000,
                4_400_500,
            ),
        ],
        crs="EPSG:32636",
    )


def create_roads() -> gpd.GeoDataFrame:
    """Create synthetic local and main-road geometries."""

    return gpd.GeoDataFrame(
        {
            "road_id": [
                "ROAD_1",
                "ROAD_2",
            ],
            "highway": [
                "residential",
                "primary",
            ],
            "is_main_road": [
                False,
                True,
            ],
        },
        geometry=[
            LineString(
                [
                    (500_000, 4_400_250),
                    (501_000, 4_400_250),
                ]
            ),
            LineString(
                [
                    (500_250, 4_399_900),
                    (500_250, 4_400_600),
                ]
            ),
        ],
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """The pipeline must default to 500-m grid and 5,000 rows."""

    arguments = parse_arguments(
        []
    )

    assert arguments.grid_size_m == 500
    assert arguments.batch_size == 5_000
    assert not arguments.force
    assert not arguments.write_geojson


def test_numeric_main_road_values_are_normalized() -> None:
    """Numeric GIS Boolean fields must become real Boolean values."""

    roads = create_roads()

    roads["is_main_road"] = [
        0,
        1,
    ]

    normalized = normalize_main_road_column(
        roads
    )

    assert normalized[
        "is_main_road"
    ].tolist() == [
        False,
        True,
    ]


def test_candidate_road_selection_uses_batch_bounds() -> None:
    """Only roads intersecting the grid-batch bounds are selected."""

    grid = create_grid()

    roads = create_roads()

    outside = roads.iloc[
        [0]
    ].copy()

    outside["road_id"] = [
        "OUTSIDE"
    ]

    outside["geometry"] = [
        LineString(
            [
                (600_000, 4_500_000),
                (601_000, 4_500_000),
            ]
        )
    ]

    all_roads = pd.concat(
        [
            roads,
            outside,
        ],
        ignore_index=True,
    )

    all_roads = gpd.GeoDataFrame(
        all_roads,
        geometry="geometry",
        crs=roads.crs,
    )

    selected = select_candidate_roads(
        all_roads,
        grid,
    )

    assert set(
        selected["road_id"]
    ) == {
        "ROAD_1",
        "ROAD_2",
    }


def test_intersection_features_cover_every_grid_cell() -> None:
    """Road aggregation must preserve road-free grid cells."""

    grid = create_grid()

    roads = create_roads()

    features = calculate_intersection_features(
        grid,
        roads,
    )

    assert len(features) == 2

    assert features[
        "grid_id"
    ].is_unique

    assert (
        features["road_length_m"]
        > 0
    ).all()

    assert (
        features["road_density_km_per_km2"]
        > 0
    ).all()


def test_nearest_main_road_features_have_no_missing_values() -> None:
    """Every grid centroid must match a nearest main road."""

    grid = create_grid()

    base = grid.copy()

    base["road_length_m"] = 0.0
    base["road_segment_count"] = 0
    base["main_road_length_m"] = 0.0
    base["main_road_segment_count"] = 0
    base["road_density_km_per_km2"] = 0.0

    main_roads = create_roads().loc[
        lambda frame: frame[
            "is_main_road"
        ]
    ].copy()

    result = add_nearest_main_road_features(
        base,
        main_roads,
    )

    assert result[
        "distance_to_main_road_m"
    ].notna().all()

    assert result[
        "nearest_main_road_type"
    ].eq(
        "primary"
    ).all()


def test_complete_batch_features_are_valid() -> None:
    """A synthetic batch must produce every required road metric."""

    grid = create_grid()

    roads = create_roads()

    main_roads = roads.loc[
        roads["is_main_road"]
    ].copy()

    features = compute_batch_features(
        grid,
        roads,
        main_roads,
    )

    assert len(features) == 2

    assert set(
        ROAD_METRIC_COLUMNS
    ).issubset(
        features.columns
    )

    assert features[
        "distance_to_main_road_m"
    ].notna().all()


def test_cached_batch_is_restored_in_expected_order(
    tmp_path: Path,
) -> None:
    """A valid checkpoint must be loaded in grid order."""

    output_path = (
        tmp_path / "batch_0001.csv"
    )

    frame = pd.DataFrame(
        {
            "grid_id": [
                "ANK_000002",
                "ANK_000001",
            ],
            "road_length_m": [
                2.0,
                1.0,
            ],
            "road_segment_count": [
                1,
                1,
            ],
            "main_road_length_m": [
                0.0,
                0.0,
            ],
            "main_road_segment_count": [
                0,
                0,
            ],
            "road_density_km_per_km2": [
                0.008,
                0.004,
            ],
            "distance_to_main_road_m": [
                20.0,
                10.0,
            ],
            "nearest_main_road_type": [
                "primary",
                "primary",
            ],
        }
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    loaded = load_cached_batch(
        output_path,
        [
            "ANK_000001",
            "ANK_000002",
        ],
    )

    assert loaded[
        "grid_id"
    ].tolist() == [
        "ANK_000001",
        "ANK_000002",
    ]


def test_cached_batch_with_wrong_ids_is_rejected(
    tmp_path: Path,
) -> None:
    """A stale checkpoint must not be silently reused."""

    output_path = (
        tmp_path / "batch_0001.csv"
    )

    frame = pd.DataFrame(
        {
            column: []
            for column in ROAD_METRIC_COLUMNS
        }
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        load_cached_batch(
            output_path,
            [
                "ANK_000001",
            ],
        )
