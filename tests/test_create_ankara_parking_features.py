from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from voltsight.features.create_ankara_parking_features import (
    CHECKPOINT_COLUMNS,
    build_batch_metrics,
    calculate_parking_area_batch,
    create_parking_union,
    load_cached_batch,
    parse_arguments,
)
from voltsight.features.create_parking_features import (
    calculate_local_parking_features,
    create_parking_points,
)


def create_grid() -> gpd.GeoDataFrame:
    """Create two synthetic grid cells."""

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
            box(0, 0, 500, 500),
            box(500, 0, 1_000, 500),
        ],
        crs="EPSG:32636",
    )


def create_parking() -> gpd.GeoDataFrame:
    """Create synthetic parking features."""

    return gpd.GeoDataFrame(
        {
            "parking_id": [
                "way_1",
                "node_2",
            ],
            "capacity_numeric": [
                20.0,
                float("nan"),
            ],
        },
        geometry=[
            box(100, 100, 200, 200),
            Point(750, 250),
        ],
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """Defaults must use 500-metre grids and 5,000 rows."""

    arguments = parse_arguments([])

    assert arguments.grid_size_m == 500
    assert arguments.batch_size == 5_000
    assert not arguments.force
    assert not arguments.write_geojson


def test_parking_union_prevents_double_counting() -> None:
    """Overlapping parking polygons must not double-count area."""

    parking = gpd.GeoDataFrame(
        {
            "parking_id": [
                "way_1",
                "way_2",
            ],
            "capacity_numeric": [
                float("nan"),
                float("nan"),
            ],
        },
        geometry=[
            box(0, 0, 60, 100),
            box(40, 0, 100, 100),
        ],
        crs="EPSG:32636",
    )

    grid = gpd.GeoDataFrame(
        {
            "grid_id": [
                "ANK_000001",
            ],
            "cell_area_m2": [
                10_000.0,
            ],
        },
        geometry=[
            box(0, 0, 100, 100),
        ],
        crs="EPSG:32636",
    )

    parking_union = create_parking_union(
        parking
    )

    result = calculate_parking_area_batch(
        grid,
        parking_union,
    )

    assert len(result) == 1

    assert result.iloc[0][
        "parking_area_m2"
    ] == pytest.approx(
        10_000.0
    )


def test_complete_batch_contains_all_metrics() -> None:
    """A synthetic batch must generate every parking metric."""

    grid = create_grid()
    parking = create_parking()

    parking_points = create_parking_points(
        parking
    )

    local_features = (
        calculate_local_parking_features(
            grid,
            parking_points,
        )
    )

    parking_union = create_parking_union(
        parking
    )

    metrics = build_batch_metrics(
        grid_batch=grid,
        parking=parking,
        parking_points=parking_points,
        parking_union=parking_union,
        local_features=local_features,
    )

    assert len(metrics) == 2

    assert set(
        CHECKPOINT_COLUMNS
    ).issubset(metrics.columns)

    assert metrics[
        "distance_to_nearest_parking_m"
    ].notna().all()

    assert (
        metrics["parking_count_within_500m"]
        <= metrics["parking_count_within_1000m"]
    ).all()


def test_cached_batch_is_restored_in_grid_order(
    tmp_path: Path,
) -> None:
    """Valid checkpoints must follow expected grid order."""

    output_path = (
        tmp_path / "batch_0001.csv"
    )

    frame = pd.DataFrame(
        {
            "grid_id": [
                "ANK_000002",
                "ANK_000001",
            ],
            "parking_count": [
                1,
                1,
            ],
            "parking_area_m2": [
                0.0,
                100.0,
            ],
            "parking_area_ratio": [
                0.0,
                0.0004,
            ],
            "distance_to_nearest_parking_m": [
                0.0,
                50.0,
            ],
            "parking_count_within_500m": [
                2,
                1,
            ],
            "parking_count_within_1000m": [
                2,
                2,
            ],
            "known_parking_capacity": [
                0.0,
                20.0,
            ],
            "parking_capacity_record_count": [
                0,
                1,
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

    assert loaded["grid_id"].tolist() == [
        "ANK_000001",
        "ANK_000002",
    ]


def test_stale_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    """Checkpoint IDs must match the requested grid batch."""

    output_path = (
        tmp_path / "batch_0001.csv"
    )

    frame = pd.DataFrame(
        {
            column: []
            for column in CHECKPOINT_COLUMNS
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
