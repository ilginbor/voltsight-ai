from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from voltsight.features.create_ankara_charging_features import (
    CHECKPOINT_COLUMNS,
    build_batch_metrics,
    load_cached_batch,
    normalize_station_flags,
    parse_arguments,
)
from voltsight.features.create_charging_features import (
    calculate_local_station_features,
    create_station_points,
)


def create_grid() -> gpd.GeoDataFrame:
    """Create synthetic Ankara grid cells."""

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


def create_stations() -> gpd.GeoDataFrame:
    """Create synthetic charging stations."""

    return gpd.GeoDataFrame(
        {
            "station_id": [
                "node_1",
                "node_2",
            ],
            "capacity_numeric": [
                4.0,
                float("nan"),
            ],
            "has_ac_connector": [
                True,
                False,
            ],
            "has_dc_connector": [
                False,
                True,
            ],
            "source_osm": [
                1,
                1,
            ],
            "source_epdk": [
                0,
                0,
            ],
        },
        geometry=[
            Point(250, 250),
            Point(750, 250),
        ],
        crs="EPSG:32636",
    )


def test_default_arguments() -> None:
    """Default batch size must be 5,000."""

    arguments = parse_arguments([])

    assert arguments.batch_size == 5_000
    assert not arguments.force
    assert not arguments.write_geojson


def test_station_flags_are_normalized() -> None:
    """GIS integer flags must become real booleans."""

    stations = create_stations()

    stations["has_ac_connector"] = [
        1,
        0,
    ]

    stations["has_dc_connector"] = [
        0,
        1,
    ]

    result = normalize_station_flags(
        stations
    )

    assert result[
        "has_ac_connector"
    ].tolist() == [
        True,
        False,
    ]

    assert result[
        "has_dc_connector"
    ].tolist() == [
        False,
        True,
    ]


def test_complete_batch_contains_all_features() -> None:
    """Synthetic batch must produce every charging metric."""

    grid = create_grid()
    stations = create_stations()

    station_points = create_station_points(
        stations
    )

    local_features = (
        calculate_local_station_features(
            grid,
            station_points,
        )
    )

    ac_points = station_points.loc[
        station_points[
            "has_ac_connector"
        ]
    ].copy()

    dc_points = station_points.loc[
        station_points[
            "has_dc_connector"
        ]
    ].copy()

    metrics = build_batch_metrics(
        grid_batch=grid,
        stations=stations,
        station_points=station_points,
        ac_station_points=ac_points,
        dc_station_points=dc_points,
        local_features=local_features,
    )

    assert len(metrics) == 2

    assert set(
        CHECKPOINT_COLUMNS
    ).issubset(
        metrics.columns
    )

    assert (
        metrics[
            "has_existing_charging_station"
        ] == 1
    ).all()

    assert (
        metrics[
            "distance_to_nearest_charging_station_m"
        ] >= 0
    ).all()

    assert (
        metrics[
            "charging_station_count_within_1000m"
        ]
        <= metrics[
            "charging_station_count_within_2000m"
        ]
    ).all()


def test_cached_batch_is_restored_in_order(
    tmp_path: Path,
) -> None:
    """Checkpoint must follow requested grid order."""

    output_path = (
        tmp_path / "batch_0001.csv"
    )

    frame = pd.DataFrame(
        {
            "grid_id": [
                "ANK_000002",
                "ANK_000001",
            ],
            "charging_station_count": [
                0,
                1,
            ],
            "has_existing_charging_station": [
                0,
                1,
            ],
            "distance_to_nearest_charging_station_m": [
                100.0,
                0.0,
            ],
            "charging_station_count_within_1000m": [
                1,
                1,
            ],
            "charging_station_count_within_2000m": [
                1,
                1,
            ],
            "known_charging_capacity": [
                0.0,
                4.0,
            ],
            "charging_capacity_record_count": [
                0,
                1,
            ],
            "ac_station_count_within_1000m": [
                0,
                1,
            ],
            "dc_station_count_within_1000m": [
                1,
                0,
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


def test_stale_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    """Wrong checkpoint IDs must not be silently reused."""

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
