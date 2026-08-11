from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from voltsight.features.create_ankara_activity_features import (
    CATEGORY_COLUMNS,
    OUTPUT_FEATURE_COLUMNS,
    assign_local_grid_ids,
    build_activity_features,
    coerce_boolean_series,
    count_points_within_radius,
    validate_activity_features,
)


def create_grid() -> gpd.GeoDataFrame:
    """Create a simple 2x2 500-m grid."""

    rows = []

    grid_number = 1

    for row_index in range(
        2
    ):
        for column_index in range(
            2
        ):
            min_x = (
                column_index
                * 500.0
            )

            min_y = (
                row_index
                * 500.0
            )

            rows.append(
                {
                    "grid_id": (
                        f"ANK_{grid_number:06d}"
                    ),
                    "geometry": box(
                        min_x,
                        min_y,
                        min_x + 500.0,
                        min_y + 500.0,
                    ),
                }
            )

            grid_number += 1

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def create_pois() -> gpd.GeoDataFrame:
    """Create inside-grid and buffered outside-grid synthetic POIs."""

    rows = [
        {
            "osm_uid": "node/1",
            "is_retail_commercial": True,
            "is_education": False,
            "is_healthcare": False,
            "is_transport_activity": False,
            "geometry": gpd.points_from_xy(
                [
                    250.0,
                ],
                [
                    250.0,
                ],
            )[
                0
            ],
        },
        {
            "osm_uid": "node/2",
            "is_retail_commercial": False,
            "is_education": True,
            "is_healthcare": False,
            "is_transport_activity": False,
            "geometry": gpd.points_from_xy(
                [
                    750.0,
                ],
                [
                    250.0,
                ],
            )[
                0
            ],
        },
        {
            "osm_uid": "node/3",
            "is_retail_commercial": False,
            "is_education": False,
            "is_healthcare": True,
            "is_transport_activity": True,
            "geometry": gpd.points_from_xy(
                [
                    1_250.0,
                ],
                [
                    250.0,
                ],
            )[
                0
            ],
        },
    ]

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def test_boolean_coercion_handles_geopackage_style_values() -> None:
    values = coerce_boolean_series(
        pd.Series(
            [
                "True",
                "False",
                "1",
                "0",
            ]
        )
    )

    assert values.tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_count_points_within_radius_uses_euclidean_distance() -> None:
    centers = np.array(
        [
            [
                0.0,
                0.0,
            ],
            [
                2_000.0,
                0.0,
            ],
        ]
    )

    points = np.array(
        [
            [
                500.0,
                0.0,
            ],
            [
                1_500.0,
                0.0,
            ],
        ]
    )

    counts = count_points_within_radius(
        centers,
        points,
        radius_m=1_000.0,
    )

    assert counts.tolist() == [
        1,
        1,
    ]


def test_assign_local_grid_ids_excludes_buffer_only_poi() -> None:
    assignments = assign_local_grid_ids(
        create_grid(),
        create_pois(),
    )

    assert set(
        assignments[
            "osm_uid"
        ]
    ) == {
        "node/1",
        "node/2",
    }


def test_build_activity_features_counts_local_and_buffered_context() -> None:
    result = build_activity_features(
        create_grid(),
        create_pois(),
    )

    first = result.loc[
        result[
            "grid_id"
        ]
        == "ANK_000001"
    ].iloc[
        0
    ]

    second = result.loc[
        result[
            "grid_id"
        ]
        == "ANK_000002"
    ].iloc[
        0
    ]

    assert first[
        "poi_count"
    ] == 1

    assert first[
        "retail_commercial_count"
    ] == 1

    assert second[
        "education_count"
    ] == 1

    # The third POI sits outside the grid but inside the neighborhood search
    # radius for the eastern cell, demonstrating buffered edge context.
    assert second[
        "poi_count_within_1000m"
    ] == 3

    assert second[
        "healthcare_within_1000m"
    ] == 1

    assert second[
        "transport_activity_within_1000m"
    ] == 1


def test_activity_feature_validation_accepts_complete_output() -> None:
    result = build_activity_features(
        create_grid(),
        create_pois(),
    )

    validate_activity_features(
        result,
        expected_rows=4,
    )

    assert all(
        column in result.columns
        for column in OUTPUT_FEATURE_COLUMNS
    )


def test_category_flags_are_individually_bounded_by_total_counts() -> None:
    result = build_activity_features(
        create_grid(),
        create_pois(),
    )

    for category_name in CATEGORY_COLUMNS:
        assert (
            result[
                f"{category_name}_count"
            ]
            <= result[
                "poi_count"
            ]
        ).all()

        assert (
            result[
                f"{category_name}_within_1000m"
            ]
            <= result[
                "poi_count_within_1000m"
            ]
        ).all()
