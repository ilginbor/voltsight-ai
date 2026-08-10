from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.create_ankara_suitability_scores import (
    ACCESSIBILITY_WEIGHTS,
    FEASIBILITY_WEIGHTS,
    NEED_WEIGHTS,
    PARKING_WEIGHTS,
    absence_score,
    assign_priority_band,
    create_suitability_scores,
    low_count_gap_score,
    percentile_score,
    weighted_score,
)


def create_candidates() -> pd.DataFrame:
    """Create three synthetic suitability candidates."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
            ],
            "main_road_segment_count": [
                2,
                1,
                0,
            ],
            "road_density_km_per_km2": [
                8.0,
                4.0,
                1.0,
            ],
            "distance_to_main_road_m": [
                100.0,
                500.0,
                2_000.0,
            ],
            "parking_area_m2": [
                1_000.0,
                500.0,
                0.0,
            ],
            "distance_to_nearest_parking_m": [
                100.0,
                500.0,
                5_000.0,
            ],
            "parking_count_within_1000m": [
                8,
                3,
                0,
            ],
            "distance_to_nearest_charging_station_m": [
                1_000.0,
                10_000.0,
                30_000.0,
            ],
            "charging_station_count_within_2000m": [
                2,
                0,
                0,
            ],
            "ac_station_count_within_1000m": [
                1,
                0,
                0,
            ],
            "dc_station_count_within_1000m": [
                1,
                0,
                0,
            ],
        }
    )


def test_weight_groups_sum_to_one() -> None:
    """Composite score weights must stay normalized."""

    for weights in (
        ACCESSIBILITY_WEIGHTS,
        PARKING_WEIGHTS,
        FEASIBILITY_WEIGHTS,
        NEED_WEIGHTS,
    ):
        assert np.isclose(
            sum(weights.values()),
            1.0,
        )


def test_lower_distance_scores_better() -> None:
    """Smaller feasibility distances must receive higher scores."""

    scores = percentile_score(
        pd.Series(
            [
                100.0,
                1_000.0,
                10_000.0,
            ]
        ),
        higher_is_better=False,
    )

    assert (
        scores.iloc[0]
        > scores.iloc[1]
        > scores.iloc[2]
    )


def test_low_station_count_creates_larger_gap() -> None:
    """Infrastructure scarcity must increase gap score."""

    scores = low_count_gap_score(
        pd.Series(
            [
                0,
                1,
                4,
            ]
        )
    )

    assert scores.iloc[0] > scores.iloc[2]


def test_absence_score_is_binary() -> None:
    """Missing AC/DC infrastructure must score 100."""

    scores = absence_score(
        pd.Series(
            [
                0,
                1,
                3,
            ]
        )
    )

    assert scores.tolist() == [
        100.0,
        0.0,
        0.0,
    ]


def test_weighted_score() -> None:
    """Weighted composites must respect configured weights."""

    frame = pd.DataFrame(
        {
            "a": [
                100.0,
            ],
            "b": [
                0.0,
            ],
        }
    )

    result = weighted_score(
        frame,
        {
            "a": 0.6,
            "b": 0.4,
        },
    )

    assert result.iloc[0] == 60.0


def test_priority_band_thresholds() -> None:
    """Priority percentile bands must remain deterministic."""

    result = assign_priority_band(
        pd.Series(
            [
                99.5,
                97.0,
                90.0,
                60.0,
                20.0,
            ]
        )
    )

    assert result.tolist() == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def test_complete_scoring_is_bounded_and_ranked() -> None:
    """Complete scoring must generate finite 0-100 outputs."""

    scores = create_suitability_scores(
        create_candidates()
    )

    assert len(scores) == 3

    assert scores[
        "grid_id"
    ].is_unique

    assert (
        scores[
            "suitability_score"
        ]
        .between(
            0,
            100,
        )
        .all()
    )

    assert scores[
        "suitability_rank"
    ].min() == 1

    assert scores[
        "score_explanation"
    ].notna().all()
