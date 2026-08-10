from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from voltsight.models.create_ankara_result_visualizations import (
    priority_band_counts,
    validate_inputs,
)


def create_scores() -> gpd.GeoDataFrame:
    """Create synthetic suitability results."""

    return gpd.GeoDataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
            ],
            "suitability_score": [
                90.0,
                70.0,
                40.0,
            ],
            "feasibility_score": [
                85.0,
                75.0,
                50.0,
            ],
            "need_score": [
                95.0,
                65.0,
                30.0,
            ],
            "priority_band": [
                "A",
                "B",
                "E",
            ],
            "suitability_rank": [
                1,
                2,
                3,
            ],
        },
        geometry=[
            box(0, 0, 500, 500),
            box(6_000, 0, 6_500, 500),
            box(12_000, 0, 12_500, 500),
        ],
        crs="EPSG:32636",
    )


def create_shortlist() -> gpd.GeoDataFrame:
    """Create a valid 20-row synthetic shortlist."""

    rows = []

    for index in range(
        20
    ):
        rows.append(
            {
                "grid_id": (
                    f"ANK_{index + 1:06d}"
                ),
                "diverse_selection_rank": (
                    index + 1
                ),
                "suitability_score": (
                    90.0
                    - index * 0.5
                ),
                "feasibility_score": 80.0,
                "need_score": 75.0,
                "suitability_rank": (
                    index + 1
                ),
                "nearest_selected_candidate_m": (
                    5_000.0
                ),
                "geometry": box(
                    index * 6_000,
                    0,
                    index * 6_000 + 500,
                    500,
                ),
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def test_priority_band_counts_include_all_bands() -> None:
    """Missing bands must still appear with zero counts."""

    counts = priority_band_counts(
        create_scores()
    )

    assert counts.index.tolist() == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    assert counts.sum() == 3
    assert counts["C"] == 0
    assert counts["D"] == 0


def test_shortlist_ids_must_exist_in_scores() -> None:
    """Unknown shortlist IDs must be rejected."""

    scores = create_scores()

    shortlist = create_shortlist()

    with pytest.raises(
        ValueError,
        match="not present",
    ):
        validate_inputs(
            scores,
            shortlist,
        )


def test_valid_twenty_row_shortlist() -> None:
    """A complete synthetic shortlist must validate."""

    shortlist = create_shortlist()

    scores = shortlist[
        [
            "grid_id",
            "suitability_score",
            "feasibility_score",
            "need_score",
            "suitability_rank",
            "geometry",
        ]
    ].copy()

    scores[
        "priority_band"
    ] = "A"

    validate_inputs(
        scores,
        shortlist,
    )


def test_duplicate_score_ids_are_rejected() -> None:
    """Scored candidates must have unique IDs."""

    shortlist = create_shortlist()

    scores = shortlist[
        [
            "grid_id",
            "suitability_score",
            "feasibility_score",
            "need_score",
            "suitability_rank",
            "geometry",
        ]
    ].copy()

    scores[
        "priority_band"
    ] = "A"

    scores.loc[
        1,
        "grid_id",
    ] = scores.loc[
        0,
        "grid_id",
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate scored",
    ):
        validate_inputs(
            scores,
            shortlist,
        )
