from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from voltsight.models.create_ankara_diverse_candidate_shortlist import (
    MINIMUM_FEASIBILITY_SCORE,
    MINIMUM_NEED_SCORE,
    MINIMUM_SPACING_METERS,
    MINIMUM_SUITABILITY_SCORE,
    add_spacing_diagnostics,
    apply_eligibility_filters,
    select_spatially_diverse_candidates,
)


def create_scores() -> gpd.GeoDataFrame:
    """Create synthetic ranked Ankara candidates."""

    rows = []

    coordinates = [
        (0, 0),
        (1_000, 0),
        (6_000, 0),
        (12_000, 0),
    ]

    for index, (
        x_coordinate,
        y_coordinate,
    ) in enumerate(
        coordinates,
        start=1,
    ):
        rows.append(
            {
                "grid_id": f"ANK_{index:06d}",
                "suitability_score": 90.0 - index,
                "suitability_rank": index,
                "suitability_percentile": 100.0 - index,
                "priority_band": "A",
                "feasibility_score": 80.0,
                "need_score": 70.0,
                "accessibility_score": 80.0,
                "parking_score": 80.0,
                "infrastructure_gap_score": 70.0,
                "technology_gap_score": 70.0,
                "score_explanation": "test",
                "geometry": box(
                    x_coordinate,
                    y_coordinate,
                    x_coordinate + 500,
                    y_coordinate + 500,
                ),
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def test_ankara_spacing_is_five_kilometres() -> None:
    """Province-wide shortlist must use wider spacing."""

    assert MINIMUM_SPACING_METERS == 5_000.0


def test_eligibility_thresholds_match_cankaya_quality_rules() -> None:
    """Ankara must preserve the Çankaya quality thresholds."""

    assert MINIMUM_SUITABILITY_SCORE == 60.0
    assert MINIMUM_FEASIBILITY_SCORE == 60.0
    assert MINIMUM_NEED_SCORE == 50.0


def test_eligibility_filter_removes_low_score() -> None:
    """Candidates below quality thresholds must be rejected."""

    scores = create_scores()

    scores.loc[
        0,
        "suitability_score",
    ] = 50.0

    eligible = apply_eligibility_filters(
        scores
    )

    assert "ANK_000001" not in set(
        eligible["grid_id"]
    )


def test_greedy_selection_respects_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nearby high-ranked candidates must not both be selected."""

    import voltsight.models.create_ankara_diverse_candidate_shortlist as module

    monkeypatch.setattr(
        module,
        "DESIRED_CANDIDATE_COUNT",
        3,
    )

    selected = (
        select_spatially_diverse_candidates(
            create_scores()
        )
    )

    assert selected[
        "grid_id"
    ].tolist() == [
        "ANK_000001",
        "ANK_000003",
        "ANK_000004",
    ]


def test_spacing_diagnostics_report_nearest_candidate() -> None:
    """Selected-candidate diagnostics must report real distances."""

    selected = create_scores().iloc[
        [
            0,
            2,
            3,
        ]
    ].copy()

    result = add_spacing_diagnostics(
        selected
    )

    assert (
        result[
            "nearest_selected_candidate_m"
        ].min()
        >= 5_000.0
    )

    assert result[
        "nearest_selected_grid_id"
    ].notna().all()
