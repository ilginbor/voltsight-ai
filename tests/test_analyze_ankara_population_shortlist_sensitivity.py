from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import voltsight.models.create_ankara_diverse_candidate_shortlist as shortlist_module
from voltsight.models.analyze_ankara_population_shortlist_sensitivity import (
    ADJUSTED_RANK_COLUMN,
    ADJUSTED_SCORE_COLUMN,
    DEMAND_COLUMN,
    DEMAND_WEIGHT,
    SUITABILITY_WEIGHT,
    add_adjusted_score_and_rank,
    apply_original_quality_filters_for_adjusted_ranking,
    attach_demand_scores,
    calculate_comparison_metrics,
    create_adjusted_shortlist,
    create_current_shortlist,
    validate_demand_frame,
)


def create_scores() -> gpd.GeoDataFrame:
    """Create synthetic spatial candidates with canonical score columns."""

    coordinates = [
        (0, 0),
        (1_000, 0),
        (26_000, 0),
        (52_000, 0),
        (78_000, 0),
    ]

    rows = []

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


def create_demand() -> pd.DataFrame:
    """Create demand scores that favor later synthetic candidates."""

    return pd.DataFrame(
        {
            "grid_id": [
                f"ANK_{index:06d}"
                for index in range(
                    1,
                    6,
                )
            ],
            DEMAND_COLUMN: [
                10.0,
                20.0,
                30.0,
                95.0,
                100.0,
            ],
        }
    )


def test_demand_weights_sum_to_one() -> None:
    assert np.isclose(
        DEMAND_WEIGHT
        + SUITABILITY_WEIGHT,
        1.0,
    )


def test_attach_demand_scores_preserves_rows() -> None:
    result = attach_demand_scores(
        create_scores(),
        create_demand(),
    )

    assert len(
        result
    ) == 5

    assert result[
        DEMAND_COLUMN
    ].notna().all()


def test_adjusted_score_uses_fixed_five_percent_blend() -> None:
    result = add_adjusted_score_and_rank(
        attach_demand_scores(
            create_scores(),
            create_demand(),
        )
    )

    first = result.iloc[
        0
    ]

    expected = (
        SUITABILITY_WEIGHT
        * first[
            "suitability_score"
        ]
        + DEMAND_WEIGHT
        * first[
            DEMAND_COLUMN
        ]
    )

    assert np.isclose(
        first[
            ADJUSTED_SCORE_COLUMN
        ],
        expected,
    )

    assert result[
        ADJUSTED_RANK_COLUMN
    ].min() == 1


def test_adjusted_eligibility_keeps_original_quality_thresholds() -> None:
    scores = add_adjusted_score_and_rank(
        attach_demand_scores(
            create_scores(),
            create_demand(),
        )
    )

    scores.loc[
        scores[
            "grid_id"
        ]
        == "ANK_000005",
        "need_score",
    ] = 40.0

    eligible = (
        apply_original_quality_filters_for_adjusted_ranking(
            scores
        )
    )

    assert "ANK_000005" not in set(
        eligible[
            "grid_id"
        ]
    )


def test_current_and_adjusted_shortlists_respect_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shortlist_module,
        "DESIRED_CANDIDATE_COUNT",
        3,
    )

    scores = add_adjusted_score_and_rank(
        attach_demand_scores(
            create_scores(),
            create_demand(),
        )
    )

    current, _ = create_current_shortlist(
        scores
    )

    adjusted, _ = create_adjusted_shortlist(
        scores
    )

    assert len(
        current
    ) == 3

    assert len(
        adjusted
    ) == 3

    assert current[
        "nearest_selected_candidate_m"
    ].min() >= 25_000.0

    assert adjusted[
        "nearest_selected_candidate_m"
    ].min() >= 25_000.0


def test_comparison_metrics_report_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shortlist_module,
        "DESIRED_CANDIDATE_COUNT",
        3,
    )

    scores = add_adjusted_score_and_rank(
        attach_demand_scores(
            create_scores(),
            create_demand(),
        )
    )

    current, current_count = (
        create_current_shortlist(
            scores
        )
    )

    adjusted, adjusted_count = (
        create_adjusted_shortlist(
            scores
        )
    )

    metrics, common, removed, added = (
        calculate_comparison_metrics(
            current,
            adjusted,
            current_eligible_count=current_count,
            adjusted_eligible_count=adjusted_count,
        )
    )

    assert len(
        metrics
    ) == 2

    assert len(
        common
    ) <= 3

    assert len(
        removed
    ) == len(
        added
    )

    adjusted_row = metrics.loc[
        metrics[
            "scenario"
        ]
        == "demand_adjusted_5pct"
    ].iloc[
        0
    ]

    assert (
        0.0
        <= adjusted_row[
            "overlap_fraction_with_current"
        ]
        <= 1.0
    )


def test_validate_demand_frame_rejects_out_of_range_score() -> None:
    demand = create_demand()

    demand.loc[
        0,
        DEMAND_COLUMN,
    ] = 101.0

    with pytest.raises(
        ValueError
    ):
        validate_demand_frame(
            demand
        )
