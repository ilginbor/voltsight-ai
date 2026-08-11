from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_population_suitability_blend_sensitivity import (
    BLEND_WEIGHTS,
    DEMAND_COLUMN,
    calculate_blend_metrics,
    create_blend_scores,
    top_fraction_ids,
    validate_blend_weights,
    validate_input_frame,
    weight_slug,
)


def create_candidates() -> pd.DataFrame:
    """Create deterministic synthetic suitability and demand data."""

    count = 100

    suitability = np.linspace(
        90,
        10,
        count,
    )

    demand = np.linspace(
        10,
        100,
        count,
    )

    return pd.DataFrame(
        {
            "grid_id": [
                f"ANK_{index:06d}"
                for index in range(
                    1,
                    count + 1,
                )
            ],
            "feasibility_score": np.linspace(
                95,
                30,
                count,
            ),
            "need_score": np.linspace(
                90,
                20,
                count,
            ),
            "suitability_score": suitability,
            "suitability_rank": np.arange(
                1,
                count + 1,
            ),
            DEMAND_COLUMN: demand,
        }
    )


def test_blend_weights_include_unchanged_baseline() -> None:
    validate_blend_weights()

    assert BLEND_WEIGHTS[
        0
    ] == 0.0

    assert weight_slug(
        0.10
    ) == "demand_10"


def test_input_validation_accepts_valid_frame() -> None:
    dataframe = validate_input_frame(
        create_candidates()
    )

    assert len(
        dataframe
    ) == 100


def test_zero_weight_reproduces_current_suitability() -> None:
    dataframe = create_blend_scores(
        create_candidates()
    )

    assert np.allclose(
        dataframe[
            "blend_score_demand_00"
        ],
        dataframe[
            "suitability_score"
        ],
    )

    assert np.array_equal(
        dataframe[
            "blend_rank_demand_00"
        ].to_numpy(
            dtype=int
        ),
        dataframe[
            "suitability_rank"
        ].to_numpy(
            dtype=int
        ),
    )


def test_positive_demand_weight_moves_score_toward_demand() -> None:
    dataframe = create_blend_scores(
        create_candidates()
    )

    row = dataframe.iloc[
        -1
    ]

    assert (
        row[
            "blend_score_demand_20"
        ]
        > row[
            "blend_score_demand_00"
        ]
    )


def test_metrics_baseline_has_perfect_overlap_and_zero_shift() -> None:
    dataframe = create_blend_scores(
        create_candidates()
    )

    metrics = calculate_blend_metrics(
        dataframe
    )

    baseline = metrics.loc[
        metrics[
            "demand_weight"
        ]
        == 0.0
    ].iloc[
        0
    ]

    assert baseline[
        "spearman_with_baseline"
    ] == 1.0

    assert baseline[
        "top_1_percent_overlap_fraction"
    ] == 1.0

    assert baseline[
        "top_5_percent_overlap_fraction"
    ] == 1.0

    assert baseline[
        "top_20_overlap_fraction"
    ] == 1.0

    assert baseline[
        "median_absolute_rank_shift"
    ] == 0.0


def test_top_fraction_ids_is_deterministic() -> None:
    dataframe = create_blend_scores(
        create_candidates()
    )

    selected = top_fraction_ids(
        dataframe,
        "blend_score_demand_00",
        fraction=0.05,
    )

    assert len(
        selected
    ) == 5

    assert "ANK_000001" in selected
