from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_candidate_ml_support import (
    MODEL_PERCENTILE_COLUMNS,
    MODEL_SCORE_COLUMNS,
    create_candidate_support,
    create_shortlist_support,
    percentile_rank_score,
    top_fraction_overlap,
)


def create_candidates() -> pd.DataFrame:
    """Create a four-row canonical candidate ID table."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
                "ANK_000004",
            ],
        }
    )


def create_oof() -> pd.DataFrame:
    """Create OOF rows containing one positive outside the candidates."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
                "ANK_000004",
                "ANK_000005",
            ],
            "has_existing_charging_station": [
                0,
                0,
                0,
                0,
                1,
            ],
            "cv_fold": [
                0,
                0,
                1,
                1,
                0,
            ],
            "logistic_regression_score": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.9,
            ],
            "random_forest_score": [
                0.2,
                0.1,
                0.4,
                0.3,
                0.9,
            ],
            "hist_gradient_boosting_score": [
                0.1,
                0.3,
                0.2,
                0.4,
                0.9,
            ],
        }
    )


def create_suitability() -> pd.DataFrame:
    """Create a matching suitability table."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
                "ANK_000004",
            ],
            "suitability_score": [
                90.0,
                80.0,
                70.0,
                60.0,
            ],
            "suitability_rank": [
                1,
                2,
                3,
                4,
            ],
            "suitability_percentile": [
                100.0,
                75.0,
                50.0,
                25.0,
            ],
            "priority_band": [
                "A",
                "B",
                "C",
                "D",
            ],
            "feasibility_score": [
                80.0,
                80.0,
                70.0,
                65.0,
            ],
            "need_score": [
                90.0,
                80.0,
                70.0,
                60.0,
            ],
        }
    )


def test_percentile_rank_score_is_monotonic_and_tie_aware() -> None:
    result = percentile_rank_score(
        pd.Series(
            [
                1.0,
                2.0,
                2.0,
                4.0,
            ]
        )
    )

    assert result.iloc[
        0
    ] == 25.0

    assert result.iloc[
        1
    ] == result.iloc[
        2
    ]

    assert result.iloc[
        3
    ] == 100.0


def test_candidate_support_uses_negative_oof_rows_only() -> None:
    result = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    assert len(
        result
    ) == 4

    assert "ANK_000005" not in set(
        result[
            "grid_id"
        ]
    )


def test_candidate_support_creates_all_model_percentiles() -> None:
    result = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    for column in MODEL_PERCENTILE_COLUMNS:
        assert column in result.columns

        assert result[
            column
        ].between(
            0.0,
            100.0,
        ).all()


def test_consensus_is_median_of_model_percentiles() -> None:
    result = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    matrix = result[
        list(
            MODEL_PERCENTILE_COLUMNS
        )
    ].to_numpy(
        dtype=float
    )

    expected = np.median(
        matrix,
        axis=1,
    )

    assert np.allclose(
        result[
            "ml_consensus_percentile"
        ],
        expected,
    )


def test_top_20_count_matches_model_percentiles() -> None:
    result = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    expected = (
        result[
            list(
                MODEL_PERCENTILE_COLUMNS
            )
        ].to_numpy(
            dtype=float
        )
        >= 80.0
    ).sum(
        axis=1
    )

    assert np.array_equal(
        result[
            "models_top_20pct_count"
        ].to_numpy(
            dtype=int
        ),
        expected,
    )


def test_candidate_percentiles_are_normalized_within_spatial_fold() -> None:
    result = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    ordered = result.set_index(
        "grid_id"
    )

    assert (
        ordered.loc[
            "ANK_000001",
            "logistic_regression_percentile",
        ]
        == 50.0
    )

    assert (
        ordered.loc[
            "ANK_000002",
            "logistic_regression_percentile",
        ]
        == 100.0
    )

    assert (
        ordered.loc[
            "ANK_000003",
            "logistic_regression_percentile",
        ]
        == 50.0
    )

    assert (
        ordered.loc[
            "ANK_000004",
            "logistic_regression_percentile",
        ]
        == 100.0
    )


def test_shortlist_support_preserves_saved_order() -> None:
    support = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    shortlist = pd.DataFrame(
        {
            "grid_id": [
                "ANK_000003",
                "ANK_000001",
            ],
        }
    )

    result = create_shortlist_support(
        shortlist,
        support,
    )

    assert result[
        "grid_id"
    ].tolist() == [
        "ANK_000003",
        "ANK_000001",
    ]

    assert result[
        "shortlist_order"
    ].tolist() == [
        1,
        2,
    ]


def test_top_fraction_overlap_is_bounded() -> None:
    support = create_candidate_support(
        create_candidates(),
        create_oof(),
        create_suitability(),
    )

    result = top_fraction_overlap(
        support,
        fraction=0.5,
    )

    assert result[
        "selected_count"
    ] == 2

    assert (
        0.0
        <= result[
            "overlap_fraction"
        ]
        <= 1.0
    )


def test_oof_score_columns_are_fixed() -> None:
    assert MODEL_SCORE_COLUMNS == (
        "logistic_regression_score",
        "random_forest_score",
        "hist_gradient_boosting_score",
    )
