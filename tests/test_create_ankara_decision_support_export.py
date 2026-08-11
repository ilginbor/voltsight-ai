from __future__ import annotations

import pandas as pd

from voltsight.models.create_ankara_decision_support_export import (
    build_export_table,
    create_candidate_payload,
    create_json_payload,
    model_support_label,
    validate_export_table,
)


def create_shortlist() -> pd.DataFrame:
    """Create a minimal shortlist table."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
            ],
            "diverse_selection_rank": [
                1,
                2,
            ],
            "center_longitude": [
                32.80,
                32.90,
            ],
            "center_latitude": [
                39.90,
                40.00,
            ],
            "suitability_score": [
                90.0,
                80.0,
            ],
            "suitability_rank": [
                1,
                50,
            ],
            "suitability_percentile": [
                100.0,
                95.0,
            ],
            "priority_band": [
                "A",
                "A",
            ],
            "feasibility_score": [
                85.0,
                75.0,
            ],
            "need_score": [
                95.0,
                85.0,
            ],
            "accessibility_score": [
                90.0,
                80.0,
            ],
            "parking_score": [
                75.0,
                70.0,
            ],
            "infrastructure_gap_score": [
                98.0,
                90.0,
            ],
            "technology_gap_score": [
                80.0,
                70.0,
            ],
            "nearest_selected_grid_id": [
                "ANK_000002",
                "ANK_000001",
            ],
            "nearest_selected_candidate_m": [
                25_100.0,
                25_100.0,
            ],
            "score_explanation": [
                "Strong gap and feasibility.",
                "Balanced candidate.",
            ],
        }
    )


def create_ml_support() -> pd.DataFrame:
    """Create matching ML-support rows."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
            ],
            "shortlist_order": [
                1,
                2,
            ],
            "logistic_regression_percentile": [
                95.0,
                75.0,
            ],
            "random_forest_percentile": [
                96.0,
                92.0,
            ],
            "hist_gradient_boosting_percentile": [
                97.0,
                93.0,
            ],
            "ml_consensus_percentile": [
                96.0,
                92.0,
            ],
            "ml_consensus_rank": [
                10,
                100,
            ],
            "ml_min_percentile": [
                95.0,
                75.0,
            ],
            "ml_max_percentile": [
                97.0,
                93.0,
            ],
            "ml_model_spread": [
                2.0,
                18.0,
            ],
            "models_top_20pct_count": [
                3,
                2,
            ],
            "models_top_10pct_count": [
                3,
                2,
            ],
            "at_least_two_models_top_20pct": [
                True,
                True,
            ],
            "all_models_top_20pct": [
                True,
                False,
            ],
        }
    )


def test_support_label_is_transparent() -> None:
    assert (
        model_support_label(
            models_top_20pct_count=3
        )
        == "all_three_top_20pct"
    )

    assert (
        model_support_label(
            models_top_20pct_count=2
        )
        == "two_of_three_top_20pct"
    )


def test_build_export_preserves_selection_order() -> None:
    result = build_export_table(
        create_shortlist(),
        create_ml_support(),
    )

    assert result[
        "grid_id"
    ].tolist() == [
        "ANK_000001",
        "ANK_000002",
    ]

    assert result[
        "diverse_selection_rank"
    ].tolist() == [
        1,
        2,
    ]


def test_build_export_marks_model_disagreement() -> None:
    result = build_export_table(
        create_shortlist(),
        create_ml_support(),
    )

    assert result.loc[
        0,
        "has_model_disagreement",
    ] == False

    assert result.loc[
        1,
        "has_model_disagreement",
    ] == True


def test_validate_export_accepts_expected_row_count() -> None:
    result = build_export_table(
        create_shortlist(),
        create_ml_support(),
    )

    validate_export_table(
        result,
        expected_count=2,
    )


def test_candidate_payload_keeps_suitability_and_ml_separate() -> None:
    result = build_export_table(
        create_shortlist(),
        create_ml_support(),
    )

    payload = create_candidate_payload(
        result.iloc[
            0
        ]
    )

    assert (
        payload[
            "suitability"
        ][
            "score"
        ]
        == 90.0
    )

    assert (
        payload[
            "ml_support"
        ][
            "consensus_percentile"
        ]
        == 96.0
    )

    assert (
        "ml_support"
        not in payload[
            "suitability"
        ]
    )


def test_json_payload_declares_non_blended_policy() -> None:
    result = build_export_table(
        create_shortlist(),
        create_ml_support(),
    )

    payload = create_json_payload(
        result
    )

    assert payload[
        "candidate_count"
    ] == 2

    assert (
        payload[
            "decision_policy"
        ][
            "primary_layer"
        ]
        == "explainable_suitability"
    )

    assert (
        payload[
            "decision_policy"
        ][
            "ml_is_blended_into_suitability"
        ]
        is False
    )


def test_rank_mismatch_is_rejected() -> None:
    support = create_ml_support()

    support.loc[
        1,
        "shortlist_order",
    ] = 1

    try:
        build_export_table(
            create_shortlist(),
            support,
        )
    except ValueError as error:
        assert (
            "Shortlist order"
            in str(
                error
            )
        )
    else:
        raise AssertionError(
            "Rank mismatch should raise ValueError."
        )
