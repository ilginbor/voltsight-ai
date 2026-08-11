from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_activity_incremental_value import (
    ACTIVITY_TOTAL_COLUMNS,
    FEATURE_SETS,
    NORMALIZED_BASE_FEATURES,
    add_incremental_deltas,
    attach_activity_features,
    calculate_ranking_metrics,
    validate_activity_frame,
    validate_feature_sets,
)


def create_activity_frame() -> pd.DataFrame:
    """Create a valid synthetic total-activity feature table."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
            ],
            "poi_count": [
                0,
                2,
                5,
            ],
            "poi_count_within_1000m": [
                0,
                5,
                10,
            ],
            "poi_count_within_2000m": [
                0,
                9,
                20,
            ],
        }
    )


def test_feature_sets_keep_expected_predictor_counts() -> None:
    validate_feature_sets()

    assert len(
        NORMALIZED_BASE_FEATURES
    ) == 12

    assert len(
        FEATURE_SETS[
            "normalized_12_plus_local_activity"
        ]
    ) == 13

    assert len(
        FEATURE_SETS[
            "normalized_12_plus_activity_context"
        ]
    ) == 15


def test_activity_validation_accepts_nested_nonnegative_counts() -> None:
    result = validate_activity_frame(
        create_activity_frame()
    )

    assert tuple(
        result.columns[
            1:
        ]
    ) == ACTIVITY_TOTAL_COLUMNS


def test_activity_validation_rejects_invalid_neighborhood_order() -> None:
    activity = create_activity_frame()

    activity.loc[
        1,
        "poi_count_within_1000m",
    ] = 1

    try:
        validate_activity_frame(
            activity
        )
    except ValueError as error:
        assert "1-km POI activity" in str(
            error
        )
    else:
        raise AssertionError(
            "Expected invalid nested activity counts to fail."
        )


def test_attach_activity_features_preserves_baseline_rows() -> None:
    baseline = pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
            ],
            "cv_fold": [
                0,
                1,
                2,
            ],
        }
    )

    result = attach_activity_features(
        baseline,
        create_activity_frame(),
    )

    assert len(
        result
    ) == len(
        baseline
    )

    assert result[
        list(
            ACTIVITY_TOTAL_COLUMNS
        )
    ].notna().all().all()


def test_ranking_metrics_are_bounded() -> None:
    metrics = calculate_ranking_metrics(
        np.array(
            [
                0,
                0,
                1,
                0,
                1,
            ]
        ),
        np.array(
            [
                0.1,
                0.2,
                0.9,
                0.3,
                0.8,
            ]
        ),
    )

    for value in metrics.values():
        assert (
            0.0
            <= value
            <= 1.0
        )


def test_incremental_deltas_use_per_model_normalized_baseline() -> None:
    rows = []

    for (
        model_name,
        baseline_ap,
        activity_ap,
    ) in (
        (
            "logistic_regression",
            0.10,
            0.12,
        ),
        (
            "random_forest",
            0.20,
            0.23,
        ),
        (
            "hist_gradient_boosting",
            0.30,
            0.29,
        ),
    ):
        rows.extend(
            [
                {
                    "model": model_name,
                    "feature_set": "normalized_12",
                    "pooled_average_precision": baseline_ap,
                    "mean_fold_average_precision": baseline_ap,
                    "pooled_roc_auc": 0.8,
                    "top_1_percent_recall": 0.4,
                    "top_5_percent_recall": 0.7,
                },
                {
                    "model": model_name,
                    "feature_set": "normalized_12_plus_activity_context",
                    "pooled_average_precision": activity_ap,
                    "mean_fold_average_precision": activity_ap,
                    "pooled_roc_auc": 0.81,
                    "top_1_percent_recall": 0.5,
                    "top_5_percent_recall": 0.8,
                },
            ]
        )

    result = add_incremental_deltas(
        pd.DataFrame(
            rows
        )
    )

    logistic_activity = result.loc[
        (
            result[
                "model"
            ]
            == "logistic_regression"
        )
        & (
            result[
                "feature_set"
            ]
            == "normalized_12_plus_activity_context"
        )
    ].iloc[
        0
    ]

    assert np.isclose(
        logistic_activity[
            "delta_pooled_average_precision"
        ],
        0.02,
    )
