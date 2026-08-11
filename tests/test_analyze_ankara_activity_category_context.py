from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_activity_category_context import (
    ACTIVITY_COLUMNS,
    CATEGORY_1KM_CONTEXT,
    FEATURE_SETS,
    NORMALIZED_BASE_FEATURES,
    PARSIMONIOUS_MIXED_CONTEXT,
    RANDOM_FOREST_MAX_FEATURES,
    add_incremental_deltas,
    attach_activity_features,
    build_model,
    validate_activity_frame,
    validate_feature_sets,
)


def create_activity_frame() -> pd.DataFrame:
    """Create a valid synthetic category-context table."""

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
                10,
                20,
            ],
            "poi_count_within_2000m": [
                0,
                20,
                40,
            ],
            "retail_commercial_within_1000m": [
                0,
                4,
                8,
            ],
            "education_within_1000m": [
                0,
                2,
                3,
            ],
            "healthcare_within_1000m": [
                0,
                1,
                2,
            ],
            "transport_activity_within_1000m": [
                0,
                3,
                7,
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
            "normalized_12_plus_total_activity_context"
        ]
    ) == 15

    assert len(
        FEATURE_SETS[
            "normalized_12_plus_category_1km_context"
        ]
    ) == 16

    assert len(
        FEATURE_SETS[
            "normalized_12_plus_parsimonious_mixed_context"
        ]
    ) == 18


def test_parsimonious_mixed_context_excludes_redundant_total_1km() -> None:
    assert (
        "poi_count_within_1000m"
        not in PARSIMONIOUS_MIXED_CONTEXT
    )

    assert (
        "transport_activity_within_1000m"
        in PARSIMONIOUS_MIXED_CONTEXT
    )


def test_activity_validation_accepts_complete_context() -> None:
    result = validate_activity_frame(
        create_activity_frame()
    )

    assert tuple(
        result.columns[
            1:
        ]
    ) == ACTIVITY_COLUMNS


def test_category_1km_counts_cannot_exceed_total_1km_count() -> None:
    activity = create_activity_frame()

    activity.loc[
        1,
        CATEGORY_1KM_CONTEXT[
            0
        ],
    ] = 11

    try:
        validate_activity_frame(
            activity
        )
    except ValueError as error:
        assert "cannot exceed" in str(
            error
        )
    else:
        raise AssertionError(
            "Expected invalid category count to fail."
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
            ACTIVITY_COLUMNS
        )
    ].notna().all().all()


def test_random_forest_uses_fixed_three_features_per_split() -> None:
    model = build_model(
        "random_forest"
    )

    assert (
        model.get_params()[
            "max_features"
        ]
        == RANDOM_FOREST_MAX_FEATURES
        == 3
    )


def test_incremental_deltas_use_per_model_baseline() -> None:
    rows = []

    for (
        model_name,
        baseline_ap,
        category_ap,
    ) in (
        (
            "logistic_regression",
            0.10,
            0.12,
        ),
        (
            "random_forest",
            0.20,
            0.25,
        ),
        (
            "hist_gradient_boosting",
            0.30,
            0.28,
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
                    "feature_set": "normalized_12_plus_category_1km_context",
                    "pooled_average_precision": category_ap,
                    "mean_fold_average_precision": category_ap,
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

    random_forest_category = result.loc[
        (
            result[
                "model"
            ]
            == "random_forest"
        )
        & (
            result[
                "feature_set"
            ]
            == "normalized_12_plus_category_1km_context"
        )
    ].iloc[
        0
    ]

    assert np.isclose(
        random_forest_category[
            "delta_pooled_average_precision"
        ],
        0.05,
    )
