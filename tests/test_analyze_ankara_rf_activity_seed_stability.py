from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_rf_activity_seed_stability import (
    FEATURE_SETS,
    RANDOM_STATES,
    build_model,
    create_aggregated_table,
    create_paired_delta_table,
    validate_configuration,
)


def create_metrics() -> pd.DataFrame:
    """Create deterministic paired seed metrics for summary tests."""

    rows = []

    for random_state in RANDOM_STATES:
        baseline_ap = (
            0.05
            + random_state
            / 100_000.0
        )

        activity_ap = (
            baseline_ap
            + 0.01
        )

        for (
            feature_set,
            pooled_ap,
            mean_fold_ap,
            top_one,
            top_five,
        ) in (
            (
                "normalized_12",
                baseline_ap,
                0.08,
                0.40,
                0.80,
            ),
            (
                "normalized_12_plus_activity_context",
                activity_ap,
                0.09,
                0.45,
                0.85,
            ),
        ):
            rows.append(
                {
                    "random_state": random_state,
                    "feature_set": feature_set,
                    "feature_set_label": feature_set,
                    "feature_count": len(
                        FEATURE_SETS[
                            feature_set
                        ]
                    ),
                    "pooled_average_precision": pooled_ap,
                    "mean_fold_average_precision": mean_fold_ap,
                    "std_fold_average_precision": 0.02,
                    "pooled_roc_auc": 0.90,
                    "top_1_percent_recall": top_one,
                    "top_5_percent_recall": top_five,
                }
            )

    return pd.DataFrame(
        rows
    )


def test_configuration_uses_12_and_15_features() -> None:
    validate_configuration()

    assert len(
        FEATURE_SETS[
            "normalized_12"
        ]
    ) == 12

    assert len(
        FEATURE_SETS[
            "normalized_12_plus_activity_context"
        ]
    ) == 15


def test_seed_list_is_fixed_and_unique() -> None:
    assert RANDOM_STATES == (
        42,
        43,
        44,
        45,
        46,
    )

    assert len(
        set(
            RANDOM_STATES
        )
    ) == len(
        RANDOM_STATES
    )


def test_random_forest_builder_preserves_baseline_settings() -> None:
    model = build_model(
        random_state=44
    )

    params = model.get_params()

    assert params[
        "n_estimators"
    ] == 400

    assert params[
        "max_depth"
    ] == 12

    assert params[
        "min_samples_leaf"
    ] == 5

    assert params[
        "max_features"
    ] == "sqrt"

    assert params[
        "class_weight"
    ] == "balanced_subsample"

    assert params[
        "random_state"
    ] == 44


def test_paired_delta_table_reports_activity_gain() -> None:
    paired = create_paired_delta_table(
        create_metrics()
    )

    assert len(
        paired
    ) == len(
        RANDOM_STATES
    )

    assert np.allclose(
        paired[
            "delta_pooled_ap"
        ],
        0.01,
    )

    assert np.allclose(
        paired[
            "delta_mean_fold_ap"
        ],
        0.01,
    )


def test_aggregated_table_contains_both_feature_sets() -> None:
    aggregated = (
        create_aggregated_table(
            create_metrics()
        )
    )

    assert set(
        aggregated[
            "feature_set"
        ]
    ) == set(
        FEATURE_SETS
    )

    activity = aggregated.loc[
        aggregated[
            "feature_set"
        ]
        == "normalized_12_plus_activity_context"
    ].iloc[
        0
    ]

    baseline = aggregated.loc[
        aggregated[
            "feature_set"
        ]
        == "normalized_12"
    ].iloc[
        0
    ]

    assert (
        activity[
            "mean_pooled_ap"
        ]
        > baseline[
            "mean_pooled_ap"
        ]
    )
