from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.compare_ankara_model_rankings import (
    ENSEMBLE_COLUMN,
    add_fold_percentile_ranks,
    calculate_fold_metrics,
    calculate_top_fraction_metrics,
    create_comparison,
    create_rank_correlation,
)


def create_oof_frame() -> pd.DataFrame:
    """Create deterministic synthetic multi-model OOF scores."""

    rows = []

    grid_number = 1

    for fold in range(
        5
    ):
        for local_index in range(
            20
        ):
            positive = (
                1
                if local_index < 2
                else 0
            )

            rows.append(
                {
                    "grid_id": (
                        f"ANK_{grid_number:06d}"
                    ),
                    "has_existing_charging_station": (
                        positive
                    ),
                    "spatial_block_id": (
                        f"B{fold}_{local_index // 5}"
                    ),
                    "cv_fold": fold,
                    "logistic_score": (
                        0.9 - local_index * 0.01
                        if positive
                        else 0.4 - local_index * 0.01
                    ),
                    "random_forest_score": (
                        0.8 - local_index * 0.01
                        if positive
                        else 0.3 - local_index * 0.01
                    ),
                    "gradient_boosting_score": (
                        0.7 - local_index * 0.01
                        if positive
                        else 0.2 - local_index * 0.01
                    ),
                }
            )

            grid_number += 1

    return pd.DataFrame(
        rows
    )


def test_fold_percentile_ranks_stay_between_zero_and_one() -> None:
    """Every fold-normalized score must remain inside 0-1."""

    result = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    for column in [
        "logistic_rank",
        "random_forest_rank",
        "gradient_boosting_rank",
        ENSEMBLE_COLUMN,
    ]:
        assert result[
            column
        ].between(
            0,
            1,
        ).all()


def test_highest_score_receives_highest_fold_rank() -> None:
    """Highest score in each fold should receive rank one."""

    result = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    maximum_by_fold = (
        result.groupby(
            "cv_fold"
        )[
            "logistic_rank"
        ]
        .max()
    )

    assert np.allclose(
        maximum_by_fold,
        1.0,
    )


def test_rank_ensemble_is_equal_weight_mean() -> None:
    """Ensemble must be the exact unweighted average."""

    result = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    expected = (
        result[
            [
                "logistic_rank",
                "random_forest_rank",
                "gradient_boosting_rank",
            ]
        ]
        .mean(
            axis=1
        )
    )

    assert np.allclose(
        result[
            ENSEMBLE_COLUMN
        ],
        expected,
    )


def test_fold_metrics_preserve_all_five_folds() -> None:
    """Per-fold evaluation should contain five folds."""

    result = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    metrics = (
        calculate_fold_metrics(
            result,
            "logistic_rank",
        )
    )

    assert len(
        metrics
    ) == 5

    assert (
        metrics[
            "positive_count"
        ] == 2
    ).all()


def test_top_fraction_perfect_ranking() -> None:
    """Perfect highest-ranked positives should be fully recovered."""

    y_true = np.array(
        [
            1,
            1,
            0,
            0,
        ]
    )

    scores = np.array(
        [
            1.0,
            0.9,
            0.2,
            0.1,
        ]
    )

    metrics = (
        calculate_top_fraction_metrics(
            y_true,
            scores,
            fraction=0.5,
        )
    )

    assert metrics[
        "positive_count"
    ] == 2.0

    assert metrics[
        "recall"
    ] == 1.0

    assert metrics[
        "precision"
    ] == 1.0


def test_comparison_contains_four_rankings() -> None:
    """Comparison should contain three models and the ensemble."""

    dataframe = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    comparison = (
        create_comparison(
            dataframe
        )
    )

    assert len(
        comparison
    ) == 4

    assert set(
        comparison[
            "model"
        ]
    ) == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "Unweighted Rank Ensemble",
    }


def test_rank_correlation_has_three_models() -> None:
    """Correlation matrix should cover all base models."""

    dataframe = (
        add_fold_percentile_ranks(
            create_oof_frame()
        )
    )

    correlation = (
        create_rank_correlation(
            dataframe
        )
    )

    assert correlation.shape == (
        3,
        3,
    )

    assert np.allclose(
        np.diag(
            correlation
        ),
        1.0,
    )
