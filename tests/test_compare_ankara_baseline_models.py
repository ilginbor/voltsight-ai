from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.compare_ankara_baseline_models import (
    calculate_top_fraction_metrics,
    create_comparison,
    validate_comparison,
)


def create_combined_oof() -> pd.DataFrame:
    """Create deterministic synthetic OOF predictions."""

    rows = []

    for index in range(
        100
    ):
        positive = (
            1
            if index < 10
            else 0
        )

        rows.append(
            {
                "grid_id": (
                    f"ANK_{index + 1:06d}"
                ),
                "has_existing_charging_station": positive,
                "spatial_block_id": (
                    f"B{index // 10}"
                ),
                "cv_fold": (
                    index % 5
                ),
                "logistic_score": (
                    0.9
                    if positive
                    else 0.1
                ),
                "random_forest_score": (
                    0.8
                    if positive
                    else 0.2
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_logistic_folds() -> pd.DataFrame:
    """Create synthetic Logistic fold metrics."""

    return pd.DataFrame(
        {
            "cv_fold": range(5),
            "logistic_average_precision": [
                0.7,
                0.8,
                0.75,
                0.85,
                0.9,
            ],
            "logistic_roc_auc": [
                0.9,
                0.91,
                0.92,
                0.93,
                0.94,
            ],
        }
    )


def create_rf_folds() -> pd.DataFrame:
    """Create synthetic Random Forest fold metrics."""

    return pd.DataFrame(
        {
            "cv_fold": range(5),
            "random_forest_average_precision": [
                0.8,
                0.85,
                0.82,
                0.88,
                0.9,
            ],
            "random_forest_roc_auc": [
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
            ],
        }
    )


def test_top_fraction_metrics_perfect_ranking() -> None:
    """Top-ranked fraction should recover positive samples first."""

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
            0.9,
            0.8,
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
        "precision"
    ] == 1.0

    assert metrics[
        "recall"
    ] == 1.0


def test_create_comparison_returns_two_models() -> None:
    """Comparison should contain both baseline models."""

    comparison = create_comparison(
        create_combined_oof(),
        create_logistic_folds(),
        create_rf_folds(),
    )

    assert len(
        comparison
    ) == 2

    assert set(
        comparison[
            "model"
        ]
    ) == {
        "Logistic Regression",
        "Random Forest",
    }


def test_perfect_synthetic_oof_has_perfect_ap() -> None:
    """Synthetic model ranking should achieve AP of one."""

    comparison = create_comparison(
        create_combined_oof(),
        create_logistic_folds(),
        create_rf_folds(),
    )

    assert np.allclose(
        comparison[
            "pooled_average_precision"
        ],
        1.0,
    )


def test_comparison_contains_top_fraction_metrics() -> None:
    """Comparison must report top-ranked retrieval metrics."""

    comparison = create_comparison(
        create_combined_oof(),
        create_logistic_folds(),
        create_rf_folds(),
    )

    assert (
        comparison[
            "top_1_percent_recall"
        ] > 0
    ).all()

    assert (
        comparison[
            "top_5_percent_recall"
        ] > 0
    ).all()


def test_validate_comparison() -> None:
    """Complete comparison output should validate."""

    combined = create_combined_oof()

    comparison = create_comparison(
        combined,
        create_logistic_folds(),
        create_rf_folds(),
    )

    synthetic = pd.concat(
        [
            combined,
            combined.iloc[
                :0
            ],
        ],
        ignore_index=True,
    )

    synthetic.loc[
        :45,
        "has_existing_charging_station",
    ] = 1

    synthetic.loc[
        46:,
        "has_existing_charging_station",
    ] = 0

    validate_comparison(
        synthetic,
        comparison,
    )
