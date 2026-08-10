from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_training_dataset import (
    FEATURE_COLUMNS,
    calculate_feature_summary,
    empirical_percentile,
    validate_feature_summary,
)


def create_training_frame() -> pd.DataFrame:
    """Create synthetic leakage-safe training data."""

    data: dict[str, list[float | int | str]] = {
        "grid_id": [
            "ANK_000001",
            "ANK_000002",
            "ANK_000003",
            "ANK_000004",
        ],
        "has_existing_charging_station": [
            0,
            0,
            0,
            1,
        ],
    }

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        data[feature] = [
            float(index),
            float(index + 1),
            float(index + 2),
            float(index + 10),
        ]

    return pd.DataFrame(
        data
    )


def test_empirical_percentile() -> None:
    """Empirical percentile must remain deterministic."""

    reference = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    assert empirical_percentile(
        reference,
        2.0,
    ) == 50.0

    assert empirical_percentile(
        reference,
        4.0,
    ) == 100.0


def test_feature_summary_has_one_row_per_feature() -> None:
    """All 14 predictors must be summarized."""

    summary = calculate_feature_summary(
        create_training_frame()
    )

    assert len(summary) == len(
        FEATURE_COLUMNS
    )

    assert summary[
        "feature"
    ].is_unique


def test_positive_high_values_have_high_percentile() -> None:
    """Synthetic positive medians should rank above negatives."""

    summary = calculate_feature_summary(
        create_training_frame()
    )

    assert (
        summary[
            "positive_median_percentile_vs_negative"
        ] == 100.0
    ).all()


def test_standardized_difference_is_finite() -> None:
    """All standardized differences must remain finite."""

    summary = calculate_feature_summary(
        create_training_frame()
    )

    assert np.isfinite(
        summary[
            "standardized_mean_difference"
        ].to_numpy(dtype=float)
    ).all()


def test_feature_summary_validation() -> None:
    """A complete summary must pass validation."""

    summary = calculate_feature_summary(
        create_training_frame()
    )

    validate_feature_summary(
        summary
    )
