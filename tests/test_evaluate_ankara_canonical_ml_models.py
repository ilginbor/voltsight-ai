from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.core.ankara_ml_features import (
    CANONICAL_ML_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from voltsight.models.evaluate_ankara_canonical_ml_models import (
    MODEL_ORDER,
    attach_spatial_folds,
    build_model,
    calculate_ranking_metrics,
    validate_fold_frame,
    validate_training_frame,
)


def create_training() -> pd.DataFrame:
    """Create a small canonical-15 training frame."""

    rows = []

    for index in range(
        1,
        11,
    ):
        row = {
            "grid_id": (
                f"ANK_{index:06d}"
            ),
            TARGET_COLUMN: (
                1
                if index
                in {
                    1,
                    6,
                }
                else 0
            ),
        }

        for feature_index, feature in enumerate(
            CANONICAL_ML_FEATURE_COLUMNS,
            start=1,
        ):
            row[
                feature
            ] = float(
                index
                + feature_index
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def create_folds() -> pd.DataFrame:
    """Create a valid five-fold assignment with matching target values."""

    training = create_training()

    return pd.DataFrame(
        {
            "grid_id": training[
                "grid_id"
            ],
            "spatial_block_id": [
                f"BLOCK_{index:03d}"
                for index in range(
                    1,
                    11,
                )
            ],
            "cv_fold": [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ],
            TARGET_COLUMN: training[
                TARGET_COLUMN
            ],
        }
    )


def test_training_validation_requires_exact_canonical_predictors() -> None:
    result = validate_training_frame(
        create_training()
    )

    assert len(
        result
    ) == 10

    assert all(
        feature in result.columns
        for feature in CANONICAL_ML_FEATURE_COLUMNS
    )


def test_fold_validation_accepts_five_fold_ids() -> None:
    result = validate_fold_frame(
        create_folds()
    )

    assert set(
        result[
            "cv_fold"
        ]
    ) == {
        0,
        1,
        2,
        3,
        4,
    }


def test_attach_spatial_folds_preserves_rows_and_targets() -> None:
    merged = attach_spatial_folds(
        create_training(),
        create_folds(),
    )

    assert len(
        merged
    ) == 10

    assert "cv_fold" in merged.columns

    assert "spatial_block_id" in merged.columns


def test_model_builders_preserve_expected_estimator_families() -> None:
    models = {
        model_name: build_model(
            model_name
        )
        for model_name in MODEL_ORDER
    }

    assert (
        models[
            "logistic_regression"
        ].named_steps[
            "classifier"
        ].class_weight
        == "balanced"
    )

    assert (
        models[
            "random_forest"
        ].get_params()[
            "n_estimators"
        ]
        == 400
    )

    assert (
        models[
            "hist_gradient_boosting"
        ].get_params()[
            "max_iter"
        ]
        == 150
    )


def test_ranking_metrics_are_bounded_and_include_lift() -> None:
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

    assert (
        0.0
        <= metrics[
            "average_precision"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= metrics[
            "roc_auc"
        ]
        <= 1.0
    )

    assert metrics[
        "top_1_percent_lift"
    ] >= 0.0

    assert metrics[
        "top_5_percent_lift"
    ] >= 0.0
