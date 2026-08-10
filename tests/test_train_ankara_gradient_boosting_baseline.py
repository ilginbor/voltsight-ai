from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.train_ankara_gradient_boosting_baseline import (
    FEATURE_COLUMNS,
    build_model,
    calculate_balanced_sample_weights,
    calculate_metrics,
    calculate_top_fraction_metrics,
    run_spatial_cross_validation,
    validate_outputs,
)


def create_training_frame() -> pd.DataFrame:
    """Create deterministic synthetic spatial-fold training data."""

    rows: list[
        dict[str, float | int | str]
    ] = []

    grid_number = 1

    for fold in range(
        5
    ):
        for local_index in range(
            60
        ):
            positive = (
                1
                if local_index < 3
                else 0
            )

            row: dict[
                str,
                float | int | str
            ] = {
                "grid_id": (
                    f"ANK_{grid_number:06d}"
                ),
                "has_existing_charging_station": positive,
                "spatial_block_id": (
                    f"B{fold}_{local_index // 10}"
                ),
                "cv_fold": fold,
                "fold_target": positive,
            }

            for feature_index, feature in enumerate(
                FEATURE_COLUMNS,
                start=1,
            ):
                base = float(
                    feature_index
                )

                row[
                    feature
                ] = (
                    base
                    + local_index * 0.05
                    + fold * 0.02
                    + positive * 4.0
                )

            rows.append(
                row
            )

            grid_number += 1

    return pd.DataFrame(
        rows
    )


def test_build_model_disables_internal_early_stopping() -> None:
    """Spatial CV should not be mixed with random internal stopping."""

    model = build_model()

    assert model.early_stopping is False
    assert model.max_iter == 150
    assert model.learning_rate == 0.05


def test_balanced_weights_upweight_positive_class() -> None:
    """Rare positive observations must receive larger weights."""

    y = np.array(
        [
            0,
            0,
            0,
            0,
            1,
        ]
    )

    weights = (
        calculate_balanced_sample_weights(
            y
        )
    )

    positive_weight = weights[
        y == 1
    ][0]

    negative_weight = weights[
        y == 0
    ][0]

    assert positive_weight > negative_weight

    assert np.isfinite(
        weights
    ).all()


def test_metrics_for_perfect_ranking() -> None:
    """Perfect scores should produce perfect ranking metrics."""

    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.2,
            0.8,
            0.9,
        ]
    )

    metrics = calculate_metrics(
        y_true,
        scores,
    )

    assert metrics[
        "average_precision"
    ] == 1.0

    assert metrics[
        "roc_auc"
    ] == 1.0

    assert metrics[
        "precision_at_0_5"
    ] == 1.0

    assert metrics[
        "recall_at_0_5"
    ] == 1.0


def test_top_fraction_metrics() -> None:
    """Top-ranked fraction should recover positives first."""

    y_true = np.array(
        [
            0,
            1,
            0,
            1,
            0,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
        ]
    )

    metrics = (
        calculate_top_fraction_metrics(
            y_true,
            scores,
            fraction=0.4,
        )
    )

    assert metrics[
        "selected_count"
    ] == 2.0

    assert metrics[
        "positive_count"
    ] == 2.0

    assert metrics[
        "precision"
    ] == 1.0

    assert metrics[
        "recall"
    ] == 1.0


def test_spatial_cv_produces_complete_oof_predictions() -> None:
    """Every synthetic row must receive an OOF prediction."""

    dataframe = (
        create_training_frame()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    assert len(
        oof
    ) == len(
        dataframe
    )

    assert not oof[
        "gradient_boosting_score"
    ].isna().any()

    assert oof[
        "gradient_boosting_score"
    ].between(
        0,
        1,
    ).all()

    assert len(
        fold_metrics
    ) == 5


def test_spatial_cv_preserves_validation_sizes() -> None:
    """Every synthetic fold should preserve its validation rows."""

    dataframe = (
        create_training_frame()
    )

    _, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    assert (
        fold_metrics[
            "validation_rows"
        ] == 60
    ).all()

    assert (
        fold_metrics[
            "validation_positives"
        ] == 3
    ).all()


def test_validate_outputs_accepts_complete_results() -> None:
    """Complete boosting outputs should pass validation."""

    dataframe = (
        create_training_frame()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
    )
