from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.train_ankara_logistic_baseline import (
    FEATURE_COLUMNS,
    calculate_binary_metrics,
    calculate_top_fraction_metrics,
    fit_full_logistic_model,
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
            20
        ):
            positive = (
                1
                if local_index < 2
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
                    f"B{fold}_{local_index // 4}"
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
                    + local_index * 0.1
                    + fold * 0.05
                    + positive * 5.0
                )

            rows.append(
                row
            )

            grid_number += 1

    return pd.DataFrame(
        rows
    )


def test_binary_metrics_for_perfect_scores() -> None:
    """Perfect ranking should produce perfect core metrics."""

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

    metrics = calculate_binary_metrics(
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

    assert metrics[
        "f1_at_0_5"
    ] == 1.0


def test_top_fraction_metrics() -> None:
    """Highest-ranked samples should be evaluated correctly."""

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


def test_spatial_cross_validation_produces_all_oof_scores() -> None:
    """Every row must receive exactly one OOF score."""

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
        "logistic_score"
    ].isna().any()

    assert not oof[
        "dummy_score"
    ].isna().any()

    assert len(
        fold_metrics
    ) == 5


def test_each_validation_fold_preserves_rows() -> None:
    """Fold metrics should contain the expected validation size."""

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
        ] == 20
    ).all()

    assert (
        fold_metrics[
            "validation_positives"
        ] == 2
    ).all()


def test_full_logistic_model_has_one_coefficient_per_feature() -> None:
    """Full-data fit should expose all standardized coefficients."""

    dataframe = (
        create_training_frame()
    )

    _, coefficients = (
        fit_full_logistic_model(
            dataframe
        )
    )

    assert len(
        coefficients
    ) == len(
        FEATURE_COLUMNS
    )

    assert coefficients[
        "feature"
    ].is_unique

    assert np.isfinite(
        coefficients[
            "standardized_coefficient"
        ].to_numpy(
            dtype=float
        )
    ).all()


def test_validate_outputs_accepts_complete_results() -> None:
    """Complete model outputs should pass validation."""

    dataframe = (
        create_training_frame()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    _, coefficients = (
        fit_full_logistic_model(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
        coefficients,
    )


def test_logistic_scores_are_probabilities() -> None:
    """OOF logistic scores must stay inside 0-1."""

    dataframe = (
        create_training_frame()
    )

    oof, _ = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    assert oof[
        "logistic_score"
    ].between(
        0,
        1,
    ).all()


def test_dummy_scores_are_probabilities() -> None:
    """OOF dummy scores must stay inside 0-1."""

    dataframe = (
        create_training_frame()
    )

    oof, _ = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    assert oof[
        "dummy_score"
    ].between(
        0,
        1,
    ).all()
