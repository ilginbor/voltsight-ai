from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.train_ankara_random_forest_baseline import (
    FEATURE_COLUMNS,
    build_model,
    calculate_metrics,
    calculate_top_fraction_metrics,
    fit_full_model,
    run_spatial_cross_validation,
    validate_outputs,
)


def create_training_frame() -> pd.DataFrame:
    """Create deterministic synthetic spatial-fold training data."""

    rows: list[
        dict[str, float | int | str]
    ] = []

    grid_number = 1

    for fold in range(5):
        for local_index in range(40):
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
                row[feature] = (
                    float(feature_index)
                    + local_index * 0.05
                    + fold * 0.02
                    + positive * 5.0
                )

            rows.append(row)

            grid_number += 1

    return pd.DataFrame(rows)


def test_random_forest_configuration() -> None:
    """Baseline configuration should remain deterministic."""

    model = build_model()

    assert model.n_estimators == 400
    assert model.max_depth == 12
    assert model.min_samples_leaf == 5
    assert model.max_features == "sqrt"
    assert model.class_weight == "balanced_subsample"
    assert model.random_state == 42


def test_metrics_for_perfect_scores() -> None:
    """Perfect scores should produce perfect core metrics."""

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


def test_spatial_cv_creates_complete_oof_predictions() -> None:
    """Every row must receive one Random Forest OOF score."""

    dataframe = (
        create_training_frame()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    assert len(oof) == len(
        dataframe
    )

    assert not oof[
        "random_forest_score"
    ].isna().any()

    assert oof[
        "random_forest_score"
    ].between(
        0,
        1,
    ).all()

    assert len(
        fold_metrics
    ) == 5


def test_spatial_cv_preserves_fold_sizes() -> None:
    """Synthetic validation folds should remain unchanged."""

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
        ] == 40
    ).all()

    assert (
        fold_metrics[
            "validation_positives"
        ] == 3
    ).all()


def test_full_model_returns_all_feature_importances() -> None:
    """Full-data model should expose all predictor importances."""

    dataframe = (
        create_training_frame()
    )

    _, importance = (
        fit_full_model(
            dataframe
        )
    )

    assert len(
        importance
    ) == len(
        FEATURE_COLUMNS
    )

    assert importance[
        "feature"
    ].is_unique

    assert np.isclose(
        importance[
            "feature_importance"
        ].sum(),
        1.0,
    )


def test_validate_outputs_accepts_complete_results() -> None:
    """Complete Random Forest outputs should validate."""

    dataframe = (
        create_training_frame()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    _, importance = (
        fit_full_model(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
        importance,
    )
