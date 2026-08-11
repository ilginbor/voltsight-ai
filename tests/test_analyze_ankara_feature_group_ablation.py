from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_feature_group_ablation import (
    ALL_FEATURE_COLUMNS,
    FEATURE_GROUPS,
    PARKING_FEATURE_COLUMNS,
    ROAD_FEATURE_COLUMNS,
    build_ablation_model,
    run_feature_group_ablation,
    run_single_configuration,
    validate_feature_definitions,
)
from voltsight.models.train_ankara_logistic_baseline import (
    FEATURE_COLUMNS,
)


def create_training_frame() -> pd.DataFrame:
    """Create deterministic synthetic data with five spatial folds."""

    rows: list[
        dict[str, float | int | str]
    ] = []

    grid_number = 1

    for fold in range(5):
        for local_index in range(30):
            positive = (
                1
                if local_index < 2
                else 0
            )

            row: dict[
                str,
                float | int | str,
            ] = {
                "grid_id": (
                    f"ANK_{grid_number:06d}"
                ),
                "has_existing_charging_station": positive,
                "spatial_block_id": (
                    f"B{fold}_{local_index // 5}"
                ),
                "cv_fold": fold,
                "fold_target": positive,
            }

            for feature_index, feature in enumerate(
                FEATURE_COLUMNS,
                start=1,
            ):
                signal = (
                    positive
                    * (
                        3.0
                        + feature_index * 0.05
                    )
                )

                row[feature] = (
                    float(feature_index)
                    + local_index * 0.1
                    + fold * 0.05
                    + signal
                )

            rows.append(
                row
            )

            grid_number += 1

    return pd.DataFrame(
        rows
    )


def test_feature_groups_partition_baseline_features() -> None:
    """Road and parking groups must exactly reconstruct the 14 predictors."""

    validate_feature_definitions()

    assert set(
        ROAD_FEATURE_COLUMNS
    ).isdisjoint(
        PARKING_FEATURE_COLUMNS
    )

    assert ALL_FEATURE_COLUMNS == tuple(
        FEATURE_COLUMNS
    )

    assert FEATURE_GROUPS[
        "all"
    ] == tuple(
        FEATURE_COLUMNS
    )


def test_expected_feature_group_sizes() -> None:
    """Ablation groups should preserve the intended 6+8 feature split."""

    assert len(
        ROAD_FEATURE_COLUMNS
    ) == 6

    assert len(
        PARKING_FEATURE_COLUMNS
    ) == 8

    assert len(
        ALL_FEATURE_COLUMNS
    ) == 14


def test_build_ablation_model_supports_all_baselines() -> None:
    """Every baseline model should be available through the ablation API."""

    for model_name in (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ):
        model = build_ablation_model(
            model_name
        )

        assert hasattr(
            model,
            "fit",
        )

        assert hasattr(
            model,
            "predict_proba",
        )


def test_single_logistic_configuration_produces_complete_oof_scores() -> None:
    """One ablation configuration must score every validation row once."""

    dataframe = create_training_frame()

    oof, fold_metrics, metrics = (
        run_single_configuration(
            dataframe,
            model_name="logistic_regression",
            feature_group="road_only",
            feature_columns=ROAD_FEATURE_COLUMNS,
        )
    )

    score_column = (
        "logistic_regression__road_only"
    )

    assert len(oof) == len(
        dataframe
    )

    assert not oof[
        score_column
    ].isna().any()

    assert oof[
        score_column
    ].between(
        0,
        1,
    ).all()

    assert len(
        fold_metrics
    ) == 5

    assert metrics[
        "feature_count"
    ] == len(
        ROAD_FEATURE_COLUMNS
    )

    assert 0 <= metrics[
        "pooled_average_precision"
    ] <= 1

    assert 0 <= metrics[
        "top_1_percent_recall"
    ] <= 1

    assert 0 <= metrics[
        "top_5_percent_recall"
    ] <= 1


def test_restricted_ablation_matrix_has_expected_shapes() -> None:
    """A restricted matrix is useful for fast regression testing."""

    dataframe = create_training_frame()

    metrics, fold_metrics, oof = (
        run_feature_group_ablation(
            dataframe,
            model_names=(
                "logistic_regression",
            ),
            feature_group_names=(
                "road_only",
                "parking_only",
                "all",
            ),
        )
    )

    assert len(metrics) == 3
    assert len(fold_metrics) == 15
    assert len(oof) == len(dataframe)

    assert set(
        metrics[
            "feature_group"
        ]
    ) == {
        "road_only",
        "parking_only",
        "all",
    }

    expected_score_columns = {
        "logistic_regression__road_only",
        "logistic_regression__parking_only",
        "logistic_regression__all",
    }

    assert expected_score_columns.issubset(
        oof.columns
    )

    numeric_metrics = metrics.select_dtypes(
        include=[np.number]
    )

    assert np.isfinite(
        numeric_metrics.to_numpy(
            dtype=float
        )
    ).all()
