from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_spatial_permutation_importance import (
    FEATURE_COLUMNS,
    MODEL_NAMES,
    N_REPEATS,
    PARKING_FEATURES,
    ROAD_FEATURES,
    calculate_rank_metrics,
    feature_group_for,
    permutation_seed,
    permute_feature_values,
    validate_configuration,
    validate_outputs,
)


def test_feature_groups_exactly_partition_baseline_features() -> None:
    """Road and parking groups should cover every predictor once."""

    validate_configuration()

    assert ROAD_FEATURES + PARKING_FEATURES == FEATURE_COLUMNS
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)

    assert all(
        feature_group_for(feature) == "road"
        for feature in ROAD_FEATURES
    )

    assert all(
        feature_group_for(feature) == "parking"
        for feature in PARKING_FEATURES
    )


def test_permutation_is_deterministic_and_preserves_values() -> None:
    """Repeated use of one seed should produce the same permutation."""

    values = np.arange(30, dtype=float)

    first = permute_feature_values(values, seed=123)
    second = permute_feature_values(values, seed=123)

    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), values)
    assert not np.array_equal(first, values)


def test_permutation_seed_changes_by_fold_feature_and_repeat() -> None:
    """Independent experiment cells should receive distinct seeds."""

    seeds = {
        permutation_seed(
            "Random Forest",
            fold,
            feature_index,
            repeat,
        )
        for fold in range(2)
        for feature_index in range(3)
        for repeat in range(2)
    }

    assert len(seeds) == 12

    assert permutation_seed(
        "Random Forest",
        0,
        0,
        0,
    ) != permutation_seed(
        "HistGradientBoosting",
        0,
        0,
        0,
    )


def test_rank_metrics_detect_degraded_ranking() -> None:
    """Worse scores should reduce AP and ROC-AUC on a simple example."""

    y_true = np.array([0, 0, 0, 1, 1, 1])

    baseline_scores = np.array(
        [0.05, 0.10, 0.15, 0.80, 0.90, 0.95]
    )

    degraded_scores = np.array(
        [0.90, 0.10, 0.80, 0.15, 0.95, 0.05]
    )

    baseline = calculate_rank_metrics(
        y_true,
        baseline_scores,
    )

    degraded = calculate_rank_metrics(
        y_true,
        degraded_scores,
    )

    assert baseline["average_precision"] == 1.0
    assert baseline["roc_auc"] == 1.0

    assert degraded["average_precision"] < baseline["average_precision"]
    assert degraded["roc_auc"] < baseline["roc_auc"]


def test_validate_outputs_accepts_complete_synthetic_tables() -> None:
    """Expected model-feature-fold-repeat output shape should validate."""

    dataframe = pd.DataFrame(
        {
            "cv_fold": np.repeat(np.arange(5), 2),
        }
    )

    importance_records = []

    for model_name in MODEL_NAMES:
        for feature in FEATURE_COLUMNS:
            importance_records.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "feature_group": feature_group_for(feature),
                    "permutation_repeats": N_REPEATS,
                    "baseline_pooled_average_precision": 0.05,
                    "mean_permuted_pooled_average_precision": 0.04,
                    "mean_pooled_ap_drop": 0.01,
                    "std_pooled_ap_drop": 0.001,
                    "mean_fold_ap_drop": 0.01,
                    "std_fold_ap_drop": 0.002,
                    "baseline_pooled_roc_auc": 0.95,
                    "mean_permuted_pooled_roc_auc": 0.90,
                    "mean_pooled_roc_auc_drop": 0.05,
                    "std_pooled_roc_auc_drop": 0.003,
                    "mean_fold_roc_auc_drop": 0.05,
                    "std_fold_roc_auc_drop": 0.004,
                }
            )

    fold_records = []

    for model_name in MODEL_NAMES:
        for fold in range(5):
            for feature in FEATURE_COLUMNS:
                for repeat in range(N_REPEATS):
                    fold_records.append(
                        {
                            "model": model_name,
                            "cv_fold": fold,
                            "feature": feature,
                            "feature_group": feature_group_for(feature),
                            "repeat": repeat,
                            "validation_rows": 2,
                            "validation_positives": 1,
                            "baseline_average_precision": 0.05,
                            "permuted_average_precision": 0.04,
                            "ap_drop": 0.01,
                            "baseline_roc_auc": 0.95,
                            "permuted_roc_auc": 0.90,
                            "roc_auc_drop": 0.05,
                        }
                    )

    validate_outputs(
        dataframe,
        pd.DataFrame(importance_records),
        pd.DataFrame(fold_records),
    )
