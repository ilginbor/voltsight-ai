from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from voltsight.models import analyze_ankara_population_incremental_value as module
from voltsight.models.analyze_ankara_population_incremental_value import (
    FEATURE_SETS,
    NORMALIZED_BASE_FEATURES,
    POPULATION_COLUMNS,
    TARGET_COLUMN,
    add_incremental_deltas,
    attach_population_features,
    calculate_ranking_metrics,
    run_configuration,
    validate_feature_sets,
)


def create_baseline_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    grid_number = 1

    for fold in range(5):
        for local_index in range(20):
            positive = int(local_index < 2)
            row: dict[str, float | int | str] = {
                "grid_id": f"ANK_{grid_number:06d}",
                TARGET_COLUMN: positive,
                "spatial_block_id": f"B{fold}_{local_index // 4}",
                "cv_fold": fold,
                "fold_target": positive,
            }

            for feature_index, feature in enumerate(
                NORMALIZED_BASE_FEATURES,
                start=1,
            ):
                row[feature] = (
                    feature_index
                    + local_index * 0.03
                    + fold * 0.02
                    + positive * 1.5
                )

            rows.append(row)
            grid_number += 1

    return pd.DataFrame(rows)


def create_population_frame(baseline: pd.DataFrame) -> pd.DataFrame:
    population_count = np.where(
        baseline[TARGET_COLUMN].to_numpy(dtype=int) == 1,
        1000.0,
        25.0,
    )

    return pd.DataFrame(
        {
            "grid_id": baseline["grid_id"].astype(str),
            "population_count": population_count,
            "population_density_per_km2": population_count / 0.25,
            "population_within_1000m": population_count + 100.0,
            "population_within_2000m": population_count + 300.0,
        }
    )


def lightweight_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )


def test_feature_sets_have_expected_counts_and_no_population_duplicate() -> None:
    validate_feature_sets()

    assert len(FEATURE_SETS["normalized_12"]) == 12
    assert len(FEATURE_SETS["normalized_12_plus_local_population"]) == 13
    assert len(FEATURE_SETS["normalized_12_plus_population_context"]) == 15

    for features in FEATURE_SETS.values():
        assert not {
            "population_count",
            "population_density_per_km2",
        }.issubset(features)


def test_attach_population_features_preserves_rows_and_values() -> None:
    baseline = create_baseline_frame()
    population = create_population_frame(baseline)

    merged = attach_population_features(baseline, population)

    assert len(merged) == len(baseline)
    assert not merged[list(POPULATION_COLUMNS)].isna().any().any()
    assert np.allclose(
        merged["population_density_per_km2"],
        merged["population_count"] / 0.25,
    )


def test_calculate_ranking_metrics_perfect_scores() -> None:
    y_true = np.array([0, 0, 1, 0, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8], dtype=float)

    metrics = calculate_ranking_metrics(y_true, scores)

    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_run_configuration_produces_complete_oof(monkeypatch) -> None:
    baseline = create_baseline_frame()
    dataframe = attach_population_features(
        baseline,
        create_population_frame(baseline),
    )

    monkeypatch.setattr(
        module,
        "build_random_forest_model",
        lightweight_model,
    )

    oof, fold_metrics, aggregate = run_configuration(
        dataframe,
        model_name="random_forest",
        feature_set_name="normalized_12_plus_population_context",
        feature_columns=FEATURE_SETS["normalized_12_plus_population_context"],
    )

    assert len(oof) == len(dataframe)
    assert oof["score"].between(0, 1).all()
    assert len(fold_metrics) == 5
    assert aggregate["feature_count"] == 15


def test_add_incremental_deltas_uses_within_model_baseline() -> None:
    rows = []

    for model_name in (
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ):
        for index, feature_set_name in enumerate(FEATURE_SETS):
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "pooled_average_precision": 0.10 + index * 0.01,
                    "mean_fold_average_precision": 0.20 + index * 0.01,
                    "pooled_roc_auc": 0.80 + index * 0.01,
                    "top_1_percent_recall": 0.30 + index * 0.01,
                    "top_5_percent_recall": 0.60 + index * 0.01,
                }
            )

    result = add_incremental_deltas(pd.DataFrame(rows))

    baseline_rows = result[result["feature_set"] == "normalized_12"]
    assert np.allclose(
        baseline_rows["delta_pooled_average_precision"],
        0.0,
    )

    context_rows = result[
        result["feature_set"] == "normalized_12_plus_population_context"
    ]
    assert np.allclose(
        context_rows["delta_pooled_average_precision"],
        0.02,
    )
