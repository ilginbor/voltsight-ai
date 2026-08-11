from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_rf_redundancy_seed_stability import (
    FEATURE_SETS,
    FULL_14,
    NORMALIZED_12,
    RANDOM_STATES,
    build_model,
    run_seed_stability,
    run_single_configuration,
)
from voltsight.models.train_ankara_random_forest_baseline import (
    TARGET_COLUMN,
)


def create_training_frame() -> pd.DataFrame:
    """Create deterministic synthetic five-fold training data."""

    rows: list[dict[str, float | int | str]] = []

    feature_names = sorted(
        set(FULL_14)
    )

    grid_number = 1

    for fold in range(5):
        for local_index in range(20):
            positive = int(
                local_index < 2
            )

            row: dict[
                str,
                float | int | str,
            ] = {
                "grid_id": f"ANK_{grid_number:06d}",
                TARGET_COLUMN: positive,
                "spatial_block_id": (
                    f"B{fold}_{local_index // 4}"
                ),
                "cv_fold": fold,
                "fold_target": positive,
            }

            for feature_index, feature in enumerate(
                feature_names,
                start=1,
            ):
                value = (
                    float(feature_index)
                    + local_index * 0.1
                    + fold * 0.05
                    + positive * 3.0
                )

                row[feature] = value

            rows.append(row)
            grid_number += 1

    return pd.DataFrame(rows)


def test_feature_sets_have_expected_sizes() -> None:
    assert len(FULL_14) == 14
    assert len(NORMALIZED_12) == 12
    assert set(FEATURE_SETS) == {
        "full_14",
        "normalized_12",
    }
    assert "road_length_m" not in NORMALIZED_12
    assert "parking_area_m2" not in NORMALIZED_12


def test_build_model_preserves_baseline_configuration() -> None:
    model = build_model(
        random_state=123
    )

    assert model.n_estimators == 400
    assert model.max_depth == 12
    assert model.min_samples_leaf == 5
    assert model.max_features == "sqrt"
    assert model.class_weight == "balanced_subsample"
    assert model.random_state == 123


def test_single_configuration_returns_bounded_metrics() -> None:
    dataframe = create_training_frame()

    result = run_single_configuration(
        dataframe,
        feature_columns=NORMALIZED_12,
        random_state=42,
    )

    for column in [
        "pooled_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    ]:
        assert 0 <= float(result[column]) <= 1

    assert np.isfinite(
        float(
            result[
                "mean_fold_average_precision"
            ]
        )
    )

    assert np.isfinite(
        float(
            result[
                "std_fold_average_precision"
            ]
        )
    )


def test_single_configuration_is_deterministic_for_seed() -> None:
    dataframe = create_training_frame()

    first = run_single_configuration(
        dataframe,
        feature_columns=NORMALIZED_12,
        random_state=42,
    )

    second = run_single_configuration(
        dataframe,
        feature_columns=NORMALIZED_12,
        random_state=42,
    )

    assert (
        first["pooled_average_precision"]
        == second["pooled_average_precision"]
    )


def test_seed_stability_returns_one_row_per_feature_set_and_seed(
    monkeypatch,
) -> None:
    dataframe = create_training_frame()

    from voltsight.models import (
        analyze_ankara_rf_redundancy_seed_stability
        as module,
    )

    monkeypatch.setattr(
        module,
        "RANDOM_STATES",
        (42, 43),
    )

    metrics = run_seed_stability(
        dataframe
    )

    assert len(metrics) == 4
    assert set(metrics["feature_set"]) == {
        "full_14",
        "normalized_12",
    }
    assert set(metrics["random_state"]) == {
        42,
        43,
    }
