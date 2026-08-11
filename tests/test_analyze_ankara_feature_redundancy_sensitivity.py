from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_feature_redundancy_sensitivity import (
    FEATURE_SETS,
    FULL_FEATURE_COLUMNS,
    NORMALIZED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    calculate_redundancy_relations,
    run_sensitivity_analysis,
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

            road_length = (
                100.0
                + local_index * 10.0
                + fold
                + positive * 300.0
            )

            parking_area = (
                50.0
                + local_index * 5.0
                + fold
                + positive * 200.0
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
                row[feature] = (
                    float(feature_index)
                    + local_index * 0.1
                    + fold * 0.05
                    + positive * 3.0
                )

            row["road_length_m"] = road_length
            row["road_density_km_per_km2"] = (
                road_length * 0.004
            )

            row["parking_area_m2"] = parking_area
            row["parking_area_ratio"] = (
                parking_area * 0.000004
            )

            rows.append(
                row
            )

            grid_number += 1

    return pd.DataFrame(
        rows
    )


def test_feature_sets_have_expected_sizes_and_drops() -> None:
    """Feature sets should represent full and two 12-feature branches."""

    validate_feature_definitions()

    assert FULL_FEATURE_COLUMNS == tuple(
        FEATURE_COLUMNS
    )

    assert len(
        NORMALIZED_FEATURE_COLUMNS
    ) == 12

    assert len(
        RAW_FEATURE_COLUMNS
    ) == 12

    assert "road_length_m" not in (
        NORMALIZED_FEATURE_COLUMNS
    )

    assert "parking_area_m2" not in (
        NORMALIZED_FEATURE_COLUMNS
    )

    assert "road_density_km_per_km2" not in (
        RAW_FEATURE_COLUMNS
    )

    assert "parking_area_ratio" not in (
        RAW_FEATURE_COLUMNS
    )


def test_redundancy_relations_detect_exact_linear_transforms() -> None:
    """Synthetic scale conversions should have unit correlation."""

    dataframe = create_training_frame()

    relations = (
        calculate_redundancy_relations(
            dataframe
        )
    )

    assert len(
        relations
    ) == 2

    assert np.allclose(
        relations[
            "pearson_correlation"
        ],
        1.0,
    )

    road = relations.loc[
        relations[
            "raw_feature"
        ] == "road_length_m"
    ].iloc[0]

    parking = relations.loc[
        relations[
            "raw_feature"
        ] == "parking_area_m2"
    ].iloc[0]

    assert np.isclose(
        road[
            "ratio_median"
        ],
        0.004,
    )

    assert np.isclose(
        parking[
            "ratio_median"
        ],
        0.000004,
    )


def test_single_logistic_normalized_configuration_scores_all_rows() -> None:
    """Deduplicated logistic evaluation should produce complete fold results."""

    dataframe = create_training_frame()

    fold_metrics, metrics = (
        run_single_configuration(
            dataframe,
            model_name="logistic_regression",
            feature_set_name="normalized_12",
            feature_columns=NORMALIZED_FEATURE_COLUMNS,
        )
    )

    assert len(
        fold_metrics
    ) == 5

    assert metrics[
        "feature_count"
    ] == 12

    assert 0 <= metrics[
        "pooled_average_precision"
    ] <= 1

    assert 0 <= metrics[
        "pooled_roc_auc"
    ] <= 1


def test_restricted_sensitivity_matrix_has_expected_shape() -> None:
    """Fast logistic-only matrix should contain three feature sets."""

    dataframe = create_training_frame()

    metrics, fold_metrics = (
        run_sensitivity_analysis(
            dataframe,
            model_names=(
                "logistic_regression",
            ),
            feature_set_names=(
                "full_14",
                "normalized_12",
                "raw_12",
            ),
        )
    )

    assert len(
        metrics
    ) == 3

    assert len(
        fold_metrics
    ) == 15

    assert set(
        metrics[
            "feature_set"
        ]
    ) == set(
        FEATURE_SETS
    )


def test_sensitivity_numeric_metrics_are_finite() -> None:
    """Generated sensitivity diagnostics must be finite."""

    dataframe = create_training_frame()

    metrics, fold_metrics = (
        run_sensitivity_analysis(
            dataframe,
            model_names=(
                "logistic_regression",
            ),
        )
    )

    metrics_numeric = metrics.select_dtypes(
        include=[np.number]
    )

    folds_numeric = fold_metrics.select_dtypes(
        include=[np.number]
    )

    assert np.isfinite(
        metrics_numeric.to_numpy(
            dtype=float
        )
    ).all()

    assert np.isfinite(
        folds_numeric.to_numpy(
            dtype=float
        )
    ).all()
