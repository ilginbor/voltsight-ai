from __future__ import annotations

import pandas as pd

from voltsight.core.ankara_ml_features import (
    ACTIVITY_CONTEXT_FEATURE_COLUMNS,
    CANONICAL_ML_FEATURE_COLUMNS,
    HISTORICAL_FULL_14_FEATURE_COLUMNS,
    REDUNDANT_SCALE_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from voltsight.features.create_ankara_canonical_ml_dataset import (
    CANDIDATE_OUTPUT_COLUMNS,
    TRAINING_OUTPUT_COLUMNS,
    build_canonical_datasets,
    validate_activity_features,
    validate_historical_training,
)


def create_historical_training() -> pd.DataFrame:
    """Create a small valid historical full-14 training table."""

    rows = []

    for index in range(
        1,
        5,
    ):
        row = {
            "grid_id": (
                f"ANK_{index:06d}"
            ),
            TARGET_COLUMN: (
                1
                if index == 1
                else 0
            ),
        }

        for feature_index, feature in enumerate(
            HISTORICAL_FULL_14_FEATURE_COLUMNS,
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


def create_activity() -> pd.DataFrame:
    """Create valid nested total-activity context."""

    return pd.DataFrame(
        {
            "grid_id": [
                "ANK_000001",
                "ANK_000002",
                "ANK_000003",
                "ANK_000004",
            ],
            "poi_count": [
                1,
                0,
                2,
                3,
            ],
            "poi_count_within_1000m": [
                3,
                0,
                5,
                7,
            ],
            "poi_count_within_2000m": [
                8,
                0,
                9,
                12,
            ],
        }
    )


def test_historical_training_validation_accepts_full_14() -> None:
    result = validate_historical_training(
        create_historical_training()
    )

    assert len(
        result
    ) == 4


def test_activity_validation_accepts_nested_counts() -> None:
    result = validate_activity_features(
        create_activity()
    )

    assert tuple(
        result.columns[
            1:
        ]
    ) == ACTIVITY_CONTEXT_FEATURE_COLUMNS


def test_canonical_dataset_has_exact_15_predictors() -> None:
    training, candidates = (
        build_canonical_datasets(
            create_historical_training(),
            create_activity(),
        )
    )

    assert tuple(
        training.columns
    ) == TRAINING_OUTPUT_COLUMNS

    assert tuple(
        candidates.columns
    ) == CANDIDATE_OUTPUT_COLUMNS

    assert len(
        CANONICAL_ML_FEATURE_COLUMNS
    ) == 15


def test_canonical_dataset_excludes_positive_cells_from_candidates() -> None:
    training, candidates = (
        build_canonical_datasets(
            create_historical_training(),
            create_activity(),
        )
    )

    positive_ids = set(
        training.loc[
            training[
                TARGET_COLUMN
            ]
            == 1,
            "grid_id",
        ]
    )

    assert not (
        positive_ids
        & set(
            candidates[
                "grid_id"
            ]
        )
    )

    assert len(
        candidates
    ) == 3


def test_canonical_dataset_excludes_redundant_scale_columns() -> None:
    training, _ = (
        build_canonical_datasets(
            create_historical_training(),
            create_activity(),
        )
    )

    assert not (
        set(
            REDUNDANT_SCALE_FEATURE_COLUMNS
        )
        & set(
            training.columns
        )
    )
