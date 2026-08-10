from __future__ import annotations

import pandas as pd
import pytest

from voltsight.features.create_ankara_model_dataset import (
    CANDIDATE_OUTPUT_COLUMNS,
    CHARGING_CONTEXT_COLUMNS,
    CHARGING_LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    TRAINING_FEATURE_COLUMNS,
    TRAINING_OUTPUT_COLUMNS,
    normalize_target,
    prepare_candidate_dataset,
    prepare_training_dataset,
    validate_output_relationships,
)


def create_source() -> pd.DataFrame:
    """Create a minimal synthetic Ankara model source."""

    rows = 3

    data: dict[str, list[object]] = {
        "grid_id": [
            "ANK_000001",
            "ANK_000002",
            "ANK_000003",
        ],
        "cell_area_m2": [
            250_000.0,
        ] * rows,
        TARGET_COLUMN: [
            1,
            0,
            0,
        ],
    }

    for column in TRAINING_FEATURE_COLUMNS:
        data[column] = [
            1.0,
            2.0,
            3.0,
        ]

    for column in CHARGING_CONTEXT_COLUMNS:
        data[column] = [
            0.0,
            1.0,
            2.0,
        ]

    for column in CHARGING_LEAKAGE_COLUMNS:
        data[column] = [
            1.0,
            0.0,
            0.0,
        ]

    return pd.DataFrame(
        data
    )


def test_training_dataset_excludes_all_charging_predictors() -> None:
    """Training predictors must contain no charging-derived context."""

    source = create_source()

    training = prepare_training_dataset(
        source
    )

    assert list(
        training.columns
    ) == list(
        TRAINING_OUTPUT_COLUMNS
    )

    forbidden = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
    }

    assert not (
        forbidden
        & set(training.columns)
    )


def test_candidate_dataset_excludes_existing_station_cells() -> None:
    """Positive target rows must never become site candidates."""

    source = create_source()

    candidates = prepare_candidate_dataset(
        source
    )

    assert len(candidates) == 2

    assert candidates[
        "grid_id"
    ].tolist() == [
        "ANK_000002",
        "ANK_000003",
    ]

    assert list(
        candidates.columns
    ) == list(
        CANDIDATE_OUTPUT_COLUMNS
    )


def test_candidate_dataset_keeps_charging_context() -> None:
    """Infrastructure-gap context is needed for suitability scoring."""

    candidates = prepare_candidate_dataset(
        create_source()
    )

    assert set(
        CHARGING_CONTEXT_COLUMNS
    ).issubset(
        candidates.columns
    )

    assert not (
        set(CHARGING_LEAKAGE_COLUMNS)
        & set(candidates.columns)
    )


def test_binary_target_is_normalized() -> None:
    """Numeric 0/1 target values must become integers."""

    source = create_source()

    source[TARGET_COLUMN] = [
        1.0,
        0.0,
        0.0,
    ]

    result = normalize_target(
        source
    )

    assert result[
        TARGET_COLUMN
    ].dtype.kind in {
        "i",
        "u",
    }

    assert result[
        TARGET_COLUMN
    ].tolist() == [
        1,
        0,
        0,
    ]


def test_invalid_target_is_rejected() -> None:
    """Non-binary targets must fail."""

    source = create_source()

    source.loc[
        1,
        TARGET_COLUMN,
    ] = 2

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        normalize_target(
            source
        )


def test_output_partition_is_valid() -> None:
    """Training and candidate relationships must be consistent."""

    source = create_source()

    training = prepare_training_dataset(
        source
    )

    candidates = prepare_candidate_dataset(
        source
    )

    validate_output_relationships(
        source,
        training,
        candidates,
    )
