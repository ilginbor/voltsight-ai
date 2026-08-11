from __future__ import annotations

from voltsight.core.ankara_ml_features import (
    ACTIVITY_CONTEXT_FEATURE_COLUMNS,
    CANONICAL_ML_FEATURE_COLUMNS,
    CHARGING_CONTEXT_COLUMNS,
    CHARGING_LEAKAGE_COLUMNS,
    FEATURE_SET_REGISTRY,
    HISTORICAL_FULL_14_FEATURE_COLUMNS,
    NORMALIZED_12_FEATURE_COLUMNS,
    REDUNDANT_SCALE_FEATURE_COLUMNS,
    validate_feature_architecture,
)


def test_feature_architecture_validates() -> None:
    validate_feature_architecture()


def test_historical_and_normalized_feature_counts() -> None:
    assert len(
        HISTORICAL_FULL_14_FEATURE_COLUMNS
    ) == 14

    assert len(
        NORMALIZED_12_FEATURE_COLUMNS
    ) == 12


def test_normalized_12_removes_only_scale_duplicates() -> None:
    expected = tuple(
        feature
        for feature in HISTORICAL_FULL_14_FEATURE_COLUMNS
        if feature
        not in REDUNDANT_SCALE_FEATURE_COLUMNS
    )

    assert (
        NORMALIZED_12_FEATURE_COLUMNS
        == expected
    )


def test_canonical_15_extends_normalized_12_with_activity_context() -> None:
    assert len(
        CANONICAL_ML_FEATURE_COLUMNS
    ) == 15

    assert (
        CANONICAL_ML_FEATURE_COLUMNS
        == (
            *NORMALIZED_12_FEATURE_COLUMNS,
            *ACTIVITY_CONTEXT_FEATURE_COLUMNS,
        )
    )


def test_canonical_set_contains_no_charging_derived_predictors() -> None:
    forbidden = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
    }

    assert not (
        forbidden
        & set(
            CANONICAL_ML_FEATURE_COLUMNS
        )
    )

    assert (
        FEATURE_SET_REGISTRY[
            "canonical_activity_15"
        ]
        == CANONICAL_ML_FEATURE_COLUMNS
    )
