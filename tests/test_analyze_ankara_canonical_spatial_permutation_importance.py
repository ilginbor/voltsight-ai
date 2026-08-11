from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.core.ankara_ml_features import (
    CANONICAL_ML_FEATURE_COLUMNS,
)
from voltsight.models.analyze_ankara_canonical_spatial_permutation_importance import (
    MODEL_NAMES,
    N_REPEATS,
    calculate_ap,
    feature_group_for,
    permutation_seed,
    permute_feature_values,
    validate_configuration,
)


def test_configuration_is_canonical_and_fixed() -> None:
    validate_configuration()

    assert len(
        CANONICAL_ML_FEATURE_COLUMNS
    ) == 15

    assert N_REPEATS == 5

    assert set(
        MODEL_NAMES
    ) == {
        "random_forest",
        "hist_gradient_boosting",
    }


def test_feature_groups_partition_canonical_features() -> None:
    groups = {
        feature_group_for(
            feature
        )
        for feature in CANONICAL_ML_FEATURE_COLUMNS
    }

    assert groups == {
        "road",
        "parking",
        "activity",
    }


def test_activity_features_are_classified_as_activity() -> None:
    for feature in (
        "poi_count",
        "poi_count_within_1000m",
        "poi_count_within_2000m",
    ):
        assert (
            feature_group_for(
                feature
            )
            == "activity"
        )


def test_permutation_seed_is_deterministic_and_model_specific() -> None:
    first = permutation_seed(
        model_name="random_forest",
        feature_index=2,
        fold=3,
        repeat=4,
    )

    second = permutation_seed(
        model_name="random_forest",
        feature_index=2,
        fold=3,
        repeat=4,
    )

    boosting = permutation_seed(
        model_name="hist_gradient_boosting",
        feature_index=2,
        fold=3,
        repeat=4,
    )

    assert first == second

    assert first != boosting


def test_permute_feature_preserves_multiset_and_other_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [
                1,
                2,
                3,
                4,
            ],
            "feature_b": [
                10,
                20,
                30,
                40,
            ],
        }
    )

    result = permute_feature_values(
        dataframe,
        feature="feature_a",
        random_state=42,
    )

    assert sorted(
        result[
            "feature_a"
        ].tolist()
    ) == [
        1,
        2,
        3,
        4,
    ]

    assert result[
        "feature_b"
    ].tolist() == [
        10,
        20,
        30,
        40,
    ]


def test_calculate_ap_is_bounded() -> None:
    score = calculate_ap(
        np.array(
            [
                0,
                0,
                1,
                0,
                1,
            ]
        ),
        np.array(
            [
                0.1,
                0.2,
                0.9,
                0.3,
                0.8,
            ]
        ),
    )

    assert 0.0 <= score <= 1.0


def test_feature_group_rejects_unknown_feature() -> None:
    try:
        feature_group_for(
            "not_a_feature"
        )
    except ValueError as error:
        assert (
            "Unknown canonical feature"
            in str(
                error
            )
        )
    else:
        raise AssertionError(
            "Unknown feature should raise ValueError."
        )
