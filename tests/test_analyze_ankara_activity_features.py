from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_activity_features import (
    ACTIVITY_FEATURE_COLUMNS,
    create_correlation_table,
    create_distribution_table,
    create_redundancy_pairs,
    create_target_comparison,
    safe_standardized_mean_difference,
    validate_activity_frame,
)


def create_audit_frame() -> pd.DataFrame:
    """Create a deterministic synthetic activity audit dataset."""

    count = 20

    dataframe = pd.DataFrame(
        {
            "grid_id": [
                f"ANK_{index:06d}"
                for index in range(
                    1,
                    count + 1,
                )
            ],
            "has_existing_charging_station": (
                [
                    1,
                    1,
                ]
                + [
                    0,
                ]
                * 18
            ),
            "population_count": np.linspace(
                10,
                200,
                count,
            ),
            "population_within_1000m": np.linspace(
                100,
                2_000,
                count,
            ),
            "population_within_2000m": np.linspace(
                300,
                6_000,
                count,
            ),
        }
    )

    base = np.arange(
        count,
        dtype=float,
    )

    for feature_index, column in enumerate(
        ACTIVITY_FEATURE_COLUMNS
    ):
        dataframe[
            column
        ] = (
            base
            + feature_index
        )

    dataframe[
        "poi_count"
    ] = base

    dataframe[
        "poi_count_within_1000m"
    ] = (
        base
        + 5
    )

    dataframe[
        "poi_count_within_2000m"
    ] = (
        base
        + 10
    )

    for category in (
        "retail_commercial",
        "education",
        "healthcare",
        "transport_activity",
    ):
        dataframe[
            f"{category}_count"
        ] = np.minimum(
            dataframe[
                f"{category}_count"
            ],
            dataframe[
                "poi_count"
            ],
        )

        dataframe[
            f"{category}_within_1000m"
        ] = np.maximum(
            dataframe[
                f"{category}_within_1000m"
            ],
            dataframe[
                f"{category}_count"
            ],
        )

        dataframe[
            f"{category}_within_1000m"
        ] = np.minimum(
            dataframe[
                f"{category}_within_1000m"
            ],
            dataframe[
                "poi_count_within_1000m"
            ],
        )

    return dataframe


def test_activity_validation_accepts_complete_nonnegative_frame() -> None:
    dataframe = create_audit_frame()

    result = validate_activity_frame(
        dataframe
    )

    assert len(
        result
    ) == 20


def test_distribution_table_reports_all_features() -> None:
    distribution = create_distribution_table(
        create_audit_frame()
    )

    assert len(
        distribution
    ) == len(
        ACTIVITY_FEATURE_COLUMNS
    )

    assert set(
        distribution[
            "feature"
        ]
    ) == set(
        ACTIVITY_FEATURE_COLUMNS
    )


def test_target_comparison_reports_all_features() -> None:
    comparison = create_target_comparison(
        create_audit_frame()
    )

    assert len(
        comparison
    ) == len(
        ACTIVITY_FEATURE_COLUMNS
    )

    assert comparison[
        "standardized_mean_difference"
    ].notna().all()


def test_safe_smd_returns_zero_for_zero_variance_groups() -> None:
    result = safe_standardized_mean_difference(
        pd.Series(
            [
                1.0,
                1.0,
            ]
        ),
        pd.Series(
            [
                1.0,
                1.0,
                1.0,
            ]
        ),
    )

    assert result == 0.0


def test_correlation_table_includes_population_when_available() -> None:
    correlation = create_correlation_table(
        create_audit_frame()
    )

    assert "population_count" in correlation.columns

    assert "poi_count" in correlation.columns


def test_redundancy_pairs_are_unique_and_above_threshold() -> None:
    correlation = create_correlation_table(
        create_audit_frame()
    )

    redundancy = create_redundancy_pairs(
        correlation
    )

    pairs = {
        tuple(
            sorted(
                (
                    row.feature_a,
                    row.feature_b,
                )
            )
        )
        for row in redundancy.itertuples(
            index=False
        )
    }

    assert len(
        pairs
    ) == len(
        redundancy
    )

    assert (
        redundancy[
            "absolute_spearman_correlation"
        ]
        >= 0.90
    ).all()
