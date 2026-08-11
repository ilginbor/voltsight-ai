from __future__ import annotations

import numpy as np
import pandas as pd

from voltsight.models.analyze_ankara_population_demand_sensitivity import (
    COMPONENT_SCORE_COLUMNS,
    DEMAND_SCENARIOS,
    add_demand_scenarios,
    calculate_scenario_metrics,
    create_correlation_table,
    create_population_component_scores,
    validate_population_frame,
    validate_scenario_weights,
)


def create_candidates() -> pd.DataFrame:
    """Create deterministic synthetic suitability/population candidates."""

    return pd.DataFrame(
        {
            "grid_id": [
                f"ANK_{index:06d}"
                for index in range(
                    1,
                    101,
                )
            ],
            "population_count": np.linspace(
                0,
                990,
                100,
            ),
            "population_density_per_km2": np.linspace(
                0,
                3960,
                100,
            ),
            "population_within_1000m": np.linspace(
                0,
                5000,
                100,
            ),
            "population_within_2000m": np.linspace(
                0,
                15000,
                100,
            ),
            "feasibility_score": np.linspace(
                20,
                90,
                100,
            ),
            "need_score": np.linspace(
                90,
                20,
                100,
            ),
            "suitability_score": np.linspace(
                10,
                100,
                100,
            ),
            "suitability_rank": np.arange(
                100,
                0,
                -1,
            ),
        }
    )


def test_scenario_weights_are_normalized() -> None:
    validate_scenario_weights()

    for weights in DEMAND_SCENARIOS.values():
        assert np.isclose(
            sum(
                weights.values()
            ),
            1.0,
        )


def test_population_validation_accepts_fixed_cell_density_transform() -> None:
    dataframe = create_candidates()

    result = validate_population_frame(
        dataframe[
            [
                "grid_id",
                "population_count",
                "population_density_per_km2",
                "population_within_1000m",
                "population_within_2000m",
            ]
        ]
    )

    assert len(
        result
    ) == 100


def test_zero_population_receives_zero_component_scores() -> None:
    result = create_population_component_scores(
        create_candidates()
    )

    zero = result.iloc[
        0
    ]

    for column in COMPONENT_SCORE_COLUMNS:
        assert zero[
            column
        ] == 0.0

    assert (
        result.iloc[
            -1
        ][
            "local_population_score"
        ]
        == 100.0
    )


def test_demand_scenarios_are_bounded_and_ranked() -> None:
    result = add_demand_scenarios(
        create_candidates()
    )

    for scenario_name in DEMAND_SCENARIOS:
        score_column = (
            f"demand_score_{scenario_name}"
        )

        rank_column = (
            f"demand_rank_{scenario_name}"
        )

        assert result[
            score_column
        ].between(
            0,
            100,
        ).all()

        assert result[
            rank_column
        ].min() == 1


def test_scenario_metrics_and_correlations_are_complete() -> None:
    result = add_demand_scenarios(
        create_candidates()
    )

    metrics = calculate_scenario_metrics(
        result
    )

    correlation = create_correlation_table(
        result
    )

    assert len(
        metrics
    ) == len(
        DEMAND_SCENARIOS
    )

    assert set(
        metrics[
            "scenario"
        ]
    ) == set(
        DEMAND_SCENARIOS
    )

    assert metrics[
        "top_1_percent_overlap_fraction"
    ].between(
        0,
        1,
    ).all()

    assert np.allclose(
        np.diag(
            correlation.to_numpy(
                dtype=float
            )
        ),
        1.0,
    )
