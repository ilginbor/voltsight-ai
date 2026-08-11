from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voltsight.models.create_ankara_suitability_scores import (
    positive_percentile_score,
    weighted_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUITABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.csv"
)

POPULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_population_features.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_demand_sensitivity.csv"
)

CORRELATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_demand_correlations.csv"
)

SCENARIO_METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_demand_scenario_metrics.csv"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_demand_sensitivity.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_demand_sensitivity_summary.md"
)


POPULATION_COLUMNS = (
    "population_count",
    "population_density_per_km2",
    "population_within_1000m",
    "population_within_2000m",
)

COMPONENT_SCORE_COLUMNS = (
    "local_population_score",
    "population_1km_score",
    "population_2km_score",
)

DEMAND_SCENARIOS: dict[
    str,
    dict[str, float],
] = {
    "local_only": {
        "local_population_score": 1.00,
        "population_1km_score": 0.00,
        "population_2km_score": 0.00,
    },
    "near_context": {
        "local_population_score": 0.30,
        "population_1km_score": 0.40,
        "population_2km_score": 0.30,
    },
    "balanced_context": {
        "local_population_score": 0.20,
        "population_1km_score": 0.35,
        "population_2km_score": 0.45,
    },
    "broad_context": {
        "local_population_score": 0.10,
        "population_1km_score": 0.30,
        "population_2km_score": 0.60,
    },
}

SCENARIO_LABELS = {
    "local_only": "Local only",
    "near_context": "30% local / 40% 1 km / 30% 2 km",
    "balanced_context": "20% local / 35% 1 km / 45% 2 km",
    "broad_context": "10% local / 30% 1 km / 60% 2 km",
}

REFERENCE_SCENARIO = "balanced_context"


def validate_scenario_weights() -> None:
    """Ensure every demand sensitivity scenario is normalized."""

    for scenario_name, weights in DEMAND_SCENARIOS.items():
        if set(
            weights
        ) != set(
            COMPONENT_SCORE_COLUMNS
        ):
            raise ValueError(
                f"{scenario_name} does not define all demand components."
            )

        if not np.isclose(
            sum(
                weights.values()
            ),
            1.0,
        ):
            raise ValueError(
                f"{scenario_name} weights do not sum to 1."
            )

        if any(
            weight < 0
            for weight in weights.values()
        ):
            raise ValueError(
                f"{scenario_name} contains a negative weight."
            )


def validate_population_frame(
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Validate population inputs before joining candidate cells."""

    required = {
        "grid_id",
        *POPULATION_COLUMNS,
    }

    missing = required - set(
        population.columns
    )

    if missing:
        raise ValueError(
            "Population columns are missing: "
            f"{sorted(missing)}"
        )

    population = population[
        [
            "grid_id",
            *POPULATION_COLUMNS,
        ]
    ].copy()

    population[
        "grid_id"
    ] = population[
        "grid_id"
    ].astype(str)

    if population[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate population grid IDs were found."
        )

    for column in POPULATION_COLUMNS:
        population[
            column
        ] = pd.to_numeric(
            population[
                column
            ],
            errors="coerce",
        )

        values = population[
            column
        ].to_numpy(
            dtype=float
        )

        if (
            population[
                column
            ].isna().any()
            or not np.isfinite(
                values
            ).all()
        ):
            raise ValueError(
                f"Invalid values found in {column}."
            )

        if (
            values < -1e-8
        ).any():
            raise ValueError(
                f"Negative values found in {column}."
            )

    expected_density = (
        population[
            "population_count"
        ].to_numpy(
            dtype=float
        )
        / 0.25
    )

    if not np.allclose(
        population[
            "population_density_per_km2"
        ].to_numpy(
            dtype=float
        ),
        expected_density,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError(
            "Population density is not the expected deterministic "
            "transform of population_count."
        )

    return population


def load_inputs() -> pd.DataFrame:
    """Load current suitability scores and attach population features."""

    validate_scenario_weights()

    if not SUITABILITY_PATH.exists():
        raise FileNotFoundError(
            f"Suitability scores not found: {SUITABILITY_PATH}"
        )

    if not POPULATION_PATH.exists():
        raise FileNotFoundError(
            f"Population features not found: {POPULATION_PATH}"
        )

    suitability = pd.read_csv(
        SUITABILITY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    required_suitability = {
        "grid_id",
        "feasibility_score",
        "need_score",
        "suitability_score",
        "suitability_rank",
    }

    missing = (
        required_suitability
        - set(
            suitability.columns
        )
    )

    if missing:
        raise ValueError(
            "Suitability columns are missing: "
            f"{sorted(missing)}"
        )

    if suitability[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate suitability grid IDs were found."
        )

    population = pd.read_csv(
        POPULATION_PATH,
        dtype={
            "grid_id": str,
        },
    )

    population = validate_population_frame(
        population
    )

    merged = suitability.merge(
        population,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        suitability
    ):
        raise ValueError(
            "Population merge changed candidate row count."
        )

    if merged[
        list(
            POPULATION_COLUMNS
        )
    ].isna().any().any():
        missing_rows = int(
            merged[
                "population_count"
            ].isna().sum()
        )

        raise ValueError(
            "Not every suitability candidate matched population data. "
            f"Missing rows: {missing_rows}."
        )

    return merged


def create_population_component_scores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create zero-preserving percentile scores for population demand."""

    result = dataframe.copy()

    result[
        "local_population_score"
    ] = positive_percentile_score(
        result[
            "population_count"
        ]
    )

    result[
        "population_1km_score"
    ] = positive_percentile_score(
        result[
            "population_within_1000m"
        ]
    )

    result[
        "population_2km_score"
    ] = positive_percentile_score(
        result[
            "population_within_2000m"
        ]
    )

    for column in COMPONENT_SCORE_COLUMNS:
        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite demand component score in {column}."
            )

        if (
            (values < 0)
            | (values > 100)
        ).any():
            raise ValueError(
                f"Demand component score outside 0-100 in {column}."
            )

    return result


def add_demand_scenarios(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the candidate demand score under each weight scenario."""

    result = create_population_component_scores(
        dataframe
    )

    for scenario_name, weights in DEMAND_SCENARIOS.items():
        score_column = (
            f"demand_score_{scenario_name}"
        )

        result[
            score_column
        ] = weighted_score(
            result,
            weights,
        )

        result[
            f"demand_rank_{scenario_name}"
        ] = (
            result[
                score_column
            ]
            .rank(
                method="min",
                ascending=False,
            )
            .astype(int)
        )

    return result


def top_fraction_ids(
    dataframe: pd.DataFrame,
    score_column: str,
    *,
    fraction: float,
) -> set[str]:
    """Return IDs in the top requested score fraction."""

    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction must be in (0, 1]."
        )

    count = max(
        1,
        int(
            np.ceil(
                len(
                    dataframe
                )
                * fraction
            )
        ),
    )

    top = (
        dataframe.sort_values(
            [
                score_column,
                "grid_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .head(
            count
        )
    )

    return set(
        top[
            "grid_id"
        ].astype(str)
    )


def calculate_scenario_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each demand weighting with the current suitability ranking."""

    suitability_top_1 = top_fraction_ids(
        dataframe,
        "suitability_score",
        fraction=0.01,
    )

    suitability_top_5 = top_fraction_ids(
        dataframe,
        "suitability_score",
        fraction=0.05,
    )

    suitability_top_20 = set(
        dataframe.sort_values(
            [
                "suitability_rank",
                "grid_id",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )
        .head(
            20
        )[
            "grid_id"
        ]
        .astype(str)
    )

    records: list[
        dict[str, float | int | str]
    ] = []

    for scenario_name in DEMAND_SCENARIOS:
        score_column = (
            f"demand_score_{scenario_name}"
        )

        demand_top_1 = top_fraction_ids(
            dataframe,
            score_column,
            fraction=0.01,
        )

        demand_top_5 = top_fraction_ids(
            dataframe,
            score_column,
            fraction=0.05,
        )

        top_20_rows = dataframe[
            dataframe[
                "grid_id"
            ].astype(str).isin(
                suitability_top_20
            )
        ]

        records.append(
            {
                "scenario": scenario_name,
                "scenario_label": SCENARIO_LABELS[
                    scenario_name
                ],
                "median_demand_score": float(
                    dataframe[
                        score_column
                    ].median()
                ),
                "mean_demand_score": float(
                    dataframe[
                        score_column
                    ].mean()
                ),
                "maximum_demand_score": float(
                    dataframe[
                        score_column
                    ].max()
                ),
                "spearman_with_suitability": float(
                    dataframe[
                        [
                            score_column,
                            "suitability_score",
                        ]
                    ].corr(
                        method="spearman"
                    ).iloc[
                        0,
                        1,
                    ]
                ),
                "spearman_with_feasibility": float(
                    dataframe[
                        [
                            score_column,
                            "feasibility_score",
                        ]
                    ].corr(
                        method="spearman"
                    ).iloc[
                        0,
                        1,
                    ]
                ),
                "spearman_with_need": float(
                    dataframe[
                        [
                            score_column,
                            "need_score",
                        ]
                    ].corr(
                        method="spearman"
                    ).iloc[
                        0,
                        1,
                    ]
                ),
                "top_1_percent_overlap_count": int(
                    len(
                        demand_top_1
                        & suitability_top_1
                    )
                ),
                "top_1_percent_overlap_fraction": float(
                    len(
                        demand_top_1
                        & suitability_top_1
                    )
                    / len(
                        suitability_top_1
                    )
                ),
                "top_5_percent_overlap_count": int(
                    len(
                        demand_top_5
                        & suitability_top_5
                    )
                ),
                "top_5_percent_overlap_fraction": float(
                    len(
                        demand_top_5
                        & suitability_top_5
                    )
                    / len(
                        suitability_top_5
                    )
                ),
                "current_top_20_median_demand_score": float(
                    top_20_rows[
                        score_column
                    ].median()
                ),
                "current_top_20_minimum_demand_score": float(
                    top_20_rows[
                        score_column
                    ].min()
                ),
                "current_top_20_demand_at_least_70": int(
                    (
                        top_20_rows[
                            score_column
                        ]
                        >= 70.0
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def create_correlation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a Spearman correlation matrix for demand and current pillars."""

    columns = [
        *COMPONENT_SCORE_COLUMNS,
        *(
            f"demand_score_{scenario_name}"
            for scenario_name in DEMAND_SCENARIOS
        ),
        "feasibility_score",
        "need_score",
        "suitability_score",
    ]

    return dataframe[
        columns
    ].corr(
        method="spearman"
    )


def create_plot(
    scenario_metrics: pd.DataFrame,
) -> None:
    """Visualize correlations and top-1% overlap by demand scenario."""

    PLOT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = scenario_metrics.copy()

    x = np.arange(
        len(
            plot_data
        )
    )

    width = 0.25

    figure, axis = plt.subplots(
        figsize=(
            12,
            6.5,
        )
    )

    axis.bar(
        x - width,
        plot_data[
            "spearman_with_feasibility"
        ],
        width=width,
        label="Spearman vs feasibility",
    )

    axis.bar(
        x,
        plot_data[
            "spearman_with_need"
        ],
        width=width,
        label="Spearman vs need",
    )

    axis.bar(
        x + width,
        plot_data[
            "spearman_with_suitability"
        ],
        width=width,
        label="Spearman vs suitability",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        plot_data[
            "scenario"
        ],
        rotation=15,
        ha="right",
    )

    axis.set_ylabel(
        "Spearman correlation"
    )

    axis.set_title(
        "Ankara Population Demand Weight Sensitivity"
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        PLOT_OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_summary(
    dataframe: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Document demand-score sensitivity without changing suitability."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario_lines = [
        "| Scenario | Weights | Median demand | Spearman vs feasibility | Spearman vs need | Spearman vs suitability | Top 1% overlap | Top 5% overlap | Current top-20 median demand | Top-20 demand >=70 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in scenario_metrics.itertuples(
        index=False
    ):
        scenario_lines.append(
            "| "
            f"{row.scenario} | "
            f"{SCENARIO_LABELS[row.scenario]} | "
            f"{row.median_demand_score:.4f} | "
            f"{row.spearman_with_feasibility:.4f} | "
            f"{row.spearman_with_need:.4f} | "
            f"{row.spearman_with_suitability:.4f} | "
            f"{row.top_1_percent_overlap_fraction:.2%} | "
            f"{row.top_5_percent_overlap_fraction:.2%} | "
            f"{row.current_top_20_median_demand_score:.4f} | "
            f"{row.current_top_20_demand_at_least_70}/20 |"
        )

    demand_score_columns = [
        f"demand_score_{scenario_name}"
        for scenario_name in DEMAND_SCENARIOS
    ]

    scenario_only_correlation = correlation.loc[
        demand_score_columns,
        demand_score_columns,
    ]

    off_diagonal = (
        scenario_only_correlation.to_numpy(
            dtype=float
        )[
            np.triu_indices(
                len(
                    demand_score_columns
                ),
                k=1,
            )
        ]
    )

    minimum_scenario_correlation = float(
        off_diagonal.min()
    )

    component_correlation = correlation.loc[
        list(
            COMPONENT_SCORE_COLUMNS
        ),
        list(
            COMPONENT_SCORE_COLUMNS
        ),
    ]

    component_off_diagonal = (
        component_correlation.to_numpy(
            dtype=float
        )[
            np.triu_indices(
                len(
                    COMPONENT_SCORE_COLUMNS
                ),
                k=1,
            )
        ]
    )

    reference = scenario_metrics.loc[
        scenario_metrics[
            "scenario"
        ]
        == REFERENCE_SCENARIO
    ].iloc[
        0
    ]

    summary = f"""# Ankara Population Demand Sensitivity

## Purpose

This diagnostic studies how a separate residential-population demand pillar
behaves before it is allowed to change the canonical suitability formula.

The existing feasibility, need, suitability scores, ranks, and shortlist are
not modified by this script.

## Candidate Data

- Candidate rows: {len(dataframe):,}
- Population-positive local cells: {int((dataframe['population_count'] > 0).sum()):,}
- Local population median: {dataframe['population_count'].median():,.2f}
- 1-km population median: {dataframe['population_within_1000m'].median():,.2f}
- 2-km population median: {dataframe['population_within_2000m'].median():,.2f}

## Zero-Preserving Demand Components

Three population variables are converted to positive-only percentile scores:

- `local_population_score`
- `population_1km_score`
- `population_2km_score`

True zero values remain zero. Positive population values are ranked only
against other positive values.

`population_density_per_km2` is not scored separately because it is a
deterministic scale transform of `population_count` on the fixed 500-m grid.

## Weight Scenarios

- `local_only`: 100% local
- `near_context`: 30% local, 40% within 1 km, 30% within 2 km
- `balanced_context`: 20% local, 35% within 1 km, 45% within 2 km
- `broad_context`: 10% local, 30% within 1 km, 60% within 2 km

The scenarios are sensitivity diagnostics, not fitted or optimized weights.

## Scenario Results

{chr(10).join(scenario_lines)}

## Correlation Diagnostics

- Minimum pairwise Spearman correlation among the four demand scenarios:
  {minimum_scenario_correlation:.4f}
- Minimum pairwise Spearman correlation among local/1-km/2-km demand
  components: {float(component_off_diagonal.min()):.4f}
- Maximum pairwise Spearman correlation among local/1-km/2-km demand
  components: {float(component_off_diagonal.max()):.4f}

For the reference `balanced_context` scenario:

- Spearman with feasibility: {reference['spearman_with_feasibility']:.4f}
- Spearman with need: {reference['spearman_with_need']:.4f}
- Spearman with current suitability: {reference['spearman_with_suitability']:.4f}
- Current suitability top-1% overlap: {reference['top_1_percent_overlap_fraction']:.2%}
- Current suitability top-5% overlap: {reference['top_5_percent_overlap_fraction']:.2%}
- Median demand score among the current suitability top 20:
  {reference['current_top_20_median_demand_score']:.4f}
- Current suitability top-20 cells with demand >= 70:
  {int(reference['current_top_20_demand_at_least_70'])}/20

## Interpretation Policy

High correlation between demand scenarios means exact local-versus-neighborhood
weights have limited effect on province-wide demand ordering. Lower correlation
means the weight choice materially changes which cells are described as
high-demand.

Correlation with feasibility or current suitability indicates overlap with the
existing road/parking decision layer. Correlation with need indicates overlap
with the charging-gap layer. These are descriptive associations, not causal
effects.

Top-fraction overlap describes whether current suitability and population
demand prioritize the same cells. Low overlap does not automatically mean one
ranking is wrong; it can indicate that demand contributes a distinct decision
dimension.

This diagnostic intentionally does not create a new final suitability score.
A top-level demand weight should only be selected after reviewing these
redundancy and ranking-stability results.

Population remains a modeled residential-demand proxy and does not directly
measure traffic, employment, commuting, retail activity, tourism, EV
ownership, or distribution-grid capacity.

## Outputs

- `data/processed/{CANDIDATE_OUTPUT_PATH.name}`
- `data/processed/{CORRELATION_OUTPUT_PATH.name}`
- `data/processed/{SCENARIO_METRICS_OUTPUT_PATH.name}`
- `docs/{PLOT_OUTPUT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def save_outputs(
    dataframe: pd.DataFrame,
    correlation: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
) -> None:
    """Save demand sensitivity tables."""

    CANDIDATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "grid_id",
        "population_count",
        "population_density_per_km2",
        "population_within_1000m",
        "population_within_2000m",
        *COMPONENT_SCORE_COLUMNS,
        *(
            column
            for scenario_name in DEMAND_SCENARIOS
            for column in (
                f"demand_score_{scenario_name}",
                f"demand_rank_{scenario_name}",
            )
        ),
        "feasibility_score",
        "need_score",
        "suitability_score",
        "suitability_rank",
    ]

    dataframe[
        output_columns
    ].to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    correlation.to_csv(
        CORRELATION_OUTPUT_PATH,
        index=True,
        encoding="utf-8",
    )

    scenario_metrics.to_csv(
        SCENARIO_METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def print_results(
    scenario_metrics: pd.DataFrame,
) -> None:
    """Print key demand-weight sensitivity metrics."""

    columns = [
        "scenario",
        "median_demand_score",
        "spearman_with_feasibility",
        "spearman_with_need",
        "spearman_with_suitability",
        "top_1_percent_overlap_fraction",
        "top_5_percent_overlap_fraction",
        "current_top_20_median_demand_score",
        "current_top_20_demand_at_least_70",
    ]

    print(
        scenario_metrics[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )


def main() -> None:
    """Run Ankara population demand weight sensitivity."""

    print("=" * 70)
    print(
        "VoltSight - Ankara Population Demand Sensitivity"
    )
    print("=" * 70)

    dataframe = load_inputs()

    dataframe = add_demand_scenarios(
        dataframe
    )

    scenario_metrics = (
        calculate_scenario_metrics(
            dataframe
        )
    )

    correlation = (
        create_correlation_table(
            dataframe
        )
    )

    save_outputs(
        dataframe,
        correlation,
        scenario_metrics,
    )

    create_plot(
        scenario_metrics
    )

    create_summary(
        dataframe,
        scenario_metrics,
        correlation,
    )

    print_results(
        scenario_metrics
    )

    print("=" * 70)
    print(
        "Ankara population demand sensitivity completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
