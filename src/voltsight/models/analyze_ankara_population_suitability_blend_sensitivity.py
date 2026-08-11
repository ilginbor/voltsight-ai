from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUITABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.csv"
)

DEMAND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_demand_sensitivity.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_suitability_blend_sensitivity.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_suitability_blend_metrics.csv"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_suitability_blend_sensitivity.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_suitability_blend_sensitivity_summary.md"
)


DEMAND_COLUMN = "demand_score_balanced_context"

BLEND_WEIGHTS = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
)

REQUIRED_SUITABILITY_COLUMNS = (
    "grid_id",
    "feasibility_score",
    "need_score",
    "suitability_score",
    "suitability_rank",
)


def weight_slug(
    demand_weight: float,
) -> str:
    """Create a deterministic compact label for one blend weight."""

    percentage = int(
        round(
            demand_weight
            * 100
        )
    )

    return f"demand_{percentage:02d}"


def validate_blend_weights() -> None:
    """Validate the diagnostic top-level demand weights."""

    if BLEND_WEIGHTS[
        0
    ] != 0.0:
        raise ValueError(
            "The first blend scenario must be the unchanged baseline."
        )

    if len(
        set(
            BLEND_WEIGHTS
        )
    ) != len(
        BLEND_WEIGHTS
    ):
        raise ValueError(
            "Duplicate blend weights were found."
        )

    for demand_weight in BLEND_WEIGHTS:
        if not 0 <= demand_weight < 1:
            raise ValueError(
                "Demand blend weights must be in [0, 1)."
            )


def validate_input_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Validate merged suitability and demand inputs."""

    required = {
        *REQUIRED_SUITABILITY_COLUMNS,
        DEMAND_COLUMN,
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Blend input columns are missing: "
            f"{sorted(missing)}"
        )

    result = dataframe.copy()

    result[
        "grid_id"
    ] = result[
        "grid_id"
    ].astype(str)

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate candidate grid IDs were found."
        )

    numeric_columns = (
        "feasibility_score",
        "need_score",
        "suitability_score",
        "suitability_rank",
        DEMAND_COLUMN,
    )

    for column in numeric_columns:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if (
            result[
                column
            ].isna().any()
            or not np.isfinite(
                values
            ).all()
        ):
            raise ValueError(
                f"Invalid values found in {column}."
            )

    for column in (
        "feasibility_score",
        "need_score",
        "suitability_score",
        DEMAND_COLUMN,
    ):
        if not result[
            column
        ].between(
            0,
            100,
        ).all():
            raise ValueError(
                f"{column} contains values outside 0-100."
            )

    if (
        result[
            "suitability_rank"
        ]
        < 1
    ).any():
        raise ValueError(
            "Suitability ranks must be positive."
        )

    return result


def load_inputs() -> pd.DataFrame:
    """Load current suitability and balanced-context demand scores."""

    validate_blend_weights()

    if not SUITABILITY_PATH.exists():
        raise FileNotFoundError(
            f"Suitability scores not found: {SUITABILITY_PATH}"
        )

    if not DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"Demand sensitivity output not found: {DEMAND_PATH}"
        )

    suitability = pd.read_csv(
        SUITABILITY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    demand = pd.read_csv(
        DEMAND_PATH,
        dtype={
            "grid_id": str,
        },
    )

    missing_suitability = (
        set(
            REQUIRED_SUITABILITY_COLUMNS
        )
        - set(
            suitability.columns
        )
    )

    if missing_suitability:
        raise ValueError(
            "Suitability columns are missing: "
            f"{sorted(missing_suitability)}"
        )

    if DEMAND_COLUMN not in demand.columns:
        raise ValueError(
            f"Demand score column is missing: {DEMAND_COLUMN}"
        )

    if suitability[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs in suitability scores."
        )

    if demand[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs in demand sensitivity output."
        )

    merged = suitability[
        list(
            REQUIRED_SUITABILITY_COLUMNS
        )
    ].merge(
        demand[
            [
                "grid_id",
                DEMAND_COLUMN,
            ]
        ],
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
            "Demand merge changed the candidate row count."
        )

    if merged[
        DEMAND_COLUMN
    ].isna().any():
        missing_rows = int(
            merged[
                DEMAND_COLUMN
            ].isna().sum()
        )

        raise ValueError(
            "Not every suitability candidate matched a demand score. "
            f"Missing rows: {missing_rows}."
        )

    return validate_input_frame(
        merged
    )


def create_blend_scores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create diagnostic convex blends of current suitability and demand.

    This is intentionally not a replacement for the canonical suitability
    formula. Both inputs are already on a 0-100 scale, so the blend is easy to
    interpret while testing how much ranking movement a small demand weight
    would create.
    """

    result = validate_input_frame(
        dataframe
    )

    baseline = result[
        "suitability_score"
    ].to_numpy(
        dtype=float
    )

    demand = result[
        DEMAND_COLUMN
    ].to_numpy(
        dtype=float
    )

    for demand_weight in BLEND_WEIGHTS:
        slug = weight_slug(
            demand_weight
        )

        score_column = (
            f"blend_score_{slug}"
        )

        rank_column = (
            f"blend_rank_{slug}"
        )

        result[
            score_column
        ] = (
            (
                1.0
                - demand_weight
            )
            * baseline
            + demand_weight
            * demand
        )

        result[
            rank_column
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

    baseline_slug = weight_slug(
        0.0
    )

    if not np.allclose(
        result[
            f"blend_score_{baseline_slug}"
        ].to_numpy(
            dtype=float
        ),
        baseline,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Zero-demand blend does not reproduce baseline suitability."
        )

    return result


def top_n_ids(
    dataframe: pd.DataFrame,
    score_column: str,
    *,
    count: int,
) -> set[str]:
    """Return the IDs of the highest-scoring deterministic top N rows."""

    if count <= 0:
        raise ValueError(
            "count must be positive."
        )

    return set(
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
        )[
            "grid_id"
        ]
        .astype(str)
    )


def top_fraction_ids(
    dataframe: pd.DataFrame,
    score_column: str,
    *,
    fraction: float,
) -> set[str]:
    """Return IDs in the highest-ranked score fraction."""

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

    return top_n_ids(
        dataframe,
        score_column,
        count=count,
    )


def calculate_blend_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Measure ranking movement and top-candidate trade-offs for each blend."""

    baseline_score_column = (
        f"blend_score_{weight_slug(0.0)}"
    )

    baseline_rank_column = (
        f"blend_rank_{weight_slug(0.0)}"
    )

    baseline_top_1 = top_fraction_ids(
        dataframe,
        baseline_score_column,
        fraction=0.01,
    )

    baseline_top_5 = top_fraction_ids(
        dataframe,
        baseline_score_column,
        fraction=0.05,
    )

    baseline_top_20 = top_n_ids(
        dataframe,
        baseline_score_column,
        count=20,
    )

    records: list[
        dict[str, float | int]
    ] = []

    for demand_weight in BLEND_WEIGHTS:
        slug = weight_slug(
            demand_weight
        )

        score_column = (
            f"blend_score_{slug}"
        )

        rank_column = (
            f"blend_rank_{slug}"
        )

        top_1 = top_fraction_ids(
            dataframe,
            score_column,
            fraction=0.01,
        )

        top_5 = top_fraction_ids(
            dataframe,
            score_column,
            fraction=0.05,
        )

        top_20 = top_n_ids(
            dataframe,
            score_column,
            count=20,
        )

        top_20_rows = dataframe[
            dataframe[
                "grid_id"
            ].astype(str).isin(
                top_20
            )
        ]

        rank_shift = (
            dataframe[
                rank_column
            ].to_numpy(
                dtype=float
            )
            - dataframe[
                baseline_rank_column
            ].to_numpy(
                dtype=float
            )
        )

        absolute_rank_shift = np.abs(
            rank_shift
        )

        records.append(
            {
                "demand_weight": demand_weight,
                "suitability_weight": (
                    1.0
                    - demand_weight
                ),
                "spearman_with_baseline": float(
                    dataframe[
                        [
                            score_column,
                            baseline_score_column,
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
                        top_1
                        & baseline_top_1
                    )
                ),
                "top_1_percent_overlap_fraction": float(
                    len(
                        top_1
                        & baseline_top_1
                    )
                    / len(
                        baseline_top_1
                    )
                ),
                "top_5_percent_overlap_count": int(
                    len(
                        top_5
                        & baseline_top_5
                    )
                ),
                "top_5_percent_overlap_fraction": float(
                    len(
                        top_5
                        & baseline_top_5
                    )
                    / len(
                        baseline_top_5
                    )
                ),
                "top_20_overlap_count": int(
                    len(
                        top_20
                        & baseline_top_20
                    )
                ),
                "top_20_overlap_fraction": float(
                    len(
                        top_20
                        & baseline_top_20
                    )
                    / len(
                        baseline_top_20
                    )
                ),
                "median_absolute_rank_shift": float(
                    np.median(
                        absolute_rank_shift
                    )
                ),
                "p95_absolute_rank_shift": float(
                    np.percentile(
                        absolute_rank_shift,
                        95,
                    )
                ),
                "top_20_median_original_suitability": float(
                    top_20_rows[
                        "suitability_score"
                    ].median()
                ),
                "top_20_minimum_original_suitability": float(
                    top_20_rows[
                        "suitability_score"
                    ].min()
                ),
                "top_20_median_feasibility": float(
                    top_20_rows[
                        "feasibility_score"
                    ].median()
                ),
                "top_20_minimum_feasibility": float(
                    top_20_rows[
                        "feasibility_score"
                    ].min()
                ),
                "top_20_median_need": float(
                    top_20_rows[
                        "need_score"
                    ].median()
                ),
                "top_20_minimum_need": float(
                    top_20_rows[
                        "need_score"
                    ].min()
                ),
                "top_20_median_demand": float(
                    top_20_rows[
                        DEMAND_COLUMN
                    ].median()
                ),
                "top_20_minimum_demand": float(
                    top_20_rows[
                        DEMAND_COLUMN
                    ].min()
                ),
                "top_20_demand_at_least_70": int(
                    (
                        top_20_rows[
                            DEMAND_COLUMN
                        ]
                        >= 70.0
                    ).sum()
                ),
                "top_20_need_at_least_50": int(
                    (
                        top_20_rows[
                            "need_score"
                        ]
                        >= 50.0
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def create_plot(
    metrics: pd.DataFrame,
) -> None:
    """Plot baseline top-set retention as demand weight increases."""

    PLOT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = (
        metrics[
            "demand_weight"
        ]
        * 100.0
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.plot(
        x,
        metrics[
            "top_1_percent_overlap_fraction"
        ],
        marker="o",
        label="Top 1% overlap",
    )

    axis.plot(
        x,
        metrics[
            "top_5_percent_overlap_fraction"
        ],
        marker="o",
        label="Top 5% overlap",
    )

    axis.plot(
        x,
        metrics[
            "top_20_overlap_fraction"
        ],
        marker="o",
        label="Top 20 overlap",
    )

    axis.set_xlabel(
        "Demand weight (%)"
    )

    axis.set_ylabel(
        "Overlap with current suitability ranking"
    )

    axis.set_ylim(
        0,
        1.05,
    )

    axis.set_title(
        "Ankara Population-Suitability Blend Sensitivity"
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
    metrics: pd.DataFrame,
) -> None:
    """Document top-level demand blend sensitivity."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_lines = [
        "| Demand weight | Suitability weight | Spearman vs baseline | Top 1% overlap | Top 5% overlap | Top-20 overlap | Median abs rank shift | P95 abs rank shift | Top-20 median suitability | Top-20 median feasibility | Top-20 median need | Top-20 median demand | Demand >=70 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in metrics.itertuples(
        index=False
    ):
        table_lines.append(
            "| "
            f"{row.demand_weight:.0%} | "
            f"{row.suitability_weight:.0%} | "
            f"{row.spearman_with_baseline:.4f} | "
            f"{row.top_1_percent_overlap_fraction:.2%} | "
            f"{row.top_5_percent_overlap_fraction:.2%} | "
            f"{row.top_20_overlap_fraction:.2%} | "
            f"{row.median_absolute_rank_shift:,.0f} | "
            f"{row.p95_absolute_rank_shift:,.0f} | "
            f"{row.top_20_median_original_suitability:.4f} | "
            f"{row.top_20_median_feasibility:.4f} | "
            f"{row.top_20_median_need:.4f} | "
            f"{row.top_20_median_demand:.4f} | "
            f"{row.top_20_demand_at_least_70}/20 |"
        )

    summary = f"""# Ankara Population-Suitability Blend Sensitivity

## Purpose

This diagnostic tests how much the current Ankara suitability ranking would
move if a small residential-demand contribution were added at the top level.

The canonical suitability formula and canonical shortlist are not modified by
this script.

## Inputs

- Candidate rows: {len(dataframe):,}
- Current suitability score: existing geometric feasibility/need score
- Demand score: `balanced_context`
- Demand score median: {dataframe[DEMAND_COLUMN].median():.4f}
- Demand score maximum: {dataframe[DEMAND_COLUMN].max():.4f}

## Diagnostic Blend

For demand weight `w`:

`diagnostic_score = (1 - w) * current_suitability + w * demand`

Weights tested:

- 0% demand / 100% current suitability
- 5% demand / 95% current suitability
- 10% demand / 90% current suitability
- 15% demand / 85% current suitability
- 20% demand / 80% current suitability

This convex blend is deliberately simple and interpretable. It is a
sensitivity device, not a claim that the final production score must be an
arithmetic blend.

## Results

{chr(10).join(table_lines)}

## Interpretation Policy

The zero-demand row must reproduce the current suitability ranking exactly.

Spearman correlation measures province-wide ranking stability. Top-1%, top-5%,
and top-20 overlap focus on decision-relevant high-ranked cells.

Rank-shift diagnostics show whether a small demand contribution only reorders
the top of the list or materially reshuffles the wider candidate universe.

The top-20 feasibility, need, and demand summaries expose the central trade-off:
population demand is positively associated with road/parking feasibility but
negatively associated with the existing charging-gap need score.

A demand weight should not be selected merely because it increases residential
demand among top candidates. The selected weight should preserve meaningful
infrastructure-need coverage and avoid turning a province-wide infrastructure
planning score into an urban-population ranking.

Population remains a modeled residential-demand proxy. It does not directly
measure traffic, employment, commuting, retail activity, tourism, EV ownership,
or distribution-grid capacity.

This analysis is descriptive decision-model sensitivity, not ML validation and
not causal inference.

## Outputs

- `data/processed/{CANDIDATE_OUTPUT_PATH.name}`
- `data/processed/{METRICS_OUTPUT_PATH.name}`
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
    metrics: pd.DataFrame,
) -> None:
    """Save candidate-level blends and scenario metrics."""

    CANDIDATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "grid_id",
        "feasibility_score",
        "need_score",
        "suitability_score",
        "suitability_rank",
        DEMAND_COLUMN,
        *(
            column
            for demand_weight in BLEND_WEIGHTS
            for column in (
                f"blend_score_{weight_slug(demand_weight)}",
                f"blend_rank_{weight_slug(demand_weight)}",
            )
        ),
    ]

    dataframe[
        output_columns
    ].to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def print_results(
    metrics: pd.DataFrame,
) -> None:
    """Print the main demand-blend sensitivity table."""

    columns = [
        "demand_weight",
        "spearman_with_baseline",
        "top_1_percent_overlap_fraction",
        "top_5_percent_overlap_fraction",
        "top_20_overlap_fraction",
        "median_absolute_rank_shift",
        "p95_absolute_rank_shift",
        "top_20_median_need",
        "top_20_median_demand",
        "top_20_demand_at_least_70",
    ]

    print(
        metrics[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )


def main() -> None:
    """Run population-suitability blend sensitivity."""

    print("=" * 70)
    print(
        "VoltSight - Ankara Population-Suitability Blend Sensitivity"
    )
    print("=" * 70)

    dataframe = load_inputs()

    dataframe = create_blend_scores(
        dataframe
    )

    metrics = calculate_blend_metrics(
        dataframe
    )

    save_outputs(
        dataframe,
        metrics,
    )

    create_plot(
        metrics
    )

    create_summary(
        dataframe,
        metrics,
    )

    print_results(
        metrics
    )

    print("=" * 70)
    print(
        "Ankara population-suitability blend sensitivity "
        "completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
