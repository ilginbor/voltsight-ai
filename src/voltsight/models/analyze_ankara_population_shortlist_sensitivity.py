from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from voltsight.models.create_ankara_diverse_candidate_shortlist import (
    DESIRED_CANDIDATE_COUNT,
    MINIMUM_FEASIBILITY_SCORE,
    MINIMUM_NEED_SCORE,
    MINIMUM_SPACING_METERS,
    MINIMUM_SUITABILITY_SCORE,
    add_spacing_diagnostics,
    apply_eligibility_filters,
    load_spatial_scores,
    select_spatially_diverse_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEMAND_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_demand_sensitivity.csv"
)

CURRENT_SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.csv"
)

CURRENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_shortlist_sensitivity_current.csv"
)

ADJUSTED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_shortlist_sensitivity_adjusted.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_population_shortlist_sensitivity_metrics.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_shortlist_sensitivity_summary.md"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_population_shortlist_sensitivity.png"
)


DEMAND_COLUMN = "demand_score_balanced_context"
DEMAND_WEIGHT = 0.05
SUITABILITY_WEIGHT = 1.0 - DEMAND_WEIGHT

ADJUSTED_SCORE_COLUMN = "population_adjusted_score"
ADJUSTED_RANK_COLUMN = "population_adjusted_rank"


def validate_configuration() -> None:
    """Validate the fixed shortlist sensitivity configuration."""

    if not np.isclose(
        DEMAND_WEIGHT
        + SUITABILITY_WEIGHT,
        1.0,
    ):
        raise ValueError(
            "Suitability and demand weights must sum to one."
        )

    if not 0 < DEMAND_WEIGHT < 1:
        raise ValueError(
            "Demand weight must be strictly between zero and one."
        )

    if DESIRED_CANDIDATE_COUNT != 20:
        raise ValueError(
            "This diagnostic expects the canonical 20-candidate shortlist."
        )

    if not np.isclose(
        MINIMUM_SPACING_METERS,
        25_000.0,
    ):
        raise ValueError(
            "This diagnostic expects the canonical 25-km spacing rule."
        )


def validate_demand_frame(
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the balanced-context demand score table."""

    required = {
        "grid_id",
        DEMAND_COLUMN,
    }

    missing = (
        required
        - set(
            demand.columns
        )
    )

    if missing:
        raise ValueError(
            "Demand sensitivity columns are missing: "
            f"{sorted(missing)}"
        )

    result = demand[
        [
            "grid_id",
            DEMAND_COLUMN,
        ]
    ].copy()

    result[
        "grid_id"
    ] = result[
        "grid_id"
    ].astype(str)

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate demand grid IDs were found."
        )

    result[
        DEMAND_COLUMN
    ] = pd.to_numeric(
        result[
            DEMAND_COLUMN
        ],
        errors="coerce",
    )

    values = result[
        DEMAND_COLUMN
    ].to_numpy(
        dtype=float
    )

    if (
        result[
            DEMAND_COLUMN
        ].isna().any()
        or not np.isfinite(
            values
        ).all()
    ):
        raise ValueError(
            "Demand scores contain missing or non-finite values."
        )

    if not result[
        DEMAND_COLUMN
    ].between(
        0,
        100,
    ).all():
        raise ValueError(
            "Demand scores must remain inside 0-100."
        )

    return result


def attach_demand_scores(
    scores: gpd.GeoDataFrame,
    demand: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach the balanced-context demand score to all scored candidates."""

    if scores[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate suitability grid IDs were found."
        )

    demand = validate_demand_frame(
        demand
    )

    merged = scores.merge(
        demand,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        scores
    ):
        raise ValueError(
            "Demand merge changed the suitability row count."
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
            "Not every suitability row matched a demand score. "
            f"Missing rows: {missing_rows}."
        )

    return gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=scores.crs,
    )


def add_adjusted_score_and_rank(
    scores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create the fixed 5% demand-adjusted diagnostic score and rank."""

    required = {
        "grid_id",
        "suitability_score",
        DEMAND_COLUMN,
    }

    missing = (
        required
        - set(
            scores.columns
        )
    )

    if missing:
        raise ValueError(
            "Adjusted-score inputs are missing: "
            f"{sorted(missing)}"
        )

    result = scores.copy()

    result[
        ADJUSTED_SCORE_COLUMN
    ] = (
        SUITABILITY_WEIGHT
        * pd.to_numeric(
            result[
                "suitability_score"
            ],
            errors="raise",
        ).astype(float)
        + DEMAND_WEIGHT
        * pd.to_numeric(
            result[
                DEMAND_COLUMN
            ],
            errors="raise",
        ).astype(float)
    )

    result[
        ADJUSTED_RANK_COLUMN
    ] = (
        result[
            ADJUSTED_SCORE_COLUMN
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    if not result[
        ADJUSTED_SCORE_COLUMN
    ].between(
        0,
        100,
    ).all():
        raise ValueError(
            "Adjusted scores must remain inside 0-100."
        )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=scores.crs,
    )


def apply_original_quality_filters_for_adjusted_ranking(
    scores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Preserve the canonical quality gates but rank by the adjusted score.

    The original suitability >= 60, feasibility >= 60, and need >= 50
    thresholds stay unchanged. This isolates the effect of ranking by a 5%
    demand adjustment rather than changing who is considered eligible.
    """

    required = {
        "grid_id",
        "suitability_score",
        "feasibility_score",
        "need_score",
        ADJUSTED_SCORE_COLUMN,
        ADJUSTED_RANK_COLUMN,
    }

    missing = (
        required
        - set(
            scores.columns
        )
    )

    if missing:
        raise ValueError(
            "Adjusted shortlist columns are missing: "
            f"{sorted(missing)}"
        )

    eligible = scores.loc[
        (
            scores[
                "suitability_score"
            ]
            >= MINIMUM_SUITABILITY_SCORE
        )
        & (
            scores[
                "feasibility_score"
            ]
            >= MINIMUM_FEASIBILITY_SCORE
        )
        & (
            scores[
                "need_score"
            ]
            >= MINIMUM_NEED_SCORE
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            ADJUSTED_RANK_COLUMN,
            "grid_id",
        ],
        ascending=[
            True,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if eligible.empty:
        raise ValueError(
            "No candidates satisfy the unchanged shortlist quality gates."
        )

    return gpd.GeoDataFrame(
        eligible,
        geometry="geometry",
        crs=scores.crs,
    )


def create_current_shortlist(
    scores: gpd.GeoDataFrame,
) -> tuple[
    gpd.GeoDataFrame,
    int,
]:
    """Reproduce the current canonical shortlist from current scores."""

    eligible = apply_eligibility_filters(
        scores
    )

    shortlist = (
        select_spatially_diverse_candidates(
            eligible
        )
    )

    shortlist = add_spacing_diagnostics(
        shortlist
    )

    return (
        shortlist,
        len(
            eligible
        ),
    )


def create_adjusted_shortlist(
    scores: gpd.GeoDataFrame,
) -> tuple[
    gpd.GeoDataFrame,
    int,
]:
    """Create the 5% demand-adjusted shortlist under unchanged quality gates."""

    eligible = (
        apply_original_quality_filters_for_adjusted_ranking(
            scores
        )
    )

    shortlist = (
        select_spatially_diverse_candidates(
            eligible
        )
    )

    shortlist = add_spacing_diagnostics(
        shortlist
    )

    return (
        shortlist,
        len(
            eligible
        ),
    )


def validate_saved_canonical_shortlist(
    current_shortlist: gpd.GeoDataFrame,
) -> None:
    """
    Confirm recomputation matches the previously saved canonical shortlist.

    The check is optional because generated data outputs may not be present in
    every fresh clone. When present, order must match exactly.
    """

    if not CURRENT_SHORTLIST_PATH.exists():
        return

    saved = pd.read_csv(
        CURRENT_SHORTLIST_PATH,
        dtype={
            "grid_id": str,
        },
    )

    if "grid_id" not in saved.columns:
        raise ValueError(
            "Saved canonical shortlist is missing grid_id."
        )

    expected_ids = (
        saved[
            "grid_id"
        ]
        .astype(str)
        .tolist()
    )

    actual_ids = (
        current_shortlist[
            "grid_id"
        ]
        .astype(str)
        .tolist()
    )

    if expected_ids != actual_ids:
        raise ValueError(
            "Recomputed current shortlist does not match the saved "
            "canonical shortlist order."
        )


def pairwise_distance_metrics(
    shortlist: gpd.GeoDataFrame,
) -> dict[str, float]:
    """Summarize spatial spread using representative-point distances."""

    points = list(
        shortlist.geometry
        .representative_point()
    )

    if len(
        points
    ) < 2:
        raise ValueError(
            "At least two shortlist points are required."
        )

    distances: list[
        float
    ] = []

    for left_index in range(
        len(
            points
        )
    ):
        for right_index in range(
            left_index + 1,
            len(
                points
            ),
        ):
            distances.append(
                float(
                    points[
                        left_index
                    ].distance(
                        points[
                            right_index
                        ]
                    )
                )
            )

    values = np.asarray(
        distances,
        dtype=float,
    )

    return {
        "mean_pairwise_distance_m": float(
            values.mean()
        ),
        "median_pairwise_distance_m": float(
            np.median(
                values
            )
        ),
        "maximum_pairwise_distance_m": float(
            values.max()
        ),
    }


def summarize_shortlist(
    shortlist: gpd.GeoDataFrame,
    *,
    scenario: str,
) -> dict[
    str,
    float | int | str,
]:
    """Create decision-relevant diagnostics for one shortlist."""

    spread = pairwise_distance_metrics(
        shortlist
    )

    return {
        "scenario": scenario,
        "selected_count": len(
            shortlist
        ),
        "minimum_spacing_m": float(
            shortlist[
                "nearest_selected_candidate_m"
            ].min()
        ),
        "median_nearest_selected_distance_m": float(
            shortlist[
                "nearest_selected_candidate_m"
            ].median()
        ),
        **spread,
        "median_original_suitability": float(
            shortlist[
                "suitability_score"
            ].median()
        ),
        "minimum_original_suitability": float(
            shortlist[
                "suitability_score"
            ].min()
        ),
        "median_feasibility": float(
            shortlist[
                "feasibility_score"
            ].median()
        ),
        "minimum_feasibility": float(
            shortlist[
                "feasibility_score"
            ].min()
        ),
        "median_need": float(
            shortlist[
                "need_score"
            ].median()
        ),
        "minimum_need": float(
            shortlist[
                "need_score"
            ].min()
        ),
        "median_demand": float(
            shortlist[
                DEMAND_COLUMN
            ].median()
        ),
        "minimum_demand": float(
            shortlist[
                DEMAND_COLUMN
            ].min()
        ),
        "demand_at_least_70_count": int(
            (
                shortlist[
                    DEMAND_COLUMN
                ]
                >= 70.0
            ).sum()
        ),
        "worst_original_suitability_rank": int(
            shortlist[
                "suitability_rank"
            ].max()
        ),
        "median_original_suitability_rank": float(
            shortlist[
                "suitability_rank"
            ].median()
        ),
        "median_adjusted_score": float(
            shortlist[
                ADJUSTED_SCORE_COLUMN
            ].median()
        ),
    }


def calculate_comparison_metrics(
    current_shortlist: gpd.GeoDataFrame,
    adjusted_shortlist: gpd.GeoDataFrame,
    *,
    current_eligible_count: int,
    adjusted_eligible_count: int,
) -> tuple[
    pd.DataFrame,
    set[str],
    set[str],
    set[str],
]:
    """Compare current and adjusted spatially diverse shortlists."""

    current_ids = set(
        current_shortlist[
            "grid_id"
        ].astype(str)
    )

    adjusted_ids = set(
        adjusted_shortlist[
            "grid_id"
        ].astype(str)
    )

    common_ids = (
        current_ids
        & adjusted_ids
    )

    removed_ids = (
        current_ids
        - adjusted_ids
    )

    added_ids = (
        adjusted_ids
        - current_ids
    )

    current_record = summarize_shortlist(
        current_shortlist,
        scenario="current",
    )

    current_record[
        "eligible_count"
    ] = current_eligible_count

    adjusted_record = summarize_shortlist(
        adjusted_shortlist,
        scenario="demand_adjusted_5pct",
    )

    adjusted_record[
        "eligible_count"
    ] = adjusted_eligible_count

    metrics = pd.DataFrame(
        [
            current_record,
            adjusted_record,
        ]
    )

    metrics[
        "overlap_count_with_current"
    ] = [
        len(
            current_ids
        ),
        len(
            common_ids
        ),
    ]

    metrics[
        "overlap_fraction_with_current"
    ] = [
        1.0,
        (
            len(
                common_ids
            )
            / len(
                current_ids
            )
        ),
    ]

    return (
        metrics,
        common_ids,
        removed_ids,
        added_ids,
    )


def validate_shortlist_quality(
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Validate the fixed quality gates and 25-km spacing."""

    if len(
        shortlist
    ) != DESIRED_CANDIDATE_COUNT:
        raise ValueError(
            "Unexpected shortlist size."
        )

    if (
        shortlist[
            "suitability_score"
        ]
        < MINIMUM_SUITABILITY_SCORE
    ).any():
        raise ValueError(
            "A selected row violates the original suitability threshold."
        )

    if (
        shortlist[
            "feasibility_score"
        ]
        < MINIMUM_FEASIBILITY_SCORE
    ).any():
        raise ValueError(
            "A selected row violates the feasibility threshold."
        )

    if (
        shortlist[
            "need_score"
        ]
        < MINIMUM_NEED_SCORE
    ).any():
        raise ValueError(
            "A selected row violates the need threshold."
        )

    if (
        shortlist[
            "nearest_selected_candidate_m"
        ].min()
        + 1e-6
        < MINIMUM_SPACING_METERS
    ):
        raise ValueError(
            "A selected row violates the 25-km spacing threshold."
        )


def create_plot(
    current_shortlist: gpd.GeoDataFrame,
    adjusted_shortlist: gpd.GeoDataFrame,
) -> None:
    """Plot current and adjusted representative points for spatial comparison."""

    PLOT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_points = (
        current_shortlist.geometry
        .representative_point()
    )

    adjusted_points = (
        adjusted_shortlist.geometry
        .representative_point()
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            9,
        )
    )

    axis.scatter(
        current_points.x,
        current_points.y,
        marker="o",
        label="Current shortlist",
    )

    axis.scatter(
        adjusted_points.x,
        adjusted_points.y,
        marker="x",
        label="5% demand-adjusted shortlist",
    )

    axis.set_xlabel(
        "Easting (m, EPSG:32636)"
    )

    axis.set_ylabel(
        "Northing (m, EPSG:32636)"
    )

    axis.set_title(
        "Ankara Shortlist Sensitivity to 5% Population Demand"
    )

    axis.legend()

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    figure.savefig(
        PLOT_OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_outputs(
    current_shortlist: gpd.GeoDataFrame,
    adjusted_shortlist: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Save shortlist sensitivity tables."""

    CURRENT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_shortlist.drop(
        columns="geometry"
    ).to_csv(
        CURRENT_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    adjusted_shortlist.drop(
        columns="geometry"
    ).to_csv(
        ADJUSTED_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    metrics: pd.DataFrame,
    *,
    common_ids: set[str],
    removed_ids: set[str],
    added_ids: set[str],
) -> None:
    """Write the shortlist sensitivity summary."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current = metrics.loc[
        metrics[
            "scenario"
        ]
        == "current"
    ].iloc[
        0
    ]

    adjusted = metrics.loc[
        metrics[
            "scenario"
        ]
        == "demand_adjusted_5pct"
    ].iloc[
        0
    ]

    removed_lines = (
        "\n".join(
            f"- `{grid_id}`"
            for grid_id in sorted(
                removed_ids
            )
        )
        if removed_ids
        else "- None"
    )

    added_lines = (
        "\n".join(
            f"- `{grid_id}`"
            for grid_id in sorted(
                added_ids
            )
        )
        if added_ids
        else "- None"
    )

    summary = f"""# Ankara Population Shortlist Sensitivity

## Purpose

This diagnostic compares the current canonical 20-site Ankara shortlist with
a shortlist ranked by the previously selected 5% residential-demand adjustment.

The canonical suitability files and canonical shortlist files are not modified.

## Fixed Selection Rules

Both scenarios use the same existing quality gates:

- Original suitability >= {MINIMUM_SUITABILITY_SCORE:.0f}
- Feasibility >= {MINIMUM_FEASIBILITY_SCORE:.0f}
- Need >= {MINIMUM_NEED_SCORE:.0f}
- Minimum representative-point spacing >= {MINIMUM_SPACING_METERS / 1000:.0f} km
- Desired shortlist size: {DESIRED_CANDIDATE_COUNT}

The current scenario orders eligible candidates by the original suitability
rank.

The adjusted scenario keeps the same original quality gates but orders eligible
candidates by:

`0.95 * current_suitability + 0.05 * balanced_population_demand`

This isolates ranking sensitivity from eligibility-policy changes.

## Shortlist Overlap

- Common selected cells: {len(common_ids)}/{DESIRED_CANDIDATE_COUNT}
- Overlap fraction: {len(common_ids) / DESIRED_CANDIDATE_COUNT:.2%}
- Removed current cells: {len(removed_ids)}
- Added adjusted cells: {len(added_ids)}

### Removed From Current Shortlist

{removed_lines}

### Added By 5% Demand Adjustment

{added_lines}

## Scenario Metrics

| Metric | Current | 5% demand-adjusted |
| --- | ---: | ---: |
| Eligible candidates | {int(current['eligible_count']):,} | {int(adjusted['eligible_count']):,} |
| Minimum spacing (km) | {current['minimum_spacing_m'] / 1000:.3f} | {adjusted['minimum_spacing_m'] / 1000:.3f} |
| Median nearest-selected distance (km) | {current['median_nearest_selected_distance_m'] / 1000:.3f} | {adjusted['median_nearest_selected_distance_m'] / 1000:.3f} |
| Mean pairwise distance (km) | {current['mean_pairwise_distance_m'] / 1000:.3f} | {adjusted['mean_pairwise_distance_m'] / 1000:.3f} |
| Maximum pairwise distance (km) | {current['maximum_pairwise_distance_m'] / 1000:.3f} | {adjusted['maximum_pairwise_distance_m'] / 1000:.3f} |
| Median original suitability | {current['median_original_suitability']:.4f} | {adjusted['median_original_suitability']:.4f} |
| Minimum original suitability | {current['minimum_original_suitability']:.4f} | {adjusted['minimum_original_suitability']:.4f} |
| Median feasibility | {current['median_feasibility']:.4f} | {adjusted['median_feasibility']:.4f} |
| Minimum feasibility | {current['minimum_feasibility']:.4f} | {adjusted['minimum_feasibility']:.4f} |
| Median need | {current['median_need']:.4f} | {adjusted['median_need']:.4f} |
| Minimum need | {current['minimum_need']:.4f} | {adjusted['minimum_need']:.4f} |
| Median demand | {current['median_demand']:.4f} | {adjusted['median_demand']:.4f} |
| Minimum demand | {current['minimum_demand']:.4f} | {adjusted['minimum_demand']:.4f} |
| Demand >= 70 | {int(current['demand_at_least_70_count'])}/20 | {int(adjusted['demand_at_least_70_count'])}/20 |
| Worst original suitability rank | {int(current['worst_original_suitability_rank']):,} | {int(adjusted['worst_original_suitability_rank']):,} |

## Interpretation Policy

This is a shortlist-level sensitivity test, not a new canonical scoring model.

A favorable 5% adjustment should increase residential-demand representation
without materially degrading original suitability, feasibility, infrastructure
need, or province-wide spatial spread.

Because the original quality thresholds are held fixed, any shortlist change
comes from ranking and the downstream greedy 25-km spacing interaction rather
than from relaxing candidate quality gates.

Population remains a modeled residential-demand proxy. It does not directly
measure traffic, employment, commuting, retail activity, tourism, EV ownership,
or electricity-grid capacity.

The final decision on whether to adopt a population adjustment should consider
this shortlist diagnostic together with the earlier ML incremental-value and
weight-sensitivity analyses.

## Outputs

- `data/processed/{CURRENT_OUTPUT_PATH.name}`
- `data/processed/{ADJUSTED_OUTPUT_PATH.name}`
- `data/processed/{METRICS_OUTPUT_PATH.name}`
- `docs/{PLOT_OUTPUT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    metrics: pd.DataFrame,
    *,
    common_ids: set[str],
    removed_ids: set[str],
    added_ids: set[str],
) -> None:
    """Print the main shortlist comparison."""

    columns = [
        "scenario",
        "eligible_count",
        "minimum_spacing_m",
        "median_nearest_selected_distance_m",
        "mean_pairwise_distance_m",
        "median_original_suitability",
        "median_feasibility",
        "median_need",
        "median_demand",
        "demand_at_least_70_count",
        "worst_original_suitability_rank",
        "overlap_fraction_with_current",
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

    print()

    print(
        "Common shortlist IDs: "
        f"{len(common_ids)}/{DESIRED_CANDIDATE_COUNT}"
    )

    print(
        "Removed IDs: "
        + (
            ", ".join(
                sorted(
                    removed_ids
                )
            )
            if removed_ids
            else "none"
        )
    )

    print(
        "Added IDs: "
        + (
            ", ".join(
                sorted(
                    added_ids
                )
            )
            if added_ids
            else "none"
        )
    )


def main() -> None:
    """Run the Ankara 5% population-demand shortlist sensitivity."""

    print("=" * 70)
    print(
        "VoltSight - Ankara Population Shortlist Sensitivity"
    )
    print("=" * 70)

    validate_configuration()

    if not DEMAND_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Demand sensitivity output not found: {DEMAND_INPUT_PATH}"
        )

    scores = load_spatial_scores()

    demand = pd.read_csv(
        DEMAND_INPUT_PATH,
        dtype={
            "grid_id": str,
        },
    )

    scores = attach_demand_scores(
        scores,
        demand,
    )

    scores = add_adjusted_score_and_rank(
        scores
    )

    (
        current_shortlist,
        current_eligible_count,
    ) = create_current_shortlist(
        scores
    )

    validate_saved_canonical_shortlist(
        current_shortlist
    )

    (
        adjusted_shortlist,
        adjusted_eligible_count,
    ) = create_adjusted_shortlist(
        scores
    )

    validate_shortlist_quality(
        current_shortlist
    )

    validate_shortlist_quality(
        adjusted_shortlist
    )

    (
        metrics,
        common_ids,
        removed_ids,
        added_ids,
    ) = calculate_comparison_metrics(
        current_shortlist,
        adjusted_shortlist,
        current_eligible_count=current_eligible_count,
        adjusted_eligible_count=adjusted_eligible_count,
    )

    save_outputs(
        current_shortlist,
        adjusted_shortlist,
        metrics,
    )

    create_plot(
        current_shortlist,
        adjusted_shortlist,
    )

    create_summary(
        metrics,
        common_ids=common_ids,
        removed_ids=removed_ids,
        added_ids=added_ids,
    )

    print_results(
        metrics,
        common_ids=common_ids,
        removed_ids=removed_ids,
        added_ids=added_ids,
    )

    print("=" * 70)
    print(
        "Ankara population shortlist sensitivity completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
