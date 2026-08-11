from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_candidate_dataset.csv"
)

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_model_oof_predictions.csv"
)

SUITABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.csv"
)

SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_ml_support.csv"
)

SHORTLIST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_shortlist_ml_support.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_ml_support_metrics.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_candidate_ml_support_summary.md"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_suitability_ml_support.png"
)


TARGET_COLUMN = "has_existing_charging_station"

MODEL_SCORE_COLUMNS = (
    "logistic_regression_score",
    "random_forest_score",
    "hist_gradient_boosting_score",
)

MODEL_PERCENTILE_COLUMNS = (
    "logistic_regression_percentile",
    "random_forest_percentile",
    "hist_gradient_boosting_percentile",
)

SUITABILITY_COLUMNS = (
    "grid_id",
    "suitability_score",
    "suitability_rank",
    "suitability_percentile",
    "priority_band",
    "feasibility_score",
    "need_score",
)


def percentile_rank_score(
    values: pd.Series,
) -> pd.Series:
    """Convert higher-is-better scores to 0-100 candidate percentiles."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            "Percentile input contains missing or non-numeric values."
        )

    array = numeric.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            "Percentile input contains non-finite values."
        )

    return (
        numeric.rank(
            method="average",
            ascending=True,
            pct=True,
        )
        * 100.0
    )


def validate_candidate_ids(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Validate canonical candidate identifiers."""

    if "grid_id" not in candidates.columns:
        raise ValueError(
            "Canonical candidate dataset is missing grid_id."
        )

    result = candidates[
        [
            "grid_id",
        ]
    ].copy()

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        "grid_id"
    ].eq("").any():
        raise ValueError(
            "Canonical candidate dataset contains empty grid IDs."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical candidate dataset contains duplicate grid IDs."
        )

    if result.empty:
        raise ValueError(
            "Canonical candidate dataset is empty."
        )

    return result


def validate_oof(
    oof: pd.DataFrame,
) -> pd.DataFrame:
    """Validate canonical out-of-fold predictions."""

    required = {
        "grid_id",
        TARGET_COLUMN,
        "cv_fold",
        *MODEL_SCORE_COLUMNS,
    }

    missing = (
        required
        - set(
            oof.columns
        )
    )

    if missing:
        raise ValueError(
            "Canonical OOF columns are missing: "
            f"{sorted(missing)}"
        )

    result = oof[
        [
            "grid_id",
            TARGET_COLUMN,
            "cv_fold",
            *MODEL_SCORE_COLUMNS,
        ]
    ].copy()

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical OOF data contains duplicate grid IDs."
        )

    result[
        "cv_fold"
    ] = pd.to_numeric(
        result[
            "cv_fold"
        ],
        errors="raise",
    ).astype(
        int
    )

    if (
        result[
            "cv_fold"
        ]
        < 0
    ).any():
        raise ValueError(
            "Canonical OOF fold IDs must be non-negative."
        )

    result[
        TARGET_COLUMN
    ] = pd.to_numeric(
        result[
            TARGET_COLUMN
        ],
        errors="raise",
    ).astype(
        int
    )

    if not set(
        result[
            TARGET_COLUMN
        ].unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Canonical OOF target must contain only 0/1."
        )

    for column in MODEL_SCORE_COLUMNS:
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
                f"Invalid OOF model scores found in {column}."
            )

        if not result[
            column
        ].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"OOF model scores in {column} are outside 0-1."
            )

    return result


def validate_suitability(
    suitability: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the candidate suitability table required for diagnostics."""

    missing = (
        set(
            SUITABILITY_COLUMNS
        )
        - set(
            suitability.columns
        )
    )

    if missing:
        raise ValueError(
            "Suitability columns are missing: "
            f"{sorted(missing)}"
        )

    result = suitability[
        list(
            SUITABILITY_COLUMNS
        )
    ].copy()

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Suitability data contains duplicate grid IDs."
        )

    numeric_columns = (
        "suitability_score",
        "suitability_rank",
        "suitability_percentile",
        "feasibility_score",
        "need_score",
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
                f"Invalid suitability values found in {column}."
            )

    return result


def create_candidate_support(
    candidates: pd.DataFrame,
    oof: pd.DataFrame,
    suitability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create candidate-level ML support from spatial out-of-fold predictions.

    No full-data model is fitted here. Each canonical candidate receives the
    OOF score generated while its spatial fold was held out.
    """

    candidate_ids = validate_candidate_ids(
        candidates
    )

    oof = validate_oof(
        oof
    )

    suitability = validate_suitability(
        suitability
    )

    negative_oof = oof.loc[
        oof[
            TARGET_COLUMN
        ]
        == 0,
        [
            "grid_id",
            "cv_fold",
            *MODEL_SCORE_COLUMNS,
        ],
    ].copy()

    candidate_id_set = set(
        candidate_ids[
            "grid_id"
        ]
    )

    negative_oof_id_set = set(
        negative_oof[
            "grid_id"
        ]
    )

    if candidate_id_set != negative_oof_id_set:
        missing_oof = (
            candidate_id_set
            - negative_oof_id_set
        )

        unexpected_oof = (
            negative_oof_id_set
            - candidate_id_set
        )

        raise ValueError(
            "Canonical candidates must exactly match negative OOF rows. "
            f"Missing OOF: {len(missing_oof)}, "
            f"unexpected negative OOF: {len(unexpected_oof)}."
        )

    suitability_id_set = set(
        suitability[
            "grid_id"
        ]
    )

    if candidate_id_set != suitability_id_set:
        missing_suitability = (
            candidate_id_set
            - suitability_id_set
        )

        unexpected_suitability = (
            suitability_id_set
            - candidate_id_set
        )

        raise ValueError(
            "Canonical candidates must exactly match suitability rows. "
            f"Missing suitability: {len(missing_suitability)}, "
            f"unexpected suitability: {len(unexpected_suitability)}."
        )

    result = (
        candidate_ids.merge(
            negative_oof,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            suitability,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )

    percentile_pairs = zip(
        MODEL_SCORE_COLUMNS,
        MODEL_PERCENTILE_COLUMNS,
        strict=True,
    )

    for (
        score_column,
        percentile_column,
    ) in percentile_pairs:
        result[
            percentile_column
        ] = (
            result.groupby(
                "cv_fold",
                sort=True,
            )[
                score_column
            ]
            .transform(
                percentile_rank_score
            )
        )

    percentile_matrix = result[
        list(
            MODEL_PERCENTILE_COLUMNS
        )
    ].to_numpy(
        dtype=float
    )

    result[
        "ml_consensus_percentile"
    ] = np.median(
        percentile_matrix,
        axis=1,
    )

    result[
        "ml_min_percentile"
    ] = np.min(
        percentile_matrix,
        axis=1,
    )

    result[
        "ml_max_percentile"
    ] = np.max(
        percentile_matrix,
        axis=1,
    )

    result[
        "ml_model_spread"
    ] = (
        result[
            "ml_max_percentile"
        ]
        - result[
            "ml_min_percentile"
        ]
    )

    result[
        "models_top_20pct_count"
    ] = (
        percentile_matrix
        >= 80.0
    ).sum(
        axis=1
    )

    result[
        "models_top_10pct_count"
    ] = (
        percentile_matrix
        >= 90.0
    ).sum(
        axis=1
    )

    result[
        "at_least_two_models_top_20pct"
    ] = (
        result[
            "models_top_20pct_count"
        ]
        >= 2
    )

    result[
        "all_models_top_20pct"
    ] = (
        result[
            "models_top_20pct_count"
        ]
        == len(
            MODEL_PERCENTILE_COLUMNS
        )
    )

    result[
        "ml_consensus_rank"
    ] = (
        result[
            "ml_consensus_percentile"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            int
        )
    )

    result[
        "ml_minus_suitability_percentile"
    ] = (
        result[
            "ml_consensus_percentile"
        ]
        - result[
            "suitability_percentile"
        ]
    )

    result = result.sort_values(
        [
            "suitability_rank",
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

    if len(
        result
    ) != len(
        candidate_ids
    ):
        raise ValueError(
            "Candidate support row count changed unexpectedly."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Candidate support contains duplicate grid IDs."
        )

    return result


def create_shortlist_support(
    shortlist: pd.DataFrame,
    candidate_support: pd.DataFrame,
) -> pd.DataFrame:
    """Attach OOF ML diagnostics to the saved canonical shortlist."""

    if "grid_id" not in shortlist.columns:
        raise ValueError(
            "Canonical shortlist is missing grid_id."
        )

    shortlist_ids = shortlist[
        [
            "grid_id",
        ]
    ].copy()

    shortlist_ids[
        "grid_id"
    ] = (
        shortlist_ids[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if shortlist_ids[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical shortlist contains duplicate grid IDs."
        )

    shortlist_ids[
        "shortlist_order"
    ] = np.arange(
        1,
        len(
            shortlist_ids
        )
        + 1,
        dtype=int,
    )

    result = shortlist_ids.merge(
        candidate_support,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if result[
        "ml_consensus_percentile"
    ].isna().any():
        raise ValueError(
            "Not every shortlist row matched candidate ML support."
        )

    return result.sort_values(
        "shortlist_order",
        kind="stable",
    ).reset_index(
        drop=True
    )


def top_fraction_overlap(
    dataframe: pd.DataFrame,
    *,
    fraction: float,
) -> dict[
    str,
    float | int,
]:
    """Measure overlap between top suitability and top ML-consensus candidates."""

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

    suitability_ids = set(
        dataframe.nsmallest(
            count,
            [
                "suitability_rank",
            ],
        )[
            "grid_id"
        ]
    )

    ml_ids = set(
        dataframe.nsmallest(
            count,
            [
                "ml_consensus_rank",
            ],
        )[
            "grid_id"
        ]
    )

    overlap_count = len(
        suitability_ids
        & ml_ids
    )

    return {
        "fraction": float(
            fraction
        ),
        "selected_count": int(
            count
        ),
        "overlap_count": int(
            overlap_count
        ),
        "overlap_fraction": float(
            overlap_count
            / count
        ),
    }


def calculate_metrics(
    candidate_support: pd.DataFrame,
    shortlist_support: pd.DataFrame,
) -> pd.DataFrame:
    """Create global and shortlist-level agreement diagnostics."""

    correlations = (
        candidate_support[
            [
                "suitability_score",
                "ml_consensus_percentile",
            ]
        ]
        .corr(
            method="spearman"
        )
    )

    spearman = float(
        correlations.loc[
            "suitability_score",
            "ml_consensus_percentile",
        ]
    )

    top_one = top_fraction_overlap(
        candidate_support,
        fraction=0.01,
    )

    top_five = top_fraction_overlap(
        candidate_support,
        fraction=0.05,
    )

    top_ten = top_fraction_overlap(
        candidate_support,
        fraction=0.10,
    )

    record = {
        "candidate_count": len(
            candidate_support
        ),
        "shortlist_count": len(
            shortlist_support
        ),
        "spearman_suitability_vs_ml_consensus": (
            spearman
        ),
        "top_1pct_selected_count": (
            top_one[
                "selected_count"
            ]
        ),
        "top_1pct_overlap_count": (
            top_one[
                "overlap_count"
            ]
        ),
        "top_1pct_overlap_fraction": (
            top_one[
                "overlap_fraction"
            ]
        ),
        "top_5pct_selected_count": (
            top_five[
                "selected_count"
            ]
        ),
        "top_5pct_overlap_count": (
            top_five[
                "overlap_count"
            ]
        ),
        "top_5pct_overlap_fraction": (
            top_five[
                "overlap_fraction"
            ]
        ),
        "top_10pct_selected_count": (
            top_ten[
                "selected_count"
            ]
        ),
        "top_10pct_overlap_count": (
            top_ten[
                "overlap_count"
            ]
        ),
        "top_10pct_overlap_fraction": (
            top_ten[
                "overlap_fraction"
            ]
        ),
        "shortlist_median_ml_consensus_percentile": float(
            shortlist_support[
                "ml_consensus_percentile"
            ].median()
        ),
        "shortlist_min_ml_consensus_percentile": float(
            shortlist_support[
                "ml_consensus_percentile"
            ].min()
        ),
        "shortlist_max_ml_consensus_percentile": float(
            shortlist_support[
                "ml_consensus_percentile"
            ].max()
        ),
        "shortlist_median_model_spread": float(
            shortlist_support[
                "ml_model_spread"
            ].median()
        ),
        "shortlist_at_least_two_models_top_20pct_count": int(
            shortlist_support[
                "at_least_two_models_top_20pct"
            ].sum()
        ),
        "shortlist_all_models_top_20pct_count": int(
            shortlist_support[
                "all_models_top_20pct"
            ].sum()
        ),
        "shortlist_at_least_two_models_top_10pct_count": int(
            (
                shortlist_support[
                    "models_top_10pct_count"
                ]
                >= 2
            ).sum()
        ),
    }

    return pd.DataFrame(
        [
            record,
        ]
    )


def create_plot(
    candidate_support: pd.DataFrame,
    shortlist_support: pd.DataFrame,
) -> None:
    """Plot suitability against spatial-OOF ML consensus."""

    PLOT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    axis.scatter(
        candidate_support[
            "suitability_score"
        ],
        candidate_support[
            "ml_consensus_percentile"
        ],
        s=8,
        alpha=0.12,
        label="All candidates",
    )

    axis.scatter(
        shortlist_support[
            "suitability_score"
        ],
        shortlist_support[
            "ml_consensus_percentile"
        ],
        s=55,
        marker="x",
        label="Canonical shortlist",
    )

    axis.set_xlabel(
        "Explainable suitability score"
    )

    axis.set_ylabel(
        "Median ML spatial-OOF percentile"
    )

    axis.set_title(
        "Ankara Suitability vs Canonical Activity15 ML Support"
    )

    axis.set_ylim(
        0,
        100,
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


def save_outputs(
    candidate_support: pd.DataFrame,
    shortlist_support: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Save candidate and shortlist ML-support diagnostics."""

    CANDIDATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_support.to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    shortlist_support.to_csv(
        SHORTLIST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    candidate_support: pd.DataFrame,
    shortlist_support: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Document candidate-level agreement between suitability and OOF ML."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric = metrics.iloc[
        0
    ]

    shortlist_lines = [
        "| # | Grid | Suitability rank | Suitability | LR pct | RF pct | HGB pct | ML consensus | Min model pct | Spread | Models top 20% |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in shortlist_support.itertuples(
        index=False
    ):
        shortlist_lines.append(
            "| "
            f"{int(row.shortlist_order)} | "
            f"`{row.grid_id}` | "
            f"{int(row.suitability_rank):,} | "
            f"{row.suitability_score:.4f} | "
            f"{row.logistic_regression_percentile:.2f} | "
            f"{row.random_forest_percentile:.2f} | "
            f"{row.hist_gradient_boosting_percentile:.2f} | "
            f"{row.ml_consensus_percentile:.2f} | "
            f"{row.ml_min_percentile:.2f} | "
            f"{row.ml_model_spread:.2f} | "
            f"{int(row.models_top_20pct_count)}/3 |"
        )

    strongest = (
        shortlist_support.sort_values(
            [
                "ml_consensus_percentile",
                "suitability_rank",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .head(
            5
        )
    )

    weakest = (
        shortlist_support.sort_values(
            [
                "ml_consensus_percentile",
                "suitability_rank",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )
        .head(
            5
        )
    )

    strongest_lines = "\n".join(
        (
            f"- `{row.grid_id}`: suitability rank "
            f"{int(row.suitability_rank):,}, ML consensus "
            f"{row.ml_consensus_percentile:.2f}"
        )
        for row in strongest.itertuples(
            index=False
        )
    )

    weakest_lines = "\n".join(
        (
            f"- `{row.grid_id}`: suitability rank "
            f"{int(row.suitability_rank):,}, ML consensus "
            f"{row.ml_consensus_percentile:.2f}"
        )
        for row in weakest.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Candidate ML Support Diagnostic

## Purpose

This diagnostic compares the canonical explainable suitability ranking with
Canonical Activity15 machine-learning evidence without blending the two into a
single decision score.

The ML side uses fold-normalized spatial OOF ranks rather than globally ranking
raw OOF scores from different fold-specific estimators.

Suitability remains the primary site-selection layer.

## Why OOF Scores Are Used

The canonical candidate set is exactly the target-negative subset of the
existing-station ML universe.

Each model score used here is therefore the spatial out-of-fold score generated
for that grid cell while its 5-km spatial fold was held out.

No full-data model is fitted to generate the candidate-support values in this
diagnostic.

This reduces in-sample optimism, but it is still internal spatial validation and
not independent external or temporal validation.

## Candidate ML Percentiles

Raw model scores come from different fold-specific estimators, so their absolute
scales are not assumed to be directly comparable across folds.

For each model, scores are converted to 0-100 candidate percentiles **within the
held-out spatial fold that produced them**. This fold-normalized ranking reduces
artifacts from fold-specific score scale or calibration differences.

Higher percentile means stronger within-fold agreement with the historical
spatial pattern learned by that model.

The cross-model ML consensus is the median of:

- Logistic Regression candidate percentile
- Random Forest candidate percentile
- HistGradientBoosting candidate percentile

The cross-model median is therefore a consensus of fold-normalized ranking
positions. Percentiles are ranking diagnostics, not calibrated probabilities.

## Province-Wide Agreement

- Candidate count: {int(metric['candidate_count']):,}
- Spearman correlation, suitability vs ML consensus: {metric['spearman_suitability_vs_ml_consensus']:.4f}
- Top 1% overlap: {int(metric['top_1pct_overlap_count']):,}/{int(metric['top_1pct_selected_count']):,} ({metric['top_1pct_overlap_fraction']:.2%})
- Top 5% overlap: {int(metric['top_5pct_overlap_count']):,}/{int(metric['top_5pct_selected_count']):,} ({metric['top_5pct_overlap_fraction']:.2%})
- Top 10% overlap: {int(metric['top_10pct_overlap_count']):,}/{int(metric['top_10pct_selected_count']):,} ({metric['top_10pct_overlap_fraction']:.2%})

## Canonical 20-Site Shortlist

- Median ML consensus percentile: {metric['shortlist_median_ml_consensus_percentile']:.2f}
- Minimum ML consensus percentile: {metric['shortlist_min_ml_consensus_percentile']:.2f}
- Maximum ML consensus percentile: {metric['shortlist_max_ml_consensus_percentile']:.2f}
- Median cross-model percentile spread: {metric['shortlist_median_model_spread']:.2f}
- At least two models in candidate top 20%: {int(metric['shortlist_at_least_two_models_top_20pct_count'])}/20
- All three models in candidate top 20%: {int(metric['shortlist_all_models_top_20pct_count'])}/20
- At least two models in candidate top 10%: {int(metric['shortlist_at_least_two_models_top_10pct_count'])}/20

{chr(10).join(shortlist_lines)}

## Strongest ML-Supported Shortlist Cells

{strongest_lines}

## Lowest ML-Consensus Shortlist Cells

{weakest_lines}

A lower ML percentile does not invalidate a suitability recommendation.

Suitability explicitly rewards infrastructure need and feasibility, while the
ML models learn patterns associated with the limited existing-station
distribution. A high-suitability / lower-ML-agreement candidate may therefore
represent a gap-oriented recommendation that differs from historical placement
patterns.

## Interpretation Policy

This diagnostic must not be used as a new blended canonical score.

Fold normalization makes the OOF ranking scales more comparable, but it also
means each candidate is interpreted relative to other candidates in the same
held-out spatial fold before cross-model consensus is formed.

It provides a second axis of evidence:

- suitability: explainable forward-looking decision support
- ML percentile: agreement with historical mapped station-placement patterns

The ML signal is limited by only 46 positive station cells, incomplete open-data
coverage, residual spatial dependence, and the absence of independent external
validation.

Mapped OSM activity is a proxy rather than direct EV demand, traffic, trips,
employment, or commercial turnover.

## Outputs

- `data/processed/{CANDIDATE_OUTPUT_PATH.name}`
- `data/processed/{SHORTLIST_OUTPUT_PATH.name}`
- `data/processed/{METRICS_OUTPUT_PATH.name}`
- `docs/{PLOT_OUTPUT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load canonical candidate, OOF, suitability, and shortlist tables."""

    for path in (
        CANDIDATE_PATH,
        OOF_PATH,
        SUITABILITY_PATH,
        SHORTLIST_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        dtype={
            "grid_id": str,
        },
    )

    oof = pd.read_csv(
        OOF_PATH,
        dtype={
            "grid_id": str,
        },
    )

    suitability = pd.read_csv(
        SUITABILITY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    shortlist = pd.read_csv(
        SHORTLIST_PATH,
        dtype={
            "grid_id": str,
        },
    )

    return (
        candidates,
        oof,
        suitability,
        shortlist,
    )


def print_results(
    metrics: pd.DataFrame,
    shortlist_support: pd.DataFrame,
) -> None:
    """Print key candidate-support diagnostics."""

    metric = metrics.iloc[
        0
    ]

    print(
        "-"
        * 70
    )

    print(
        "Candidate count:",
        f"{int(metric['candidate_count']):,}",
    )

    print(
        "Suitability vs ML consensus Spearman:",
        f"{metric['spearman_suitability_vs_ml_consensus']:.6f}",
    )

    print(
        "Top 1% overlap:",
        f"{metric['top_1pct_overlap_fraction']:.2%}",
    )

    print(
        "Top 5% overlap:",
        f"{metric['top_5pct_overlap_fraction']:.2%}",
    )

    print(
        "Top 10% overlap:",
        f"{metric['top_10pct_overlap_fraction']:.2%}",
    )

    print()

    print(
        "Shortlist median ML consensus percentile:",
        f"{metric['shortlist_median_ml_consensus_percentile']:.2f}",
    )

    print(
        "Shortlist with >=2 models in candidate top 20%:",
        f"{int(metric['shortlist_at_least_two_models_top_20pct_count'])}/20",
    )

    print(
        "Shortlist with all 3 models in candidate top 20%:",
        f"{int(metric['shortlist_all_models_top_20pct_count'])}/20",
    )

    print()

    print(
        shortlist_support[
            [
                "shortlist_order",
                "grid_id",
                "suitability_rank",
                "suitability_score",
                "ml_consensus_percentile",
                "ml_min_percentile",
                "ml_model_spread",
                "models_top_20pct_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def main() -> None:
    """Create OOF ML support diagnostics for Ankara candidate decisions."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Candidate ML Support Diagnostic"
    )

    print(
        "="
        * 70
    )

    (
        candidates,
        oof,
        suitability,
        shortlist,
    ) = load_inputs()

    candidate_support = create_candidate_support(
        candidates,
        oof,
        suitability,
    )

    shortlist_support = create_shortlist_support(
        shortlist,
        candidate_support,
    )

    metrics = calculate_metrics(
        candidate_support,
        shortlist_support,
    )

    save_outputs(
        candidate_support,
        shortlist_support,
        metrics,
    )

    create_plot(
        candidate_support,
        shortlist_support,
    )

    create_summary(
        candidate_support,
        shortlist_support,
        metrics,
    )

    print_results(
        metrics,
        shortlist_support,
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara candidate ML support diagnostic completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
