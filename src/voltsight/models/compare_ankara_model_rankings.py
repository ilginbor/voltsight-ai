from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGISTIC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_logistic_baseline_oof_predictions.csv"
)

RANDOM_FOREST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_oof_predictions.csv"
)

GRADIENT_BOOSTING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_gradient_boosting_baseline_oof_predictions.csv"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_model_ranking_comparison.csv"
)

COMBINED_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_model_rank_ensemble_oof.csv"
)

CORRELATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_model_rank_correlation.csv"
)

PR_CURVE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_model_rank_comparison_pr_curve.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_model_rank_comparison_summary.md"
)

TARGET_COLUMN = "has_existing_charging_station"

BASE_MODELS = {
    "Logistic Regression": (
        "logistic_score",
        "logistic_rank",
    ),
    "Random Forest": (
        "random_forest_score",
        "random_forest_rank",
    ),
    "Gradient Boosting": (
        "gradient_boosting_score",
        "gradient_boosting_rank",
    ),
}

ENSEMBLE_COLUMN = "rank_ensemble_score"


def read_oof(
    path: Path,
    score_column: str,
) -> pd.DataFrame:
    """Read and validate one model OOF file."""

    if not path.exists():
        raise FileNotFoundError(
            f"OOF file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "grid_id": str,
            "spatial_block_id": str,
        },
    )

    required_columns = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
        score_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            f"Duplicate grid IDs found in {path.name}."
        )

    dataframe[
        TARGET_COLUMN
    ] = (
        pd.to_numeric(
            dataframe[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .astype(int)
    )

    dataframe[
        "cv_fold"
    ] = (
        pd.to_numeric(
            dataframe[
                "cv_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    dataframe[
        score_column
    ] = pd.to_numeric(
        dataframe[
            score_column
        ],
        errors="coerce",
    )

    if dataframe[
        score_column
    ].isna().any():
        raise ValueError(
            f"Missing scores found in {path.name}."
        )

    scores = dataframe[
        score_column
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            f"Non-finite scores found in {path.name}."
        )

    if not dataframe[
        score_column
    ].between(
        0,
        1,
    ).all():
        raise ValueError(
            f"Scores outside 0-1 found in {path.name}."
        )

    return dataframe


def load_and_align_oof() -> pd.DataFrame:
    """Load three OOF datasets and verify identical evaluation rows."""

    logistic = read_oof(
        LOGISTIC_PATH,
        "logistic_score",
    )

    random_forest = read_oof(
        RANDOM_FOREST_PATH,
        "random_forest_score",
    )

    gradient_boosting = read_oof(
        GRADIENT_BOOSTING_PATH,
        "gradient_boosting_score",
    )

    combined = logistic[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
            "logistic_score",
        ]
    ].copy()

    for name, dataframe, score_column in [
        (
            "random_forest",
            random_forest,
            "random_forest_score",
        ),
        (
            "gradient_boosting",
            gradient_boosting,
            "gradient_boosting_score",
        ),
    ]:
        prepared = dataframe[
            [
                "grid_id",
                TARGET_COLUMN,
                "spatial_block_id",
                "cv_fold",
                score_column,
            ]
        ].rename(
            columns={
                TARGET_COLUMN: (
                    f"{name}_target"
                ),
                "spatial_block_id": (
                    f"{name}_spatial_block_id"
                ),
                "cv_fold": (
                    f"{name}_cv_fold"
                ),
            }
        )

        combined = combined.merge(
            prepared,
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )

        if not np.array_equal(
            combined[
                TARGET_COLUMN
            ].to_numpy(),
            combined[
                f"{name}_target"
            ].to_numpy(),
        ):
            raise ValueError(
                f"{name} targets do not match."
            )

        if not np.array_equal(
            combined[
                "cv_fold"
            ].to_numpy(),
            combined[
                f"{name}_cv_fold"
            ].to_numpy(),
        ):
            raise ValueError(
                f"{name} folds do not match."
            )

        if not np.array_equal(
            combined[
                "spatial_block_id"
            ].to_numpy(),
            combined[
                f"{name}_spatial_block_id"
            ].to_numpy(),
        ):
            raise ValueError(
                f"{name} spatial blocks do not match."
            )

        combined = combined.drop(
            columns=[
                f"{name}_target",
                f"{name}_spatial_block_id",
                f"{name}_cv_fold",
            ]
        )

    if len(
        combined
    ) != len(
        logistic
    ):
        raise ValueError(
            "Model OOF datasets do not contain identical rows."
        )

    if int(
        combined[
            TARGET_COLUMN
        ].sum()
    ) != 46:
        raise ValueError(
            "Unexpected Ankara positive target count."
        )

    return combined


def add_fold_percentile_ranks(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize each model score inside its own validation fold.

    This prevents absolute probability scale differences between
    independently fitted fold models from dominating pooled ranking.
    """

    result = dataframe.copy()

    for (
        _,
        (
            score_column,
            rank_column,
        ),
    ) in BASE_MODELS.items():
        result[
            rank_column
        ] = (
            result.groupby(
                "cv_fold"
            )[
                score_column
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        if not result[
            rank_column
        ].between(
            0,
            1,
        ).all():
            raise ValueError(
                f"Invalid percentile ranks in {rank_column}."
            )

    rank_columns = [
        rank_column
        for (
            _,
            rank_column,
        ) in BASE_MODELS.values()
    ]

    result[
        ENSEMBLE_COLUMN
    ] = (
        result[
            rank_columns
        ]
        .mean(
            axis=1
        )
    )

    return result


def calculate_top_fraction_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fraction: float,
) -> dict[str, float]:
    """Evaluate retrieval within a fixed highest-ranked fraction."""

    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction must be in the interval (0, 1]."
        )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    selected_count = max(
        1,
        int(
            np.ceil(
                len(scores)
                * fraction
            )
        ),
    )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    selected_indices = order[
        :selected_count
    ]

    recovered = int(
        y_true[
            selected_indices
        ].sum()
    )

    total_positives = int(
        y_true.sum()
    )

    precision = (
        recovered
        / selected_count
    )

    recall = (
        recovered
        / total_positives
    )

    prevalence = (
        total_positives
        / len(
            y_true
        )
    )

    lift = (
        precision
        / prevalence
        if prevalence > 0
        else float("nan")
    )

    return {
        "selected_count": float(
            selected_count
        ),
        "positive_count": float(
            recovered
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "lift": float(
            lift
        ),
    }


def calculate_fold_metrics(
    dataframe: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Calculate AP and ROC-AUC independently for each spatial fold."""

    records: list[
        dict[str, float | int]
    ] = []

    for fold in sorted(
        dataframe[
            "cv_fold"
        ].unique()
    ):
        subset = dataframe.loc[
            dataframe[
                "cv_fold"
            ] == fold
        ]

        y_true = subset[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        scores = subset[
            score_column
        ].to_numpy(
            dtype=float
        )

        records.append(
            {
                "cv_fold": int(
                    fold
                ),
                "row_count": len(
                    subset
                ),
                "positive_count": int(
                    y_true.sum()
                ),
                "average_precision": float(
                    average_precision_score(
                        y_true,
                        scores,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_true,
                        scores,
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def create_model_record(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    score_column: str,
) -> dict[str, float | str]:
    """Create comparable fold-normalized ranking metrics."""

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = dataframe[
        score_column
    ].to_numpy(
        dtype=float
    )

    fold_metrics = (
        calculate_fold_metrics(
            dataframe,
            score_column,
        )
    )

    top_one = (
        calculate_top_fraction_metrics(
            y_true,
            scores,
            fraction=0.01,
        )
    )

    top_five = (
        calculate_top_fraction_metrics(
            y_true,
            scores,
            fraction=0.05,
        )
    )

    return {
        "model": model_name,
        "pooled_average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "pooled_roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "mean_fold_average_precision": float(
            fold_metrics[
                "average_precision"
            ].mean()
        ),
        "std_fold_average_precision": float(
            fold_metrics[
                "average_precision"
            ].std(
                ddof=1
            )
        ),
        "min_fold_average_precision": float(
            fold_metrics[
                "average_precision"
            ].min()
        ),
        "max_fold_average_precision": float(
            fold_metrics[
                "average_precision"
            ].max()
        ),
        "mean_fold_roc_auc": float(
            fold_metrics[
                "roc_auc"
            ].mean()
        ),
        "std_fold_roc_auc": float(
            fold_metrics[
                "roc_auc"
            ].std(
                ddof=1
            )
        ),
        "top_1_percent_positive_count": float(
            top_one[
                "positive_count"
            ]
        ),
        "top_1_percent_recall": float(
            top_one[
                "recall"
            ]
        ),
        "top_1_percent_lift": float(
            top_one[
                "lift"
            ]
        ),
        "top_5_percent_positive_count": float(
            top_five[
                "positive_count"
            ]
        ),
        "top_5_percent_recall": float(
            top_five[
                "recall"
            ]
        ),
        "top_5_percent_lift": float(
            top_five[
                "lift"
            ]
        ),
    }


def create_comparison(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compare three models and their fixed unweighted rank ensemble."""

    records: list[
        dict[str, float | str]
    ] = []

    for (
        model_name,
        (
            _,
            rank_column,
        ),
    ) in BASE_MODELS.items():
        records.append(
            create_model_record(
                dataframe,
                model_name=model_name,
                score_column=rank_column,
            )
        )

    records.append(
        create_model_record(
            dataframe,
            model_name=(
                "Unweighted Rank Ensemble"
            ),
            score_column=ENSEMBLE_COLUMN,
        )
    )

    comparison = pd.DataFrame(
        records
    )

    return comparison


def create_rank_correlation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pairwise Spearman agreement after fold normalization."""

    rank_columns = {
        "Logistic Regression": (
            "logistic_rank"
        ),
        "Random Forest": (
            "random_forest_rank"
        ),
        "Gradient Boosting": (
            "gradient_boosting_rank"
        ),
    }

    renamed = dataframe[
        list(
            rank_columns.values()
        )
    ].rename(
        columns={
            value: key
            for key, value in rank_columns.items()
        }
    )

    return renamed.corr(
        method="spearman"
    )


def validate_outputs(
    dataframe: pd.DataFrame,
    comparison: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Validate model-comparison outputs."""

    if len(
        comparison
    ) != 4:
        raise ValueError(
            "Expected three base models and one ensemble."
        )

    if comparison[
        "model"
    ].duplicated().any():
        raise ValueError(
            "Duplicate comparison model names."
        )

    if dataframe[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate combined OOF grid IDs."
        )

    if dataframe[
        ENSEMBLE_COLUMN
    ].isna().any():
        raise ValueError(
            "Missing ensemble scores."
        )

    if not dataframe[
        ENSEMBLE_COLUMN
    ].between(
        0,
        1,
    ).all():
        raise ValueError(
            "Ensemble scores are outside 0-1."
        )

    expected_correlation_shape = (
        len(
            BASE_MODELS
        ),
        len(
            BASE_MODELS
        ),
    )

    if correlation.shape != (
        expected_correlation_shape
    ):
        raise ValueError(
            "Unexpected rank-correlation matrix shape."
        )

    numeric_columns = [
        column
        for column in comparison.columns
        if column != "model"
    ]

    for column in numeric_columns:
        values = comparison[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )


def create_pr_curve(
    dataframe: pd.DataFrame,
) -> None:
    """Plot fold-normalized OOF precision-recall curves."""

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    curve_columns = {
        "Logistic Regression": (
            "logistic_rank"
        ),
        "Random Forest": (
            "random_forest_rank"
        ),
        "Gradient Boosting": (
            "gradient_boosting_rank"
        ),
        "Rank Ensemble": (
            ENSEMBLE_COLUMN
        ),
    }

    for (
        label,
        score_column,
    ) in curve_columns.items():
        scores = dataframe[
            score_column
        ].to_numpy(
            dtype=float
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                scores,
            )
        )

        ap = float(
            average_precision_score(
                y_true,
                scores,
            )
        )

        axis.plot(
            recall,
            precision,
            label=(
                f"{label} "
                f"(AP={ap:.4f})"
            ),
        )

    prevalence = float(
        y_true.mean()
    )

    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Positive prevalence "
            f"({prevalence:.4%})"
        ),
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_xlabel(
        "Recall"
    )

    axis.set_ylabel(
        "Precision"
    )

    axis.set_title(
        "VoltSight - Fold-Normalized Spatial OOF Ranking"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        PR_CURVE_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_outputs(
    dataframe: pd.DataFrame,
    comparison: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Save ranking comparison outputs."""

    COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    dataframe.sort_values(
        "grid_id",
        kind="stable",
    ).to_csv(
        COMBINED_OOF_PATH,
        index=False,
        encoding="utf-8",
    )

    correlation.to_csv(
        CORRELATION_PATH,
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
    comparison: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Create final ranking-comparison documentation."""

    comparison_lines = "\n".join(
        (
            f"- {row.model}: "
            f"pooled AP "
            f"{row.pooled_average_precision:.6f}, "
            f"mean fold AP "
            f"{row.mean_fold_average_precision:.6f}, "
            f"AP std "
            f"{row.std_fold_average_precision:.6f}, "
            f"top-1% recall "
            f"{row.top_1_percent_recall:.4f}, "
            f"top-5% recall "
            f"{row.top_5_percent_recall:.4f}"
        )
        for row in comparison.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Model Ranking Comparison

## Evaluation Design

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Spatial folds: 5
- Spatial block size: 5 km

## Fold-Normalized Ranking

Each base model was independently trained inside each spatial
cross-validation iteration.

Absolute score scales can therefore differ across folds.

Before pooled top-ranked retrieval analysis, every model score is
converted to a percentile rank inside its own validation fold.

This normalization preserves within-fold ordering while reducing the
effect of incompatible absolute score scales across independently
fitted fold models.

## Models

{comparison_lines}

## Ensemble

The ensemble is a fixed equal-weight mean of the three fold-normalized
model ranks.

No model-specific ensemble weights were tuned.

This avoids selecting weights against only 46 positive observations.

The ensemble should be interpreted as an exploratory ranking
combination rather than an independently validated production model.

## Base-Model Rank Correlation

{correlation.to_string()}

Lower rank agreement indicates that models capture partially different
patterns and may therefore contain complementary ranking information.

## Important Limitation

The same spatial OOF predictions are used to describe the individual
models and the fixed ensemble.

Although ensemble weights are not tuned, the ensemble analysis is still
exploratory and should not be treated as an independent external
validation result.

## Outputs

- `data/processed/ankara_model_ranking_comparison.csv`
- `data/processed/ankara_model_rank_ensemble_oof.csv`
- `data/processed/ankara_model_rank_correlation.csv`
- `docs/ankara_model_rank_comparison_pr_curve.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    comparison: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    """Print concise final ranking comparison."""

    print("-" * 70)

    print(
        comparison[
            [
                "model",
                "pooled_average_precision",
                "mean_fold_average_precision",
                "std_fold_average_precision",
                "pooled_roc_auc",
                "top_1_percent_recall",
                "top_5_percent_recall",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Fold-normalized base-model "
        "Spearman rank correlation:"
    )

    print(
        correlation.to_string()
    )


def main() -> None:
    """Compare fold-normalized model rankings and fixed ensemble."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Model Ranking Comparison"
    )

    print("=" * 70)

    dataframe = (
        load_and_align_oof()
    )

    dataframe = (
        add_fold_percentile_ranks(
            dataframe
        )
    )

    comparison = (
        create_comparison(
            dataframe
        )
    )

    correlation = (
        create_rank_correlation(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        comparison,
        correlation,
    )

    save_outputs(
        dataframe,
        comparison,
        correlation,
    )

    create_pr_curve(
        dataframe
    )

    create_summary(
        dataframe,
        comparison,
        correlation,
    )

    print_results(
        comparison,
        correlation,
    )

    print("=" * 70)

    print(
        "Ankara model ranking comparison "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
