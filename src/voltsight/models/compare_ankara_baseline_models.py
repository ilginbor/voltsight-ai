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

LOGISTIC_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_logistic_baseline_oof_predictions.csv"
)

LOGISTIC_FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_logistic_baseline_fold_metrics.csv"
)

RF_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_oof_predictions.csv"
)

RF_FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_fold_metrics.csv"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_baseline_model_comparison.csv"
)

COMBINED_OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_baseline_combined_oof_predictions.csv"
)

PR_CURVE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_baseline_model_comparison_pr_curve.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_baseline_model_comparison.md"
)

TARGET_COLUMN = "has_existing_charging_station"


def calculate_top_fraction_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fraction: float,
) -> dict[str, float]:
    """Calculate retrieval performance in the top-ranked fraction."""

    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction must be between 0 and 1."
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
        / len(y_true)
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


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and align Logistic and Random Forest OOF outputs."""

    required_paths = (
        LOGISTIC_OOF_PATH,
        LOGISTIC_FOLD_PATH,
        RF_OOF_PATH,
        RF_FOLD_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model output not found: {path}"
            )

    logistic_oof = pd.read_csv(
        LOGISTIC_OOF_PATH,
        dtype={
            "grid_id": str,
            "spatial_block_id": str,
        },
    )

    rf_oof = pd.read_csv(
        RF_OOF_PATH,
        dtype={
            "grid_id": str,
            "spatial_block_id": str,
        },
    )

    logistic_folds = pd.read_csv(
        LOGISTIC_FOLD_PATH
    )

    rf_folds = pd.read_csv(
        RF_FOLD_PATH
    )

    required_logistic = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
        "logistic_score",
    }

    required_rf = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
        "random_forest_score",
    }

    missing = (
        required_logistic
        - set(logistic_oof.columns)
    )

    if missing:
        raise ValueError(
            "Logistic OOF columns missing: "
            f"{sorted(missing)}"
        )

    missing = (
        required_rf
        - set(rf_oof.columns)
    )

    if missing:
        raise ValueError(
            "Random Forest OOF columns missing: "
            f"{sorted(missing)}"
        )

    if logistic_oof[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Logistic OOF grid IDs."
        )

    if rf_oof[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Random Forest OOF grid IDs."
        )

    combined = logistic_oof[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
            "logistic_score",
        ]
    ].merge(
        rf_oof[
            [
                "grid_id",
                TARGET_COLUMN,
                "spatial_block_id",
                "cv_fold",
                "random_forest_score",
            ]
        ].rename(
            columns={
                TARGET_COLUMN: "rf_target",
                "spatial_block_id": "rf_spatial_block_id",
                "cv_fold": "rf_cv_fold",
            }
        ),
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(
        logistic_oof
    ):
        raise ValueError(
            "OOF model outputs do not contain identical rows."
        )

    if not np.array_equal(
        combined[
            TARGET_COLUMN
        ].to_numpy(),
        combined[
            "rf_target"
        ].to_numpy(),
    ):
        raise ValueError(
            "Model OOF targets do not match."
        )

    if not np.array_equal(
        combined[
            "cv_fold"
        ].to_numpy(),
        combined[
            "rf_cv_fold"
        ].to_numpy(),
    ):
        raise ValueError(
            "Model CV folds do not match."
        )

    if not np.array_equal(
        combined[
            "spatial_block_id"
        ].to_numpy(),
        combined[
            "rf_spatial_block_id"
        ].to_numpy(),
    ):
        raise ValueError(
            "Model spatial blocks do not match."
        )

    combined = combined.drop(
        columns=[
            "rf_target",
            "rf_spatial_block_id",
            "rf_cv_fold",
        ]
    )

    for score_column in [
        "logistic_score",
        "random_forest_score",
    ]:
        if combined[
            score_column
        ].isna().any():
            raise ValueError(
                f"Missing values found in {score_column}."
            )

        if not combined[
            score_column
        ].between(
            0,
            1,
        ).all():
            raise ValueError(
                f"{score_column} is outside 0-1."
            )

    return (
        combined,
        logistic_folds,
        rf_folds,
    )


def create_model_record(
    *,
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    fold_ap: pd.Series,
    fold_roc_auc: pd.Series,
) -> dict[str, float | str]:
    """Create one comparable model-performance record."""

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
            fold_ap.mean()
        ),
        "std_fold_average_precision": float(
            fold_ap.std(
                ddof=1
            )
        ),
        "min_fold_average_precision": float(
            fold_ap.min()
        ),
        "max_fold_average_precision": float(
            fold_ap.max()
        ),
        "mean_fold_roc_auc": float(
            fold_roc_auc.mean()
        ),
        "std_fold_roc_auc": float(
            fold_roc_auc.std(
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
    combined: pd.DataFrame,
    logistic_folds: pd.DataFrame,
    rf_folds: pd.DataFrame,
) -> pd.DataFrame:
    """Create comparable model summary table."""

    y_true = combined[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    logistic_record = (
        create_model_record(
            model_name=(
                "Logistic Regression"
            ),
            y_true=y_true,
            scores=combined[
                "logistic_score"
            ].to_numpy(
                dtype=float
            ),
            fold_ap=logistic_folds[
                "logistic_average_precision"
            ],
            fold_roc_auc=logistic_folds[
                "logistic_roc_auc"
            ],
        )
    )

    rf_record = (
        create_model_record(
            model_name=(
                "Random Forest"
            ),
            y_true=y_true,
            scores=combined[
                "random_forest_score"
            ].to_numpy(
                dtype=float
            ),
            fold_ap=rf_folds[
                "random_forest_average_precision"
            ],
            fold_roc_auc=rf_folds[
                "random_forest_roc_auc"
            ],
        )
    )

    comparison = pd.DataFrame(
        [
            logistic_record,
            rf_record,
        ]
    )

    return comparison


def validate_comparison(
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Validate comparison output."""

    if comparison[
        "model"
    ].duplicated().any():
        raise ValueError(
            "Duplicate model records found."
        )

    if len(
        comparison
    ) != 2:
        raise ValueError(
            "Unexpected number of baseline models."
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
                f"Non-finite comparison values in {column}."
            )

    if int(
        combined[
            TARGET_COLUMN
        ].sum()
    ) != 46:
        raise ValueError(
            "Unexpected positive target count."
        )


def create_pr_curve(
    combined: pd.DataFrame,
) -> None:
    """Create directly comparable pooled OOF PR curves."""

    y_true = combined[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    prevalence = float(
        y_true.mean()
    )

    logistic_scores = combined[
        "logistic_score"
    ].to_numpy(
        dtype=float
    )

    rf_scores = combined[
        "random_forest_score"
    ].to_numpy(
        dtype=float
    )

    logistic_precision, logistic_recall, _ = (
        precision_recall_curve(
            y_true,
            logistic_scores,
        )
    )

    rf_precision, rf_recall, _ = (
        precision_recall_curve(
            y_true,
            rf_scores,
        )
    )

    logistic_ap = float(
        average_precision_score(
            y_true,
            logistic_scores,
        )
    )

    rf_ap = float(
        average_precision_score(
            y_true,
            rf_scores,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.plot(
        logistic_recall,
        logistic_precision,
        label=(
            "Logistic Regression "
            f"(AP={logistic_ap:.4f})"
        ),
    )

    axis.plot(
        rf_recall,
        rf_precision,
        label=(
            "Random Forest "
            f"(AP={rf_ap:.4f})"
        ),
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
        "VoltSight - Ankara Spatial OOF Baseline Comparison"
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
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Save model comparison outputs."""

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

    combined.sort_values(
        "grid_id",
        kind="stable",
    ).to_csv(
        COMBINED_OOF_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Create Markdown model-comparison summary."""

    logistic = comparison.loc[
        comparison[
            "model"
        ] == "Logistic Regression"
    ].iloc[0]

    random_forest = comparison.loc[
        comparison[
            "model"
        ] == "Random Forest"
    ].iloc[0]

    score_correlation = float(
        combined[
            [
                "logistic_score",
                "random_forest_score",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    summary = f"""# Ankara Baseline Model Comparison

## Shared Evaluation Design

- Rows: {len(combined):,}
- Positive station cells: {int(combined[TARGET_COLUMN].sum()):,}
- Spatial block size: 5 km
- Cross-validation folds: 5
- Predictor set: identical leakage-safe road and parking features

Both models are evaluated using exactly the same predefined spatial
fold assignments.

## Logistic Regression

- Pooled AP: {logistic["pooled_average_precision"]:.6f}
- Mean fold AP: {logistic["mean_fold_average_precision"]:.6f}
- Fold AP std: {logistic["std_fold_average_precision"]:.6f}
- Pooled ROC-AUC: {logistic["pooled_roc_auc"]:.6f}
- Top 1% recall: {logistic["top_1_percent_recall"]:.6f}
- Top 5% recall: {logistic["top_5_percent_recall"]:.6f}

## Random Forest

- Pooled AP: {random_forest["pooled_average_precision"]:.6f}
- Mean fold AP: {random_forest["mean_fold_average_precision"]:.6f}
- Fold AP std: {random_forest["std_fold_average_precision"]:.6f}
- Pooled ROC-AUC: {random_forest["pooled_roc_auc"]:.6f}
- Top 1% recall: {random_forest["top_1_percent_recall"]:.6f}
- Top 5% recall: {random_forest["top_5_percent_recall"]:.6f}

## Interpretation

Random Forest produces the stronger pooled average precision and
slightly stronger top-5-percent retrieval.

Logistic Regression produces substantially lower fold-to-fold AP
variation and stronger retrieval within the highest-ranked one percent
of grid cells.

Random Forest therefore does not unambiguously replace the Logistic
Regression baseline.

The two models capture partially different ranking behavior.

## OOF Ranking Agreement

- Spearman score correlation: {score_correlation:.6f}

## Model-Selection Policy

No final model is selected using accuracy.

Primary criteria are:

- pooled average precision
- fold-level average precision stability
- top-1-percent recall
- top-5-percent recall
- lift over class prevalence

ROC-AUC is retained as a secondary ranking metric.

## Important Limitations

Only 46 positive station cells are available.

Performance differences are therefore sensitive to the geographic
distribution of rare positive examples.

Random Forest impurity importance and Logistic Regression coefficients
are descriptive rather than causal.

## Outputs

- `data/processed/ankara_baseline_model_comparison.csv`
- `data/processed/ankara_baseline_combined_oof_predictions.csv`
- `docs/ankara_baseline_model_comparison_pr_curve.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_comparison(
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Print concise baseline comparison."""

    print("-" * 70)

    display_columns = [
        "model",
        "pooled_average_precision",
        "mean_fold_average_precision",
        "std_fold_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    correlation = (
        combined[
            [
                "logistic_score",
                "random_forest_score",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    print(
        "OOF score Spearman correlation:",
        f"{correlation:.6f}",
    )


def main() -> None:
    """Compare Ankara baseline models."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Baseline Model Comparison"
    )

    print("=" * 70)

    (
        combined,
        logistic_folds,
        rf_folds,
    ) = load_inputs()

    comparison = create_comparison(
        combined,
        logistic_folds,
        rf_folds,
    )

    validate_comparison(
        combined,
        comparison,
    )

    save_outputs(
        combined,
        comparison,
    )

    create_pr_curve(
        combined
    )

    create_summary(
        combined,
        comparison,
    )

    print_comparison(
        combined,
        comparison,
    )

    print("=" * 70)

    print(
        "Ankara baseline model comparison "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
