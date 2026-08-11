from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from voltsight.core.ankara_ml_features import (
    HISTORICAL_FULL_14_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_cv_folds.csv"
)

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_oof_predictions.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_fold_metrics.csv"
)

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_random_forest_baseline_feature_importance.csv"
)

PR_CURVE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_random_forest_baseline_pr_curve.png"
)

IMPORTANCE_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_random_forest_baseline_feature_importance.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_random_forest_baseline_summary.md"
)

FEATURE_COLUMNS = (
    HISTORICAL_FULL_14_FEATURE_COLUMNS
)

N_SPLITS = 5
RANDOM_STATE = 42


def load_inputs() -> pd.DataFrame:
    """Load predictor data and predefined spatial folds."""

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_PATH}"
        )

    if not FOLD_PATH.exists():
        raise FileNotFoundError(
            f"Spatial fold dataset not found: {FOLD_PATH}"
        )

    training = pd.read_csv(
        TRAINING_PATH,
        dtype={"grid_id": str},
    )

    folds = pd.read_csv(
        FOLD_PATH,
        dtype={
            "grid_id": str,
            "spatial_block_id": str,
        },
    )

    required_training = {
        "grid_id",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing = (
        required_training
        - set(training.columns)
    )

    if missing:
        raise ValueError(
            "Training columns are missing: "
            f"{sorted(missing)}"
        )

    required_folds = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
    }

    missing = (
        required_folds
        - set(folds.columns)
    )

    if missing:
        raise ValueError(
            "Fold columns are missing: "
            f"{sorted(missing)}"
        )

    if training["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate training grid IDs found."
        )

    if folds["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate fold grid IDs found."
        )

    merged = training.merge(
        folds[
            [
                "grid_id",
                "spatial_block_id",
                "cv_fold",
                TARGET_COLUMN,
            ]
        ].rename(
            columns={
                TARGET_COLUMN: "fold_target",
            }
        ),
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(training):
        raise ValueError(
            "Not every training row matched a spatial fold."
        )

    merged[TARGET_COLUMN] = (
        pd.to_numeric(
            merged[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    merged["fold_target"] = (
        pd.to_numeric(
            merged["fold_target"],
            errors="raise",
        )
        .astype(int)
    )

    if not np.array_equal(
        merged[TARGET_COLUMN].to_numpy(),
        merged["fold_target"].to_numpy(),
    ):
        raise ValueError(
            "Training and fold targets do not match."
        )

    merged["cv_fold"] = (
        pd.to_numeric(
            merged["cv_fold"],
            errors="raise",
        )
        .astype(int)
    )

    if set(
        merged["cv_fold"].unique()
    ) != set(
        range(N_SPLITS)
    ):
        raise ValueError(
            "Unexpected CV fold identifiers."
        )

    for feature in FEATURE_COLUMNS:
        merged[feature] = pd.to_numeric(
            merged[feature],
            errors="coerce",
        )

        values = merged[
            feature
        ].to_numpy(
            dtype=float
        )

        if (
            merged[feature].isna().any()
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Invalid values found in {feature}."
            )

    return merged


def build_model() -> RandomForestClassifier:
    """Create conservative class-weighted tree baseline."""

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate ranking and threshold diagnostics."""

    predictions = (
        scores >= threshold
    ).astype(int)

    return {
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
        "precision_at_0_5": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


def calculate_top_fraction_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fraction: float,
) -> dict[str, float]:
    """Calculate retrieval metrics for highest-ranked cells."""

    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction must be between 0 and 1."
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

    selected = order[
        :selected_count
    ]

    recovered = int(
        y_true[
            selected
        ].sum()
    )

    total_positive = int(
        y_true.sum()
    )

    precision = (
        recovered
        / selected_count
    )

    recall = (
        recovered
        / total_positive
    )

    prevalence = (
        total_positive
        / len(y_true)
    )

    lift = (
        precision
        / prevalence
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


def run_spatial_cross_validation(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Generate spatial out-of-fold Random Forest predictions."""

    oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    oof["random_forest_score"] = np.nan

    fold_records: list[
        dict[str, float | int]
    ] = []

    base_model = build_model()

    features = list(
        FEATURE_COLUMNS
    )

    for fold in range(
        N_SPLITS
    ):
        train_mask = (
            dataframe["cv_fold"]
            != fold
        )

        validation_mask = (
            dataframe["cv_fold"]
            == fold
        )

        train = dataframe.loc[
            train_mask
        ]

        validation = dataframe.loc[
            validation_mask
        ]

        x_train = train[
            features
        ]

        y_train = train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        x_validation = validation[
            features
        ]

        y_validation = validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        if len(
            np.unique(y_train)
        ) != 2:
            raise ValueError(
                f"Fold {fold} training data "
                "does not contain both classes."
            )

        if len(
            np.unique(y_validation)
        ) != 2:
            raise ValueError(
                f"Fold {fold} validation data "
                "does not contain both classes."
            )

        model = clone(
            base_model
        )

        model.fit(
            x_train,
            y_train,
        )

        scores = (
            model.predict_proba(
                x_validation
            )[
                :,
                1,
            ]
        )

        oof.loc[
            validation_mask,
            "random_forest_score",
        ] = scores

        metrics = calculate_metrics(
            y_validation,
            scores,
        )

        fold_records.append(
            {
                "cv_fold": fold,
                "validation_rows": len(
                    validation
                ),
                "validation_positives": int(
                    y_validation.sum()
                ),
                "training_rows": len(
                    train
                ),
                "training_positives": int(
                    y_train.sum()
                ),
                "random_forest_average_precision": (
                    metrics[
                        "average_precision"
                    ]
                ),
                "random_forest_roc_auc": (
                    metrics[
                        "roc_auc"
                    ]
                ),
                "random_forest_precision_at_0_5": (
                    metrics[
                        "precision_at_0_5"
                    ]
                ),
                "random_forest_recall_at_0_5": (
                    metrics[
                        "recall_at_0_5"
                    ]
                ),
                "random_forest_f1_at_0_5": (
                    metrics[
                        "f1_at_0_5"
                    ]
                ),
            }
        )

    if oof[
        "random_forest_score"
    ].isna().any():
        raise ValueError(
            "Missing Random Forest OOF predictions."
        )

    return (
        oof,
        pd.DataFrame(
            fold_records
        ),
    )


def fit_full_model(
    dataframe: pd.DataFrame,
) -> tuple[
    RandomForestClassifier,
    pd.DataFrame,
]:
    """Fit descriptive full-data model and collect importance."""

    model = build_model()

    model.fit(
        dataframe[
            list(FEATURE_COLUMNS)
        ],
        dataframe[
            TARGET_COLUMN
        ],
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_importance": (
                model.feature_importances_
            ),
        }
    )

    importance = (
        importance.sort_values(
            [
                "feature_importance",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        model,
        importance,
    )


def validate_outputs(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Validate generated Random Forest outputs."""

    if len(oof) != len(
        dataframe
    ):
        raise ValueError(
            "OOF row count mismatch."
        )

    if oof[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate OOF grid IDs."
        )

    if not oof[
        "random_forest_score"
    ].between(
        0,
        1,
    ).all():
        raise ValueError(
            "Random Forest scores are outside 0-1."
        )

    if len(
        fold_metrics
    ) != N_SPLITS:
        raise ValueError(
            "Unexpected fold metric count."
        )

    if len(
        importance
    ) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unexpected feature importance count."
        )

    if importance[
        "feature"
    ].duplicated().any():
        raise ValueError(
            "Duplicate feature importances."
        )

    if not np.isclose(
        importance[
            "feature_importance"
        ].sum(),
        1.0,
    ):
        raise ValueError(
            "Feature importances do not sum to one."
        )


def create_pr_curve(
    oof: pd.DataFrame,
) -> None:
    """Create pooled spatial OOF precision-recall curve."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "random_forest_score"
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

    prevalence = float(
        y_true.mean()
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        recall,
        precision,
        label=(
            "Random Forest "
            f"(AP={ap:.4f})"
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
        "Ankara Random Forest - Spatial OOF Precision-Recall"
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


def create_importance_plot(
    importance: pd.DataFrame,
) -> None:
    """Create descriptive full-data impurity importance plot."""

    plot_data = (
        importance.sort_values(
            "feature_importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        plot_data[
            "feature"
        ],
        plot_data[
            "feature_importance"
        ],
    )

    axis.set_xlabel(
        "Impurity-based feature importance"
    )

    axis.set_title(
        "Ankara Random Forest - Feature Importance"
    )

    figure.tight_layout()

    figure.savefig(
        IMPORTANCE_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_outputs(
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Save Random Forest tabular outputs."""

    OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof.sort_values(
        "grid_id",
        kind="stable",
    ).to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_metrics.sort_values(
        "cv_fold",
        kind="stable",
    ).to_csv(
        FOLD_METRICS_PATH,
        index=False,
        encoding="utf-8",
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Create Markdown model summary."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "random_forest_score"
    ].to_numpy(
        dtype=float
    )

    metrics = calculate_metrics(
        y_true,
        scores,
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

    fold_ap_mean = float(
        fold_metrics[
            "random_forest_average_precision"
        ].mean()
    )

    fold_ap_std = float(
        fold_metrics[
            "random_forest_average_precision"
        ].std(
            ddof=1
        )
    )

    fold_roc_mean = float(
        fold_metrics[
            "random_forest_roc_auc"
        ].mean()
    )

    fold_roc_std = float(
        fold_metrics[
            "random_forest_roc_auc"
        ].std(
            ddof=1
        )
    )

    importance_lines = "\n".join(
        (
            f"- `{row.feature}`: "
            f"{row.feature_importance:.6f}"
        )
        for row in importance.head(
            10
        ).itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Random Forest Baseline

## Dataset

- Rows: {len(dataframe):,}
- Positive cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Predictors: {len(FEATURE_COLUMNS):,}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km

## Model

- Random Forest
- 400 trees
- Maximum depth: 12
- Minimum leaf samples: 5
- `class_weight="balanced_subsample"`
- `max_features="sqrt"`

Charging-derived variables are excluded from training predictors.

## Spatial OOF Performance

- Pooled average precision: {metrics["average_precision"]:.6f}
- Pooled ROC-AUC: {metrics["roc_auc"]:.6f}
- Mean fold average precision: {fold_ap_mean:.6f}
- Fold AP standard deviation: {fold_ap_std:.6f}
- Mean fold ROC-AUC: {fold_roc_mean:.6f}
- Fold ROC-AUC standard deviation: {fold_roc_std:.6f}

## Threshold 0.5 Diagnostic

- Precision: {metrics["precision_at_0_5"]:.6f}
- Recall: {metrics["recall_at_0_5"]:.6f}
- F1: {metrics["f1_at_0_5"]:.6f}

The threshold is diagnostic only and is not interpreted as a calibrated
real-world probability threshold.

## Ranking Performance

### Top 1%

- Cells inspected: {int(top_one["selected_count"]):,}
- Positive cells recovered: {int(top_one["positive_count"]):,}
- Recall: {top_one["recall"]:.6f}
- Lift: {top_one["lift"]:.2f}x

### Top 5%

- Cells inspected: {int(top_five["selected_count"]):,}
- Positive cells recovered: {int(top_five["positive_count"]):,}
- Recall: {top_five["recall"]:.6f}
- Lift: {top_five["lift"]:.2f}x

## Descriptive Feature Importance

{importance_lines}

Impurity-based Random Forest importance is descriptive and should not
be interpreted as causal importance.

Correlated predictors can share or distort feature importance.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Print key Random Forest results."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "random_forest_score"
    ].to_numpy(
        dtype=float
    )

    metrics = calculate_metrics(
        y_true,
        scores,
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

    print("-" * 70)

    print(
        "Training rows:",
        f"{len(dataframe):,}",
    )

    print(
        "Positive rows:",
        f"{int(y_true.sum()):,}",
    )

    print()

    print(
        "Spatial OOF Random Forest AP:",
        f"{metrics['average_precision']:.6f}",
    )

    print(
        "Spatial OOF Random Forest ROC-AUC:",
        f"{metrics['roc_auc']:.6f}",
    )

    print()

    print(
        "Mean fold AP:",
        f"{fold_metrics['random_forest_average_precision'].mean():.6f}",
    )

    print(
        "Std fold AP:",
        f"{fold_metrics['random_forest_average_precision'].std(ddof=1):.6f}",
    )

    print()

    print(
        "Threshold 0.5 precision:",
        f"{metrics['precision_at_0_5']:.6f}",
    )

    print(
        "Threshold 0.5 recall:",
        f"{metrics['recall_at_0_5']:.6f}",
    )

    print(
        "Threshold 0.5 F1:",
        f"{metrics['f1_at_0_5']:.6f}",
    )

    print()

    print(
        "Top 1% recovered positives:",
        f"{int(top_one['positive_count'])}"
        f"/{int(y_true.sum())}",
    )

    print(
        "Top 1% recall:",
        f"{top_one['recall']:.6f}",
    )

    print(
        "Top 1% lift:",
        f"{top_one['lift']:.2f}x",
    )

    print()

    print(
        "Top 5% recovered positives:",
        f"{int(top_five['positive_count'])}"
        f"/{int(y_true.sum())}",
    )

    print(
        "Top 5% recall:",
        f"{top_five['recall']:.6f}",
    )

    print(
        "Top 5% lift:",
        f"{top_five['lift']:.2f}x",
    )

    print()

    print(
        "Fold metrics:"
    )

    print(
        fold_metrics[
            [
                "cv_fold",
                "validation_positives",
                "random_forest_average_precision",
                "random_forest_roc_auc",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Train and evaluate Ankara Random Forest baseline."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Random Forest Baseline"
    )

    print("=" * 70)

    dataframe = load_inputs()

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    _, importance = (
        fit_full_model(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
        importance,
    )

    save_outputs(
        oof,
        fold_metrics,
        importance,
    )

    create_pr_curve(
        oof
    )

    create_importance_plot(
        importance
    )

    create_summary(
        dataframe,
        oof,
        fold_metrics,
        importance,
    )

    print_results(
        dataframe,
        oof,
        fold_metrics,
    )

    print("=" * 70)

    print(
        "Ankara Random Forest baseline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
