from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


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
    / "ankara_gradient_boosting_baseline_oof_predictions.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_gradient_boosting_baseline_fold_metrics.csv"
)

PR_CURVE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_gradient_boosting_baseline_pr_curve.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_gradient_boosting_baseline_summary.md"
)

TARGET_COLUMN = "has_existing_charging_station"

FEATURE_COLUMNS = (
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
)

N_SPLITS = 5
RANDOM_STATE = 42


def create_output_directories() -> None:
    """Create directories required by model outputs."""

    OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_inputs() -> pd.DataFrame:
    """Load leakage-safe predictors and predefined spatial folds."""

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
        dtype={
            "grid_id": str,
        },
    )

    folds = pd.read_csv(
        FOLD_PATH,
        dtype={
            "grid_id": str,
            "spatial_block_id": str,
        },
    )

    required_training_columns = {
        "grid_id",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_training_columns = (
        required_training_columns
        - set(training.columns)
    )

    if missing_training_columns:
        raise ValueError(
            "Training dataset is missing columns: "
            f"{sorted(missing_training_columns)}"
        )

    required_fold_columns = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
    }

    missing_fold_columns = (
        required_fold_columns
        - set(folds.columns)
    )

    if missing_fold_columns:
        raise ValueError(
            "Spatial fold dataset is missing columns: "
            f"{sorted(missing_fold_columns)}"
        )

    if training.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if folds.empty:
        raise ValueError(
            "Spatial fold dataset is empty."
        )

    if training[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate training grid IDs were found."
        )

    if folds[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate fold grid IDs were found."
        )

    merged = training.merge(
        folds[
            [
                "grid_id",
                TARGET_COLUMN,
                "spatial_block_id",
                "cv_fold",
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

    if len(
        merged
    ) != len(
        training
    ):
        raise ValueError(
            "Not every training row matched a spatial fold."
        )

    merged[
        TARGET_COLUMN
    ] = (
        pd.to_numeric(
            merged[
                TARGET_COLUMN
            ],
            errors="raise",
        )
        .astype(int)
    )

    merged[
        "fold_target"
    ] = (
        pd.to_numeric(
            merged[
                "fold_target"
            ],
            errors="raise",
        )
        .astype(int)
    )

    if not np.array_equal(
        merged[
            TARGET_COLUMN
        ].to_numpy(),
        merged[
            "fold_target"
        ].to_numpy(),
    ):
        raise ValueError(
            "Training and spatial-fold targets do not match."
        )

    merged[
        "cv_fold"
    ] = (
        pd.to_numeric(
            merged[
                "cv_fold"
            ],
            errors="raise",
        )
        .astype(int)
    )

    expected_folds = set(
        range(
            N_SPLITS
        )
    )

    actual_folds = set(
        merged[
            "cv_fold"
        ].unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            "Unexpected CV fold identifiers: "
            f"{sorted(actual_folds)}"
        )

    if set(
        merged[
            TARGET_COLUMN
        ].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Target must contain both classes."
        )

    for feature in FEATURE_COLUMNS:
        merged[
            feature
        ] = pd.to_numeric(
            merged[
                feature
            ],
            errors="coerce",
        )

        if merged[
            feature
        ].isna().any():
            raise ValueError(
                f"Missing values found in {feature}."
            )

        values = merged[
            feature
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite values found in {feature}."
            )

    return merged


def build_model() -> HistGradientBoostingClassifier:
    """
    Create a conservative nonlinear boosting baseline.

    Early stopping is intentionally disabled so that the estimator does
    not create an internal random validation split inside each spatial
    training fold.
    """

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def calculate_balanced_sample_weights(
    y: np.ndarray,
) -> np.ndarray:
    """Create balanced observation weights for the rare target."""

    y = np.asarray(
        y,
        dtype=int,
    )

    if len(
        np.unique(
            y
        )
    ) != 2:
        raise ValueError(
            "Both classes are required to create sample weights."
        )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if not np.isfinite(
        weights
    ).all():
        raise ValueError(
            "Sample weights contain non-finite values."
        )

    if (
        weights <= 0
    ).any():
        raise ValueError(
            "Sample weights must be positive."
        )

    return weights


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate rare-class ranking and threshold diagnostics."""

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(
        np.unique(
            y_true
        )
    ) != 2:
        raise ValueError(
            "Both classes are required for evaluation."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Prediction scores contain non-finite values."
        )

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
        "predicted_positive_count_at_0_5": float(
            predictions.sum()
        ),
    }


def calculate_top_fraction_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    fraction: float,
) -> dict[str, float]:
    """Evaluate retrieval inside the highest-ranked fraction."""

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

    recovered_positive_count = int(
        y_true[
            selected_indices
        ].sum()
    )

    total_positive_count = int(
        y_true.sum()
    )

    precision = (
        recovered_positive_count
        / selected_count
    )

    recall = (
        recovered_positive_count
        / total_positive_count
    )

    prevalence = (
        total_positive_count
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
            recovered_positive_count
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
    """Generate spatial OOF HistGradientBoosting predictions."""

    oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    oof[
        "gradient_boosting_score"
    ] = np.nan

    fold_records: list[
        dict[str, float | int]
    ] = []

    base_model = build_model()

    feature_list = list(
        FEATURE_COLUMNS
    )

    for fold in range(
        N_SPLITS
    ):
        train_mask = (
            dataframe[
                "cv_fold"
            ] != fold
        )

        validation_mask = (
            dataframe[
                "cv_fold"
            ] == fold
        )

        train = dataframe.loc[
            train_mask
        ]

        validation = dataframe.loc[
            validation_mask
        ]

        if train.empty:
            raise ValueError(
                f"Fold {fold} has no training rows."
            )

        if validation.empty:
            raise ValueError(
                f"Fold {fold} has no validation rows."
            )

        x_train = train[
            feature_list
        ]

        y_train = train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        x_validation = validation[
            feature_list
        ]

        y_validation = validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

        if len(
            np.unique(
                y_train
            )
        ) != 2:
            raise ValueError(
                f"Fold {fold} training data "
                "does not contain both classes."
            )

        if len(
            np.unique(
                y_validation
            )
        ) != 2:
            raise ValueError(
                f"Fold {fold} validation data "
                "does not contain both classes."
            )

        sample_weights = (
            calculate_balanced_sample_weights(
                y_train
            )
        )

        model = clone(
            base_model
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
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
            "gradient_boosting_score",
        ] = scores

        metrics = (
            calculate_metrics(
                y_validation,
                scores,
            )
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
                "gradient_boosting_average_precision": (
                    metrics[
                        "average_precision"
                    ]
                ),
                "gradient_boosting_roc_auc": (
                    metrics[
                        "roc_auc"
                    ]
                ),
                "gradient_boosting_precision_at_0_5": (
                    metrics[
                        "precision_at_0_5"
                    ]
                ),
                "gradient_boosting_recall_at_0_5": (
                    metrics[
                        "recall_at_0_5"
                    ]
                ),
                "gradient_boosting_f1_at_0_5": (
                    metrics[
                        "f1_at_0_5"
                    ]
                ),
                "gradient_boosting_predicted_positive_count_at_0_5": int(
                    metrics[
                        "predicted_positive_count_at_0_5"
                    ]
                ),
            }
        )

    if oof[
        "gradient_boosting_score"
    ].isna().any():
        raise ValueError(
            "Missing Gradient Boosting OOF predictions."
        )

    return (
        oof,
        pd.DataFrame(
            fold_records
        ),
    )


def fit_full_model(
    dataframe: pd.DataFrame,
) -> HistGradientBoostingClassifier:
    """Fit the descriptive full-data model after OOF evaluation."""

    model = build_model()

    x = dataframe[
        list(
            FEATURE_COLUMNS
        )
    ]

    y = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    sample_weights = (
        calculate_balanced_sample_weights(
            y
        )
    )

    model.fit(
        x,
        y,
        sample_weight=sample_weights,
    )

    return model


def validate_outputs(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Validate generated boosting outputs."""

    if len(
        oof
    ) != len(
        dataframe
    ):
        raise ValueError(
            "OOF row count does not match training data."
        )

    if oof[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate OOF grid IDs were found."
        )

    scores = oof[
        "gradient_boosting_score"
    ]

    if scores.isna().any():
        raise ValueError(
            "Missing Gradient Boosting scores."
        )

    if not scores.between(
        0,
        1,
    ).all():
        raise ValueError(
            "Gradient Boosting scores are outside 0-1."
        )

    if len(
        fold_metrics
    ) != N_SPLITS:
        raise ValueError(
            "Unexpected fold metric count."
        )

    if int(
        fold_metrics[
            "validation_rows"
        ].sum()
    ) != len(
        dataframe
    ):
        raise ValueError(
            "Fold validation rows do not preserve the dataset."
        )

    if int(
        fold_metrics[
            "validation_positives"
        ].sum()
    ) != int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    ):
        raise ValueError(
            "Fold positives do not preserve the target."
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
        "gradient_boosting_score"
    ].to_numpy(
        dtype=float
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    average_precision = float(
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
            "HistGradientBoosting "
            f"(AP={average_precision:.4f})"
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
        "Ankara Gradient Boosting - Spatial OOF Precision-Recall"
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
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Save boosting model outputs."""

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


def create_summary(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> None:
    """Create Markdown summary for the boosting baseline."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "gradient_boosting_score"
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
            "gradient_boosting_average_precision"
        ].mean()
    )

    fold_ap_std = float(
        fold_metrics[
            "gradient_boosting_average_precision"
        ].std(
            ddof=1
        )
    )

    fold_roc_mean = float(
        fold_metrics[
            "gradient_boosting_roc_auc"
        ].mean()
    )

    fold_roc_std = float(
        fold_metrics[
            "gradient_boosting_roc_auc"
        ].std(
            ddof=1
        )
    )

    fold_lines = "\n".join(
        (
            f"- Fold {int(row.cv_fold)}: "
            f"{int(row.validation_positives)} positives, "
            f"AP {row.gradient_boosting_average_precision:.6f}, "
            f"ROC-AUC {row.gradient_boosting_roc_auc:.6f}"
        )
        for row in fold_metrics.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Gradient Boosting Baseline

## Dataset

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Predictor features: {len(FEATURE_COLUMNS):,}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km

## Model

The nonlinear baseline uses scikit-learn HistGradientBoostingClassifier.

Fixed configuration:

- learning rate: 0.05
- maximum iterations: 150
- maximum leaf nodes: 15
- minimum samples per leaf: 100
- L2 regularization: 1.0
- early stopping: disabled
- balanced training sample weights

No hyperparameter search was performed.

Disabling internal early stopping prevents the estimator from creating
a random internal validation split inside the predefined spatial
training folds.

Charging-derived features are excluded from the predictors.

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
- Predicted-positive cells: {int(metrics["predicted_positive_count_at_0_5"]):,}

Balanced sample weights change the effective class distribution seen
during fitting.

The resulting scores should therefore be treated as ranking scores
rather than calibrated real-world charging-station probabilities.

## Ranking Performance

### Top 1%

- Cells inspected: {int(top_one["selected_count"]):,}
- Existing-station cells recovered: {int(top_one["positive_count"]):,}
- Recall: {top_one["recall"]:.6f}
- Lift: {top_one["lift"]:.2f}x

### Top 5%

- Cells inspected: {int(top_five["selected_count"]):,}
- Existing-station cells recovered: {int(top_five["positive_count"]):,}
- Recall: {top_five["recall"]:.6f}
- Lift: {top_five["lift"]:.2f}x

## Fold-Level Results

{fold_lines}

## Interpretation Policy

This model is compared directly with Logistic Regression and Random
Forest using exactly the same predefined 5-km spatial folds.

Model selection should consider:

- pooled average precision
- fold AP stability
- top-1-percent recall
- top-5-percent recall
- lift over prevalence

ROC-AUC remains a secondary metric.

Accuracy is not used as a primary metric.

## Outputs

- `data/processed/ankara_gradient_boosting_baseline_oof_predictions.csv`
- `data/processed/ankara_gradient_boosting_baseline_fold_metrics.csv`
- `docs/ankara_gradient_boosting_baseline_pr_curve.png`

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
    """Print key spatial boosting results."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "gradient_boosting_score"
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
        "Spatial OOF Gradient Boosting AP:",
        f"{metrics['average_precision']:.6f}",
    )

    print(
        "Spatial OOF Gradient Boosting ROC-AUC:",
        f"{metrics['roc_auc']:.6f}",
    )

    print()

    print(
        "Mean fold AP:",
        f"{fold_metrics['gradient_boosting_average_precision'].mean():.6f}",
    )

    print(
        "Std fold AP:",
        f"{fold_metrics['gradient_boosting_average_precision'].std(ddof=1):.6f}",
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
                "gradient_boosting_average_precision",
                "gradient_boosting_roc_auc",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Train and evaluate Ankara Gradient Boosting baseline."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Gradient Boosting Baseline"
    )

    print("=" * 70)

    create_output_directories()

    dataframe = load_inputs()

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    fit_full_model(
        dataframe
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
    )

    save_outputs(
        oof,
        fold_metrics,
    )

    create_pr_curve(
        oof
    )

    create_summary(
        dataframe,
        oof,
        fold_metrics,
    )

    print_results(
        dataframe,
        oof,
        fold_metrics,
    )

    print("=" * 70)

    print(
        "Ankara Gradient Boosting baseline "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
