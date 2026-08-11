from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    / "ankara_logistic_baseline_oof_predictions.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_logistic_baseline_fold_metrics.csv"
)

COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_logistic_baseline_coefficients.csv"
)

PR_CURVE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_logistic_baseline_pr_curve.png"
)

COEFFICIENT_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_logistic_baseline_coefficients.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_logistic_baseline_summary.md"
)

FEATURE_COLUMNS = (
    HISTORICAL_FULL_14_FEATURE_COLUMNS
)

N_SPLITS = 5
RANDOM_STATE = 42


def create_output_directories() -> None:
    """Create directories required by generated outputs."""

    OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_inputs() -> pd.DataFrame:
    """Load training predictors and precomputed spatial folds."""

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_PATH}"
        )

    if not FOLD_PATH.exists():
        raise FileNotFoundError(
            f"Spatial CV fold file not found: {FOLD_PATH}"
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
            "Fold dataset is missing columns: "
            f"{sorted(missing_fold_columns)}"
        )

    if training["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in training data."
        )

    if folds["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in fold data."
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
                TARGET_COLUMN: (
                    "fold_target"
                ),
            }
        ),
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(training):
        raise ValueError(
            "Not every training row received a spatial fold."
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

    expected_folds = set(
        range(N_SPLITS)
    )

    actual_folds = set(
        merged[
            "cv_fold"
        ].unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            "Unexpected spatial fold identifiers: "
            f"{sorted(actual_folds)}"
        )

    for feature in FEATURE_COLUMNS:
        merged[feature] = pd.to_numeric(
            merged[feature],
            errors="coerce",
        )

        if merged[feature].isna().any():
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

    return merged


def build_logistic_pipeline() -> Pipeline:
    """Create class-weighted regularized logistic baseline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_dummy_classifier() -> DummyClassifier:
    """Create prior-probability dummy baseline."""

    return DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE,
    )


def calculate_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate ranking and threshold-based binary metrics."""

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
    """Evaluate retrieval performance within the highest-ranked cells."""

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

    count = max(
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
        :count
    ]

    positives_selected = int(
        y_true[
            selected
        ].sum()
    )

    total_positives = int(
        y_true.sum()
    )

    precision = (
        positives_selected
        / count
    )

    recall = (
        positives_selected
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
            count
        ),
        "positive_count": float(
            positives_selected
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
    """Generate logistic and dummy out-of-fold predictions."""

    oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    oof[
        "logistic_score"
    ] = np.nan

    oof[
        "dummy_score"
    ] = np.nan

    fold_records: list[
        dict[str, float | int]
    ] = []

    base_logistic = (
        build_logistic_pipeline()
    )

    base_dummy = (
        build_dummy_classifier()
    )

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

        if train[
            TARGET_COLUMN
        ].nunique() != 2:
            raise ValueError(
                f"Fold {fold} training data "
                "does not contain both classes."
            )

        if validation[
            TARGET_COLUMN
        ].nunique() != 2:
            raise ValueError(
                f"Fold {fold} validation data "
                "does not contain both classes."
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

        logistic = clone(
            base_logistic
        )

        dummy = clone(
            base_dummy
        )

        logistic.fit(
            x_train,
            y_train,
        )

        dummy.fit(
            x_train,
            y_train,
        )

        logistic_scores = (
            logistic.predict_proba(
                x_validation
            )[
                :,
                1,
            ]
        )

        dummy_scores = (
            dummy.predict_proba(
                x_validation
            )[
                :,
                1,
            ]
        )

        oof.loc[
            validation_mask,
            "logistic_score",
        ] = logistic_scores

        oof.loc[
            validation_mask,
            "dummy_score",
        ] = dummy_scores

        logistic_metrics = (
            calculate_binary_metrics(
                y_validation,
                logistic_scores,
            )
        )

        dummy_metrics = (
            calculate_binary_metrics(
                y_validation,
                dummy_scores,
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
                "logistic_average_precision": (
                    logistic_metrics[
                        "average_precision"
                    ]
                ),
                "logistic_roc_auc": (
                    logistic_metrics[
                        "roc_auc"
                    ]
                ),
                "logistic_precision_at_0_5": (
                    logistic_metrics[
                        "precision_at_0_5"
                    ]
                ),
                "logistic_recall_at_0_5": (
                    logistic_metrics[
                        "recall_at_0_5"
                    ]
                ),
                "logistic_f1_at_0_5": (
                    logistic_metrics[
                        "f1_at_0_5"
                    ]
                ),
                "logistic_predicted_positive_count_at_0_5": int(
                    logistic_metrics[
                        "predicted_positive_count_at_0_5"
                    ]
                ),
                "dummy_average_precision": (
                    dummy_metrics[
                        "average_precision"
                    ]
                ),
                "dummy_roc_auc": (
                    dummy_metrics[
                        "roc_auc"
                    ]
                ),
            }
        )

    if oof[
        "logistic_score"
    ].isna().any():
        raise ValueError(
            "Missing logistic out-of-fold predictions."
        )

    if oof[
        "dummy_score"
    ].isna().any():
        raise ValueError(
            "Missing dummy out-of-fold predictions."
        )

    return (
        oof,
        pd.DataFrame(
            fold_records
        ),
    )


def fit_full_logistic_model(
    dataframe: pd.DataFrame,
) -> tuple[
    Pipeline,
    pd.DataFrame,
]:
    """Fit final descriptive logistic model on all training rows."""

    model = (
        build_logistic_pipeline()
    )

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

    model.fit(
        x,
        y,
    )

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "standardized_coefficient": (
                classifier.coef_[0]
            ),
        }
    )

    coefficients[
        "absolute_standardized_coefficient"
    ] = (
        coefficients[
            "standardized_coefficient"
        ]
        .abs()
    )

    coefficients = (
        coefficients.sort_values(
            by=[
                "absolute_standardized_coefficient",
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
        coefficients,
    )


def validate_outputs(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    """Validate model outputs before saving."""

    if len(oof) != len(
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

    for column in [
        "logistic_score",
        "dummy_score",
    ]:
        if not oof[
            column
        ].between(
            0,
            1,
        ).all():
            raise ValueError(
                f"{column} contains values outside 0-1."
            )

    if len(
        fold_metrics
    ) != N_SPLITS:
        raise ValueError(
            "Unexpected fold-metric row count."
        )

    if len(
        coefficients
    ) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unexpected coefficient count."
        )

    if coefficients[
        "feature"
    ].duplicated().any():
        raise ValueError(
            "Duplicate coefficient feature names."
        )

    coefficient_values = coefficients[
        "standardized_coefficient"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        coefficient_values
    ).all():
        raise ValueError(
            "Non-finite logistic coefficients were found."
        )


def create_pr_curve_plot(
    oof: pd.DataFrame,
) -> None:
    """Plot pooled out-of-fold precision-recall curve."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    logistic_scores = oof[
        "logistic_score"
    ].to_numpy(
        dtype=float
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            logistic_scores,
        )
    )

    average_precision = float(
        average_precision_score(
            y_true,
            logistic_scores,
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
            "Logistic regression "
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
        "Ankara Logistic Baseline - Spatial OOF Precision-Recall"
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


def create_coefficient_plot(
    coefficients: pd.DataFrame,
) -> None:
    """Plot standardized full-data logistic coefficients."""

    plot_data = (
        coefficients.sort_values(
            "standardized_coefficient",
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
            "standardized_coefficient"
        ],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_xlabel(
        "Standardized logistic coefficient"
    )

    axis.set_title(
        "Ankara Logistic Baseline - Standardized Coefficients"
    )

    figure.tight_layout()

    figure.savefig(
        COEFFICIENT_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_outputs(
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    """Save tabular model outputs."""

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

    coefficients.to_csv(
        COEFFICIENT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    """Create Markdown summary of baseline model performance."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    logistic_scores = oof[
        "logistic_score"
    ].to_numpy(
        dtype=float
    )

    dummy_scores = oof[
        "dummy_score"
    ].to_numpy(
        dtype=float
    )

    logistic_metrics = (
        calculate_binary_metrics(
            y_true,
            logistic_scores,
        )
    )

    dummy_metrics = (
        calculate_binary_metrics(
            y_true,
            dummy_scores,
        )
    )

    top_one_percent = (
        calculate_top_fraction_metrics(
            y_true,
            logistic_scores,
            fraction=0.01,
        )
    )

    top_five_percent = (
        calculate_top_fraction_metrics(
            y_true,
            logistic_scores,
            fraction=0.05,
        )
    )

    coefficient_lines = "\n".join(
        (
            f"- `{row.feature}`: "
            f"{row.standardized_coefficient:+.4f}"
        )
        for row in coefficients.head(
            10
        ).itertuples(
            index=False
        )
    )

    fold_lines = "\n".join(
        (
            f"- Fold {int(row.cv_fold)}: "
            f"{int(row.validation_rows):,} validation rows, "
            f"{int(row.validation_positives):,} positives, "
            f"AP {row.logistic_average_precision:.4f}, "
            f"ROC-AUC {row.logistic_roc_auc:.4f}"
        )
        for row in fold_metrics.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Logistic Regression Baseline

## Dataset

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Predictor features: {len(FEATURE_COLUMNS):,}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km

## Model

The baseline is an L2-regularized logistic regression using standardized
road and parking predictors.

`class_weight="balanced"` is used because the existing-station target is
extremely imbalanced.

Charging-derived variables are not used as predictors.

## Spatial Out-of-Fold Performance

- Logistic average precision / PR-AUC: {logistic_metrics["average_precision"]:.6f}
- Dummy average precision: {dummy_metrics["average_precision"]:.6f}
- Logistic ROC-AUC: {logistic_metrics["roc_auc"]:.6f}
- Dummy ROC-AUC: {dummy_metrics["roc_auc"]:.6f}

### Threshold 0.5 Diagnostic

- Precision: {logistic_metrics["precision_at_0_5"]:.6f}
- Recall: {logistic_metrics["recall_at_0_5"]:.6f}
- F1: {logistic_metrics["f1_at_0_5"]:.6f}
- Predicted-positive cells: {int(logistic_metrics["predicted_positive_count_at_0_5"]):,}

Because class weighting changes the effective class distribution seen
during fitting, the logistic scores should not be interpreted as
calibrated real-world station probabilities.

The 0.5 threshold is therefore reported only as a diagnostic.

## Ranking Performance

### Top 1%

- Cells inspected: {int(top_one_percent["selected_count"]):,}
- Existing-station cells recovered: {int(top_one_percent["positive_count"]):,}
- Precision: {top_one_percent["precision"]:.6f}
- Recall: {top_one_percent["recall"]:.6f}
- Lift over prevalence: {top_one_percent["lift"]:.2f}x

### Top 5%

- Cells inspected: {int(top_five_percent["selected_count"]):,}
- Existing-station cells recovered: {int(top_five_percent["positive_count"]):,}
- Precision: {top_five_percent["precision"]:.6f}
- Recall: {top_five_percent["recall"]:.6f}
- Lift over prevalence: {top_five_percent["lift"]:.2f}x

## Fold-Level Results

{fold_lines}

## Largest Standardized Coefficients

{coefficient_lines}

Coefficient magnitude is descriptive rather than causal.

Correlated road and parking variables can redistribute coefficient
magnitude among one another.

## Evaluation Policy

Accuracy is intentionally not used as a primary model metric.

With only {int(dataframe[TARGET_COLUMN].sum()):,} positive cells among
{len(dataframe):,} total rows, a trivial negative classifier would
produce extremely high apparent accuracy while identifying no station
cells.

Primary evaluation focuses on ranking quality and rare-class retrieval:

- average precision / PR-AUC
- precision
- recall
- F1
- top-ranked-cell recall
- lift over prevalence

ROC-AUC is reported as a secondary metric.

## Spatial Validation Limitation

The predefined 5-km folds keep cells within the same spatial block
together.

Adjacent blocks can still belong to different folds, so this procedure
reduces local spatial dependence but does not eliminate every possible
form of spatial autocorrelation.

## Outputs

- `data/processed/ankara_logistic_baseline_oof_predictions.csv`
- `data/processed/ankara_logistic_baseline_fold_metrics.csv`
- `data/processed/ankara_logistic_baseline_coefficients.csv`
- `docs/ankara_logistic_baseline_pr_curve.png`
- `docs/ankara_logistic_baseline_coefficients.png`

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
    """Print key baseline model statistics."""

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    logistic_scores = oof[
        "logistic_score"
    ].to_numpy(
        dtype=float
    )

    dummy_scores = oof[
        "dummy_score"
    ].to_numpy(
        dtype=float
    )

    logistic_metrics = (
        calculate_binary_metrics(
            y_true,
            logistic_scores,
        )
    )

    dummy_metrics = (
        calculate_binary_metrics(
            y_true,
            dummy_scores,
        )
    )

    top_one_percent = (
        calculate_top_fraction_metrics(
            y_true,
            logistic_scores,
            fraction=0.01,
        )
    )

    top_five_percent = (
        calculate_top_fraction_metrics(
            y_true,
            logistic_scores,
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
        f"{int(dataframe[TARGET_COLUMN].sum()):,}",
    )

    print()

    print(
        "Spatial OOF logistic AP:",
        f"{logistic_metrics['average_precision']:.6f}",
    )

    print(
        "Dummy AP:",
        f"{dummy_metrics['average_precision']:.6f}",
    )

    print(
        "Spatial OOF logistic ROC-AUC:",
        f"{logistic_metrics['roc_auc']:.6f}",
    )

    print()

    print(
        "Threshold 0.5 precision:",
        f"{logistic_metrics['precision_at_0_5']:.6f}",
    )

    print(
        "Threshold 0.5 recall:",
        f"{logistic_metrics['recall_at_0_5']:.6f}",
    )

    print(
        "Threshold 0.5 F1:",
        f"{logistic_metrics['f1_at_0_5']:.6f}",
    )

    print()

    print(
        "Top 1% recovered positives:",
        f"{int(top_one_percent['positive_count'])}"
        f"/{int(y_true.sum())}",
    )

    print(
        "Top 1% recall:",
        f"{top_one_percent['recall']:.6f}",
    )

    print(
        "Top 1% lift:",
        f"{top_one_percent['lift']:.2f}x",
    )

    print()

    print(
        "Top 5% recovered positives:",
        f"{int(top_five_percent['positive_count'])}"
        f"/{int(y_true.sum())}",
    )

    print(
        "Top 5% recall:",
        f"{top_five_percent['recall']:.6f}",
    )

    print(
        "Top 5% lift:",
        f"{top_five_percent['lift']:.2f}x",
    )

    print()

    print(
        "Fold metrics:"
    )

    print(
        fold_metrics[
            [
                "cv_fold",
                "validation_rows",
                "validation_positives",
                "logistic_average_precision",
                "logistic_roc_auc",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Train and evaluate Ankara logistic regression baseline."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Logistic Regression Baseline"
    )

    print("=" * 70)

    create_output_directories()

    dataframe = (
        load_inputs()
    )

    oof, fold_metrics = (
        run_spatial_cross_validation(
            dataframe
        )
    )

    _, coefficients = (
        fit_full_logistic_model(
            dataframe
        )
    )

    validate_outputs(
        dataframe,
        oof,
        fold_metrics,
        coefficients,
    )

    save_outputs(
        oof,
        fold_metrics,
        coefficients,
    )

    create_pr_curve_plot(
        oof
    )

    create_coefficient_plot(
        coefficients
    )

    create_summary(
        dataframe,
        oof,
        fold_metrics,
        coefficients,
    )

    print_results(
        dataframe,
        oof,
        fold_metrics,
    )

    print("=" * 70)

    print(
        "Ankara logistic baseline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
