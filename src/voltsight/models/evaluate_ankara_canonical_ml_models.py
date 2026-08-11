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
    roc_auc_score,
)

from voltsight.core.ankara_ml_features import (
    CANONICAL_ML_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_architecture,
)
from voltsight.models.train_ankara_gradient_boosting_baseline import (
    build_model as build_gradient_boosting_model,
    calculate_balanced_sample_weights,
)
from voltsight.models.train_ankara_logistic_baseline import (
    N_SPLITS,
    build_logistic_pipeline,
    calculate_top_fraction_metrics,
)
from voltsight.models.train_ankara_random_forest_baseline import (
    build_model as build_random_forest_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_training_dataset.csv"
)

FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_cv_folds.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_model_metrics.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_model_fold_metrics.csv"
)

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_model_oof_predictions.csv"
)

PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_canonical_ml_model_comparison.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_canonical_ml_model_summary.md"
)

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}


def validate_training_frame(
    training: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the canonical-15 training schema and values."""

    validate_feature_architecture()

    required = {
        "grid_id",
        *CANONICAL_ML_FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    missing = required - set(
        training.columns
    )

    if missing:
        raise ValueError(
            "Canonical training columns are missing: "
            f"{sorted(missing)}"
        )

    result = training[
        [
            "grid_id",
            *CANONICAL_ML_FEATURE_COLUMNS,
            TARGET_COLUMN,
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
            "Canonical training contains duplicate grid IDs."
        )

    for feature in CANONICAL_ML_FEATURE_COLUMNS:
        result[
            feature
        ] = pd.to_numeric(
            result[
                feature
            ],
            errors="coerce",
        )

        values = result[
            feature
        ].to_numpy(
            dtype=float
        )

        if (
            result[
                feature
            ].isna().any()
            or not np.isfinite(
                values
            ).all()
        ):
            raise ValueError(
                f"Invalid values found in {feature}."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"Negative values found in {feature}."
            )

    target = pd.to_numeric(
        result[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "Canonical target contains invalid values."
        )

    result[
        TARGET_COLUMN
    ] = target.astype(
        int
    )

    if set(
        result[
            TARGET_COLUMN
        ].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Canonical target must contain both classes."
        )

    return result


def validate_fold_frame(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the fixed five-fold spatial block assignment."""

    required = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
    }

    missing = required - set(
        folds.columns
    )

    if missing:
        raise ValueError(
            "Spatial fold columns are missing: "
            f"{sorted(missing)}"
        )

    result = folds[
        [
            "grid_id",
            "spatial_block_id",
            "cv_fold",
            TARGET_COLUMN,
        ]
    ].copy()

    result[
        "grid_id"
    ] = result[
        "grid_id"
    ].astype(str)

    result[
        "spatial_block_id"
    ] = result[
        "spatial_block_id"
    ].astype(str)

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Spatial folds contain duplicate grid IDs."
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

    if set(
        result[
            "cv_fold"
        ].unique()
    ) != set(
        range(
            N_SPLITS
        )
    ):
        raise ValueError(
            "Unexpected spatial fold identifiers."
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

    return result


def attach_spatial_folds(
    training: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the fixed spatial folds and verify target identity."""

    training = validate_training_frame(
        training
    )

    folds = validate_fold_frame(
        folds
    )

    merged = training.merge(
        folds.rename(
            columns={
                TARGET_COLUMN: (
                    "fold_target"
                ),
            }
        ),
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        training
    ):
        raise ValueError(
            "Spatial fold merge changed the canonical row count."
        )

    if merged[
        "cv_fold"
    ].isna().any():
        raise ValueError(
            "Not every canonical row received a spatial fold."
        )

    if not np.array_equal(
        merged[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        ),
        merged[
            "fold_target"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise ValueError(
            "Canonical training and spatial-fold targets do not match."
        )

    return merged.drop(
        columns=[
            "fold_target",
        ]
    )


def load_inputs() -> pd.DataFrame:
    """Load canonical training data and fixed spatial folds."""

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Canonical training dataset not found: {TRAINING_PATH}"
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

    return attach_spatial_folds(
        training,
        folds,
    )


def build_model(
    model_name: str,
):
    """Build one unchanged historical estimator for canonical-15 evaluation."""

    if model_name == "logistic_regression":
        return build_logistic_pipeline()

    if model_name == "random_forest":
        return build_random_forest_model()

    if model_name == "hist_gradient_boosting":
        return build_gradient_boosting_model()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def fit_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    """Fit one estimator using its existing imbalance treatment."""

    if model_name == "hist_gradient_boosting":
        model.fit(
            x_train,
            y_train,
            sample_weight=(
                calculate_balanced_sample_weights(
                    y_train
                )
            ),
        )

        return

    model.fit(
        x_train,
        y_train,
    )


def calculate_ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[
    str,
    float,
]:
    """Calculate rare-class ranking metrics used throughout VoltSight."""

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
            "Both target classes are required."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Prediction scores contain non-finite values."
        )

    top_one = calculate_top_fraction_metrics(
        y_true,
        scores,
        fraction=0.01,
    )

    top_five = calculate_top_fraction_metrics(
        y_true,
        scores,
        fraction=0.05,
    )

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
        "top_1_percent_recall": float(
            top_one[
                "recall"
            ]
        ),
        "top_5_percent_recall": float(
            top_five[
                "recall"
            ]
        ),
        "top_1_percent_lift": float(
            top_one[
                "lift"
            ]
        ),
        "top_5_percent_lift": float(
            top_five[
                "lift"
            ]
        ),
    }


def run_model(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[
        str,
        float | int | str,
    ],
]:
    """Run one complete five-fold spatial OOF canonical evaluation."""

    score_column = (
        f"{model_name}_score"
    )

    oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    oof[
        score_column
    ] = np.nan

    fold_records: list[
        dict[
            str,
            float | int | str,
        ]
    ] = []

    feature_columns = list(
        CANONICAL_ML_FEATURE_COLUMNS
    )

    for fold in range(
        N_SPLITS
    ):
        train_mask = (
            dataframe[
                "cv_fold"
            ]
            != fold
        )

        validation_mask = (
            dataframe[
                "cv_fold"
            ]
            == fold
        )

        train = dataframe.loc[
            train_mask
        ]

        validation = dataframe.loc[
            validation_mask
        ]

        y_train = train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        )

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
                f"Fold {fold} training data lacks both classes."
            )

        if len(
            np.unique(
                y_validation
            )
        ) != 2:
            raise ValueError(
                f"Fold {fold} validation data lacks both classes."
            )

        model = build_model(
            model_name
        )

        fit_model(
            model_name,
            model,
            train[
                feature_columns
            ],
            y_train,
        )

        scores = model.predict_proba(
            validation[
                feature_columns
            ]
        )[
            :,
            1,
        ]

        oof.loc[
            validation_mask,
            score_column,
        ] = scores

        fold_metrics = (
            calculate_ranking_metrics(
                y_validation,
                scores,
            )
        )

        fold_records.append(
            {
                "model": model_name,
                "model_label": (
                    MODEL_LABELS[
                        model_name
                    ]
                ),
                "cv_fold": fold,
                "training_rows": len(
                    train
                ),
                "training_positives": int(
                    y_train.sum()
                ),
                "validation_rows": len(
                    validation
                ),
                "validation_positives": int(
                    y_validation.sum()
                ),
                **fold_metrics,
            }
        )

    if oof[
        score_column
    ].isna().any():
        raise ValueError(
            f"{model_name} OOF predictions are incomplete."
        )

    pooled = calculate_ranking_metrics(
        oof[
            TARGET_COLUMN
        ].to_numpy(
            dtype=int
        ),
        oof[
            score_column
        ].to_numpy(
            dtype=float
        ),
    )

    fold_frame = pd.DataFrame(
        fold_records
    )

    aggregate = {
        "model": model_name,
        "model_label": (
            MODEL_LABELS[
                model_name
            ]
        ),
        "feature_count": len(
            CANONICAL_ML_FEATURE_COLUMNS
        ),
        "pooled_average_precision": (
            pooled[
                "average_precision"
            ]
        ),
        "mean_fold_average_precision": float(
            fold_frame[
                "average_precision"
            ].mean()
        ),
        "std_fold_average_precision": float(
            fold_frame[
                "average_precision"
            ].std(
                ddof=1
            )
        ),
        "pooled_roc_auc": (
            pooled[
                "roc_auc"
            ]
        ),
        "mean_fold_roc_auc": float(
            fold_frame[
                "roc_auc"
            ].mean()
        ),
        "std_fold_roc_auc": float(
            fold_frame[
                "roc_auc"
            ].std(
                ddof=1
            )
        ),
        "top_1_percent_recall": (
            pooled[
                "top_1_percent_recall"
            ]
        ),
        "top_5_percent_recall": (
            pooled[
                "top_5_percent_recall"
            ]
        ),
        "top_1_percent_lift": (
            pooled[
                "top_1_percent_lift"
            ]
        ),
        "top_5_percent_lift": (
            pooled[
                "top_5_percent_lift"
            ]
        ),
    }

    return (
        oof,
        fold_frame,
        aggregate,
    )


def run_evaluation(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate all canonical-15 estimators."""

    metric_records = []
    fold_frames = []
    oof_frames = []

    for model_name in MODEL_ORDER:
        print(
            f"Running {MODEL_LABELS[model_name]}..."
        )

        (
            oof,
            fold_metrics,
            aggregate,
        ) = run_model(
            dataframe,
            model_name=model_name,
        )

        metric_records.append(
            aggregate
        )

        fold_frames.append(
            fold_metrics
        )

        oof_frames.append(
            oof
        )

    metrics = pd.DataFrame(
        metric_records
    )

    fold_metrics = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    merged_oof = oof_frames[
        0
    ].copy()

    for frame in oof_frames[
        1:
    ]:
        score_columns = [
            column
            for column in frame.columns
            if column.endswith(
                "_score"
            )
        ]

        merged_oof = merged_oof.merge(
            frame[
                [
                    "grid_id",
                    *score_columns,
                ]
            ],
            on="grid_id",
            how="inner",
            validate="one_to_one",
        )

    return (
        metrics,
        fold_metrics,
        merged_oof,
    )


def save_outputs(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    """Save canonical evaluation tables."""

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_metrics.to_csv(
        FOLD_METRICS_PATH,
        index=False,
        encoding="utf-8",
    )

    oof.sort_values(
        "grid_id",
        kind="stable",
    ).to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8",
    )


def create_plot(
    metrics: pd.DataFrame,
) -> None:
    """Plot pooled AP for the three canonical models."""

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = (
        metrics.set_index(
            "model"
        )
        .loc[
            list(
                MODEL_ORDER
            )
        ]
        .reset_index()
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    axis.bar(
        ordered[
            "model_label"
        ],
        ordered[
            "pooled_average_precision"
        ],
    )

    axis.set_ylabel(
        "Pooled spatial OOF average precision"
    )

    axis.set_title(
        "Ankara Canonical Activity15 ML Evaluation"
    )

    figure.tight_layout()

    figure.savefig(
        PLOT_PATH,
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
    """Write the canonical-15 ML evaluation summary."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_lines = [
        "| Model | Pooled AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Top 5% recall | Top 1% lift | Top 5% lift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name in MODEL_ORDER:
        row = metrics.loc[
            metrics[
                "model"
            ]
            == model_name
        ].iloc[
            0
        ]

        table_lines.append(
            "| "
            f"{row['model_label']} | "
            f"{row['pooled_average_precision']:.6f} | "
            f"{row['mean_fold_average_precision']:.6f} | "
            f"{row['std_fold_average_precision']:.6f} | "
            f"{row['pooled_roc_auc']:.6f} | "
            f"{row['top_1_percent_recall']:.6f} | "
            f"{row['top_5_percent_recall']:.6f} | "
            f"{row['top_1_percent_lift']:.2f}x | "
            f"{row['top_5_percent_lift']:.2f}x |"
        )

    feature_lines = "\n".join(
        f"- `{feature}`"
        for feature in CANONICAL_ML_FEATURE_COLUMNS
    )

    summary = f"""# Ankara Canonical Activity15 ML Evaluation

## Purpose

This evaluation establishes the forward-looking canonical Ankara ML reference
using the 15-feature predictor architecture selected after redundancy, population,
and OSM activity experiments.

Historical Full14 and Normalized12 experiments remain historical references.

## Dataset

- Rows: {len(dataframe):,}
- Positive existing-station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Predictors: {len(CANONICAL_ML_FEATURE_COLUMNS)}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km

## Canonical Predictors

{feature_lines}

## Models

The existing untuned model configurations are reused unchanged:

- class-weighted standardized Logistic Regression
- Random Forest with 400 trees, depth 12, minimum leaf size 5,
  `max_features="sqrt"`, and `balanced_subsample`
- HistGradientBoosting with learning rate 0.05, 150 iterations, 15 leaf nodes,
  minimum leaf size 100, L2 regularization 1.0, balanced sample weights, and
  internal early stopping disabled

No hyperparameter search is performed.

## Spatial OOF Results

{chr(10).join(table_lines)}

## Evaluation Policy

Average precision is primary because only a very small fraction of Ankara cells
contain known existing charging stations.

Top-1% and top-5% recall/lift remain decision-relevant because VoltSight is a
candidate-ranking system rather than a conventional balanced classifier.

The model scores are predictive ranking signals, not causal effects and not
calibrated probabilities.

The 5-km spatial block design reduces local train-validation dependence but
does not eliminate all spatial autocorrelation.

Only 46 positive cells are available, so fold-level variability remains an
important limitation.

OSM total-activity features are mapped urban-activity proxies. They do not
directly observe EV ownership, traffic, employment, trips, electricity-grid
capacity, or future charging demand.

## Historical Compatibility

Existing Full14 baseline outputs are not overwritten by this evaluation.

This script writes dedicated canonical-15 outputs so historical comparisons
remain reproducible.

## Outputs

- `data/processed/{METRICS_PATH.name}`
- `data/processed/{FOLD_METRICS_PATH.name}`
- `data/processed/{OOF_PATH.name}`
- `docs/{PLOT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    metrics: pd.DataFrame,
) -> None:
    """Print the main canonical evaluation metrics."""

    print(
        metrics[
            [
                "model_label",
                "feature_count",
                "pooled_average_precision",
                "mean_fold_average_precision",
                "std_fold_average_precision",
                "pooled_roc_auc",
                "top_1_percent_recall",
                "top_5_percent_recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )


def main() -> None:
    """Run the Ankara canonical Activity15 ML evaluation."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Canonical Activity15 ML Evaluation"
    )

    print(
        "="
        * 70
    )

    dataframe = load_inputs()

    (
        metrics,
        fold_metrics,
        oof,
    ) = run_evaluation(
        dataframe
    )

    save_outputs(
        metrics,
        fold_metrics,
        oof,
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

    print(
        "="
        * 70
    )

    print(
        "Ankara canonical Activity15 ML evaluation completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
