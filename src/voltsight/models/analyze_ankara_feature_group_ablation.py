from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__:
    from .train_ankara_gradient_boosting_baseline import (
        FEATURE_COLUMNS as GRADIENT_BOOSTING_FEATURE_COLUMNS,
        build_model as build_gradient_boosting_model,
        calculate_balanced_sample_weights,
    )
    from .train_ankara_logistic_baseline import (
        FEATURE_COLUMNS as LOGISTIC_FEATURE_COLUMNS,
        N_SPLITS,
        TARGET_COLUMN,
        build_logistic_pipeline,
        calculate_top_fraction_metrics,
        load_inputs,
    )
    from .train_ankara_random_forest_baseline import (
        FEATURE_COLUMNS as RANDOM_FOREST_FEATURE_COLUMNS,
        build_model as build_random_forest_model,
    )
else:
    from train_ankara_gradient_boosting_baseline import (
        FEATURE_COLUMNS as GRADIENT_BOOSTING_FEATURE_COLUMNS,
        build_model as build_gradient_boosting_model,
        calculate_balanced_sample_weights,
    )
    from train_ankara_logistic_baseline import (
        FEATURE_COLUMNS as LOGISTIC_FEATURE_COLUMNS,
        N_SPLITS,
        TARGET_COLUMN,
        build_logistic_pipeline,
        calculate_top_fraction_metrics,
        load_inputs,
    )
    from train_ankara_random_forest_baseline import (
        FEATURE_COLUMNS as RANDOM_FOREST_FEATURE_COLUMNS,
        build_model as build_random_forest_model,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_feature_group_ablation_metrics.csv"
)

FOLD_METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_feature_group_ablation_fold_metrics.csv"
)

OOF_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_feature_group_ablation_oof_predictions.csv"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_feature_group_ablation_ap.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_feature_group_ablation_summary.md"
)

ROAD_FEATURE_COLUMNS = (
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
)

PARKING_FEATURE_COLUMNS = (
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
)

ALL_FEATURE_COLUMNS = (
    *ROAD_FEATURE_COLUMNS,
    *PARKING_FEATURE_COLUMNS,
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "road_only": ROAD_FEATURE_COLUMNS,
    "parking_only": PARKING_FEATURE_COLUMNS,
    "all": ALL_FEATURE_COLUMNS,
}

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "HistGradientBoosting",
}

FEATURE_GROUP_LABELS = {
    "road_only": "Road only",
    "parking_only": "Parking only",
    "all": "Road + parking",
}


ModelBuilder = Callable[[], BaseEstimator]


def create_output_directories() -> None:
    """Create directories required by generated outputs."""

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_feature_definitions() -> None:
    """Ensure all baseline scripts still use the same predictor schema."""

    canonical = tuple(LOGISTIC_FEATURE_COLUMNS)

    if tuple(RANDOM_FOREST_FEATURE_COLUMNS) != canonical:
        raise ValueError(
            "Random Forest feature columns differ from the logistic baseline."
        )

    if tuple(GRADIENT_BOOSTING_FEATURE_COLUMNS) != canonical:
        raise ValueError(
            "Gradient Boosting feature columns differ from the logistic baseline."
        )

    if ALL_FEATURE_COLUMNS != canonical:
        raise ValueError(
            "Road/parking feature groups do not reconstruct the baseline feature list."
        )

    if set(ROAD_FEATURE_COLUMNS) & set(PARKING_FEATURE_COLUMNS):
        raise ValueError(
            "Road and parking feature groups must be disjoint."
        )


def build_ablation_model(model_name: str) -> BaseEstimator:
    """Build one baseline model without changing its fixed configuration."""

    if model_name == "logistic_regression":
        return build_logistic_pipeline()

    if model_name == "random_forest":
        return build_random_forest_model()

    if model_name == "gradient_boosting":
        return build_gradient_boosting_model()

    raise ValueError(
        f"Unknown model name: {model_name}"
    )


def fit_ablation_model(
    model: BaseEstimator,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Fit one baseline model using the same imbalance handling as its baseline."""

    if model_name == "gradient_boosting":
        sample_weights = calculate_balanced_sample_weights(
            y_train
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
        )

        return model

    model.fit(
        x_train,
        y_train,
    )

    return model


def validate_ablation_input(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    """Validate one feature-group input before spatial cross-validation."""

    required_columns = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
        *feature_columns,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Ablation input is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Ablation input is empty."
        )

    if dataframe["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in ablation input."
        )

    if set(
        dataframe[TARGET_COLUMN].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Ablation target must contain both classes."
        )

    expected_folds = set(
        range(N_SPLITS)
    )

    actual_folds = set(
        dataframe["cv_fold"].unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            "Unexpected spatial fold identifiers: "
            f"{sorted(actual_folds)}"
        )

    for feature in feature_columns:
        values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Invalid values found in {feature}."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Non-finite values found in {feature}."
            )


def run_single_configuration(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    feature_group: str,
    feature_columns: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """Evaluate one model/feature-group pair on the fixed spatial folds."""

    validate_ablation_input(
        dataframe,
        feature_columns,
    )

    score_column = (
        f"{model_name}__{feature_group}"
    )

    oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    oof[score_column] = np.nan

    fold_records: list[
        dict[str, float | int | str]
    ] = []

    base_model = build_ablation_model(
        model_name
    )

    feature_list = list(
        feature_columns
    )

    for fold in range(N_SPLITS):
        train_mask = (
            dataframe["cv_fold"] != fold
        )

        validation_mask = (
            dataframe["cv_fold"] == fold
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

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Fold {fold} training data does not contain both classes."
            )

        if len(np.unique(y_validation)) != 2:
            raise ValueError(
                f"Fold {fold} validation data does not contain both classes."
            )

        x_train = train[
            feature_list
        ]

        x_validation = validation[
            feature_list
        ]

        model = clone(
            base_model
        )

        model = fit_ablation_model(
            model,
            model_name,
            x_train,
            y_train,
        )

        scores = model.predict_proba(
            x_validation
        )[
            :,
            1,
        ]

        scores = np.asarray(
            scores,
            dtype=float,
        )

        if not np.isfinite(scores).all():
            raise ValueError(
                f"Non-finite scores produced for {model_name}/{feature_group}."
            )

        oof.loc[
            validation_mask,
            score_column,
        ] = scores

        fold_records.append(
            {
                "model": model_name,
                "feature_group": feature_group,
                "feature_count": len(feature_columns),
                "cv_fold": fold,
                "validation_rows": len(validation),
                "validation_positives": int(y_validation.sum()),
                "average_precision": float(
                    average_precision_score(
                        y_validation,
                        scores,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_validation,
                        scores,
                    )
                ),
            }
        )

    if oof[score_column].isna().any():
        raise ValueError(
            f"Missing OOF scores for {model_name}/{feature_group}."
        )

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    pooled_scores = oof[
        score_column
    ].to_numpy(
        dtype=float
    )

    fold_metrics = pd.DataFrame(
        fold_records
    )

    top_one = calculate_top_fraction_metrics(
        y_true,
        pooled_scores,
        fraction=0.01,
    )

    top_five = calculate_top_fraction_metrics(
        y_true,
        pooled_scores,
        fraction=0.05,
    )

    metrics: dict[
        str,
        float | int | str,
    ] = {
        "model": model_name,
        "feature_group": feature_group,
        "feature_count": len(feature_columns),
        "pooled_average_precision": float(
            average_precision_score(
                y_true,
                pooled_scores,
            )
        ),
        "pooled_roc_auc": float(
            roc_auc_score(
                y_true,
                pooled_scores,
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
            ].std(ddof=1)
        ),
        "mean_fold_roc_auc": float(
            fold_metrics[
                "roc_auc"
            ].mean()
        ),
        "std_fold_roc_auc": float(
            fold_metrics[
                "roc_auc"
            ].std(ddof=1)
        ),
        "top_1_percent_recovered_positives": int(
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
        "top_5_percent_recovered_positives": int(
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

    return (
        oof,
        fold_metrics,
        metrics,
    )


def run_feature_group_ablation(
    dataframe: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    feature_group_names: tuple[str, ...] = tuple(
        FEATURE_GROUPS
    ),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the full model-by-feature-group ablation matrix."""

    validate_feature_definitions()

    unknown_models = (
        set(model_names)
        - set(MODEL_NAMES)
    )

    if unknown_models:
        raise ValueError(
            "Unknown model names: "
            f"{sorted(unknown_models)}"
        )

    unknown_groups = (
        set(feature_group_names)
        - set(FEATURE_GROUPS)
    )

    if unknown_groups:
        raise ValueError(
            "Unknown feature groups: "
            f"{sorted(unknown_groups)}"
        )

    base_oof = dataframe[
        [
            "grid_id",
            TARGET_COLUMN,
            "spatial_block_id",
            "cv_fold",
        ]
    ].copy()

    metrics_records: list[
        dict[str, float | int | str]
    ] = []

    fold_frames: list[
        pd.DataFrame
    ] = []

    for model_name in model_names:
        for feature_group in feature_group_names:
            feature_columns = FEATURE_GROUPS[
                feature_group
            ]

            print(
                "Running:",
                MODEL_LABELS[model_name],
                "/",
                FEATURE_GROUP_LABELS[feature_group],
            )

            configuration_oof, fold_metrics, metrics = (
                run_single_configuration(
                    dataframe,
                    model_name=model_name,
                    feature_group=feature_group,
                    feature_columns=feature_columns,
                )
            )

            score_column = (
                f"{model_name}__{feature_group}"
            )

            base_oof = base_oof.merge(
                configuration_oof[
                    [
                        "grid_id",
                        score_column,
                    ]
                ],
                on="grid_id",
                how="left",
                validate="one_to_one",
            )

            metrics_records.append(
                metrics
            )

            fold_frames.append(
                fold_metrics
            )

    metrics_table = pd.DataFrame(
        metrics_records
    )

    fold_metrics_table = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    validate_outputs(
        dataframe,
        metrics_table,
        fold_metrics_table,
        base_oof,
        model_names=model_names,
        feature_group_names=feature_group_names,
    )

    return (
        metrics_table,
        fold_metrics_table,
        base_oof,
    )


def validate_outputs(
    dataframe: pd.DataFrame,
    metrics_table: pd.DataFrame,
    fold_metrics_table: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    feature_group_names: tuple[str, ...] = tuple(
        FEATURE_GROUPS
    ),
) -> None:
    """Validate ablation results before they are saved."""

    expected_configurations = (
        len(model_names)
        * len(feature_group_names)
    )

    if len(metrics_table) != expected_configurations:
        raise ValueError(
            "Unexpected ablation metrics row count."
        )

    if len(fold_metrics_table) != (
        expected_configurations * N_SPLITS
    ):
        raise ValueError(
            "Unexpected ablation fold-metrics row count."
        )

    if len(oof) != len(dataframe):
        raise ValueError(
            "Ablation OOF row count does not match training data."
        )

    if oof["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in ablation OOF output."
        )

    score_columns = [
        f"{model_name}__{feature_group}"
        for model_name in model_names
        for feature_group in feature_group_names
    ]

    for score_column in score_columns:
        if score_column not in oof.columns:
            raise ValueError(
                f"Missing OOF score column: {score_column}"
            )

        scores = oof[
            score_column
        ]

        if scores.isna().any():
            raise ValueError(
                f"Missing OOF values in {score_column}."
            )

        if not scores.between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"OOF scores outside 0-1 in {score_column}."
            )

    bounded_metric_columns = (
        "pooled_average_precision",
        "pooled_roc_auc",
        "mean_fold_average_precision",
        "mean_fold_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    )

    for column in bounded_metric_columns:
        if not metrics_table[
            column
        ].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"Metric outside 0-1 in {column}."
            )

    numeric_columns = metrics_table.select_dtypes(
        include=[np.number]
    ).columns

    if not np.isfinite(
        metrics_table[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite values found in ablation metrics."
        )


def save_outputs(
    metrics_table: pd.DataFrame,
    fold_metrics_table: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    """Save tabular ablation outputs."""

    metrics_table.sort_values(
        [
            "model",
            "feature_group",
        ],
        kind="stable",
    ).to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_metrics_table.sort_values(
        [
            "model",
            "feature_group",
            "cv_fold",
        ],
        kind="stable",
    ).to_csv(
        FOLD_METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    oof.sort_values(
        "grid_id",
        kind="stable",
    ).to_csv(
        OOF_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_ap_plot(
    metrics_table: pd.DataFrame,
) -> None:
    """Plot pooled average precision for all ablation configurations."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    model_positions = np.arange(
        len(MODEL_NAMES),
        dtype=float,
    )

    width = 0.24

    for group_index, feature_group in enumerate(
        FEATURE_GROUPS
    ):
        group_metrics = (
            metrics_table.loc[
                metrics_table[
                    "feature_group"
                ] == feature_group
            ]
            .set_index(
                "model"
            )
            .reindex(
                MODEL_NAMES
            )
        )

        offsets = (
            group_index
            - 1
        ) * width

        axis.bar(
            model_positions + offsets,
            group_metrics[
                "pooled_average_precision"
            ].to_numpy(
                dtype=float
            ),
            width=width,
            label=FEATURE_GROUP_LABELS[
                feature_group
            ],
        )

    axis.set_xticks(
        model_positions
    )

    axis.set_xticklabels(
        [
            MODEL_LABELS[
                model_name
            ]
            for model_name in MODEL_NAMES
        ]
    )

    axis.set_ylabel(
        "Spatial OOF average precision"
    )

    axis.set_title(
        "Ankara Feature-Group Ablation - Pooled Spatial OOF AP"
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


def create_markdown_table(
    metrics_table: pd.DataFrame,
) -> str:
    """Create a compact Markdown comparison table without extra packages."""

    header = (
        "| Model | Features | Pooled AP | Mean fold AP | Fold AP std | "
        "ROC-AUC | Top 1% recall | Top 5% recall |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )

    rows = [
        header
    ]

    ordered = metrics_table.copy()

    ordered["model_order"] = ordered[
        "model"
    ].map(
        {
            model: index
            for index, model in enumerate(
                MODEL_NAMES
            )
        }
    )

    ordered["group_order"] = ordered[
        "feature_group"
    ].map(
        {
            group: index
            for index, group in enumerate(
                FEATURE_GROUPS
            )
        }
    )

    ordered = ordered.sort_values(
        [
            "model_order",
            "group_order",
        ],
        kind="stable",
    )

    for row in ordered.itertuples(
        index=False
    ):
        rows.append(
            "| "
            f"{MODEL_LABELS[row.model]} | "
            f"{FEATURE_GROUP_LABELS[row.feature_group]} | "
            f"{row.pooled_average_precision:.6f} | "
            f"{row.mean_fold_average_precision:.6f} | "
            f"{row.std_fold_average_precision:.6f} | "
            f"{row.pooled_roc_auc:.6f} | "
            f"{row.top_1_percent_recall:.6f} | "
            f"{row.top_5_percent_recall:.6f} |"
        )

    return "\n".join(
        rows
    )


def create_summary(
    dataframe: pd.DataFrame,
    metrics_table: pd.DataFrame,
) -> None:
    """Create Markdown documentation for the ablation experiment."""

    comparison_table = create_markdown_table(
        metrics_table
    )

    all_rows = metrics_table.loc[
        metrics_table[
            "feature_group"
        ] == "all"
    ].set_index(
        "model"
    )

    interpretation_lines: list[str] = []

    for model_name in MODEL_NAMES:
        model_rows = metrics_table.loc[
            metrics_table[
                "model"
            ] == model_name
        ].set_index(
            "feature_group"
        )

        all_ap = float(
            all_rows.loc[
                model_name,
                "pooled_average_precision",
            ]
        )

        road_ap = float(
            model_rows.loc[
                "road_only",
                "pooled_average_precision",
            ]
        )

        parking_ap = float(
            model_rows.loc[
                "parking_only",
                "pooled_average_precision",
            ]
        )

        interpretation_lines.append(
            "- "
            f"{MODEL_LABELS[model_name]}: "
            f"road-only AP {road_ap:.6f}, "
            f"parking-only AP {parking_ap:.6f}, "
            f"combined AP {all_ap:.6f}."
        )

    summary = f"""# Ankara Feature-Group Ablation

## Purpose

This experiment isolates the contribution of road and parking predictor
families while preserving the existing Ankara baseline model settings and
the predefined 5-km spatial cross-validation folds.

No hyperparameter search is performed.

## Dataset

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Spatial folds: {N_SPLITS}
- Road predictors: {len(ROAD_FEATURE_COLUMNS)}
- Parking predictors: {len(PARKING_FEATURE_COLUMNS)}
- Combined predictors: {len(ALL_FEATURE_COLUMNS)}

## Feature Groups

### Road only

{chr(10).join(f'- `{feature}`' for feature in ROAD_FEATURE_COLUMNS)}

### Parking only

{chr(10).join(f'- `{feature}`' for feature in PARKING_FEATURE_COLUMNS)}

## Models

The experiment reuses the fixed baseline configurations for:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Class-imbalance handling is also preserved. Logistic Regression and Random
Forest retain their class-weight settings, while HistGradientBoosting uses
the same balanced sample-weight calculation as its baseline.

## Spatial OOF Results

{comparison_table}

## Model-Level Feature-Group Comparison

{chr(10).join(interpretation_lines)}

These comparisons are descriptive ablations, not causal estimates of the
real-world effect of roads or parking on charging-station placement.

A lower score after removing a feature family indicates that the model's
ranking performance depends on information in that family under the current
dataset and validation design. Correlated predictors and sparse OSM coverage
can affect the magnitude of the observed differences.

## Evaluation Policy

Primary emphasis remains on rare-class ranking quality:

- pooled average precision / PR-AUC
- mean and standard deviation of fold AP
- top-1-percent recall
- top-5-percent recall

ROC-AUC is reported as a secondary metric.

Accuracy is intentionally not used as a primary metric.

## Spatial Validation Limitation

The same predefined 5-km spatial block folds are reused for every model and
feature group, making the comparisons directly paired by validation fold.

Cells inside one block stay together, but neighboring blocks can still be
assigned to different folds. The procedure therefore reduces local spatial
dependence without claiming to eliminate all spatial autocorrelation.

## Outputs

- `data/processed/ankara_feature_group_ablation_metrics.csv`
- `data/processed/ankara_feature_group_ablation_fold_metrics.csv`
- `data/processed/ankara_feature_group_ablation_oof_predictions.csv`
- `docs/ankara_feature_group_ablation_ap.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    metrics_table: pd.DataFrame,
) -> None:
    """Print the compact ablation result table."""

    display_columns = [
        "model",
        "feature_group",
        "pooled_average_precision",
        "mean_fold_average_precision",
        "std_fold_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    ]

    print("-" * 100)

    print(
        metrics_table[
            display_columns
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Run Ankara road-versus-parking feature-group ablation."""

    print("=" * 100)
    print(
        "VoltSight - Ankara Feature-Group Ablation"
    )
    print("=" * 100)

    create_output_directories()
    validate_feature_definitions()

    dataframe = load_inputs()

    metrics_table, fold_metrics_table, oof = (
        run_feature_group_ablation(
            dataframe
        )
    )

    save_outputs(
        metrics_table,
        fold_metrics_table,
        oof,
    )

    create_ap_plot(
        metrics_table
    )

    create_summary(
        dataframe,
        metrics_table,
    )

    print_results(
        metrics_table
    )

    print("=" * 100)
    print(
        "Ankara feature-group ablation completed successfully."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
