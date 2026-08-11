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

from voltsight.models.train_ankara_gradient_boosting_baseline import (
    build_model as build_gradient_boosting_model,
    calculate_balanced_sample_weights,
)
from voltsight.models.train_ankara_logistic_baseline import (
    FEATURE_COLUMNS as BASELINE_FEATURE_COLUMNS,
    N_SPLITS,
    TARGET_COLUMN,
    build_logistic_pipeline,
    calculate_top_fraction_metrics,
)
from voltsight.models.train_ankara_random_forest_baseline import (
    build_model as build_random_forest_model,
    load_inputs as load_baseline_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ACTIVITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_activity_features.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_incremental_value_metrics.csv"
)

FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_incremental_value_fold_metrics.csv"
)

OOF_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_incremental_value_oof_predictions.csv"
)

PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_activity_incremental_value.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_activity_incremental_value_summary.md"
)


NORMALIZED_BASE_FEATURES = tuple(
    feature
    for feature in BASELINE_FEATURE_COLUMNS
    if feature
    not in {
        "road_length_m",
        "parking_area_m2",
    }
)

ACTIVITY_TOTAL_COLUMNS = (
    "poi_count",
    "poi_count_within_1000m",
    "poi_count_within_2000m",
)

FEATURE_SETS = {
    "normalized_12": (
        NORMALIZED_BASE_FEATURES
    ),
    "normalized_12_plus_local_activity": (
        NORMALIZED_BASE_FEATURES
        + (
            "poi_count",
        )
    ),
    "normalized_12_plus_activity_context": (
        NORMALIZED_BASE_FEATURES
        + ACTIVITY_TOTAL_COLUMNS
    ),
}

FEATURE_SET_LABELS = {
    "normalized_12": (
        "Normalized 12"
    ),
    "normalized_12_plus_local_activity": (
        "Normalized 12 + local POI activity"
    ),
    "normalized_12_plus_activity_context": (
        "Normalized 12 + local + 1 km + 2 km POI activity"
    ),
}

MODEL_ORDER = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)

MODEL_LABELS = {
    "logistic_regression": (
        "Logistic Regression"
    ),
    "random_forest": (
        "Random Forest"
    ),
    "hist_gradient_boosting": (
        "HistGradientBoosting"
    ),
}


def validate_feature_sets() -> None:
    """Validate the fixed target-agnostic activity feature-set design."""

    if len(
        NORMALIZED_BASE_FEATURES
    ) != 12:
        raise ValueError(
            "Expected 12 normalized baseline predictors."
        )

    expected_counts = {
        "normalized_12": 12,
        "normalized_12_plus_local_activity": 13,
        "normalized_12_plus_activity_context": 15,
    }

    for (
        name,
        expected_count,
    ) in expected_counts.items():
        if len(
            FEATURE_SETS[
                name
            ]
        ) != expected_count:
            raise ValueError(
                f"{name} has an unexpected predictor count."
            )

    if (
        FEATURE_SETS[
            "normalized_12_plus_activity_context"
        ][
            -3:
        ]
        != ACTIVITY_TOTAL_COLUMNS
    ):
        raise ValueError(
            "Activity-context feature order changed unexpectedly."
        )


def validate_activity_frame(
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the three total-count activity predictors used in this test."""

    required = {
        "grid_id",
        *ACTIVITY_TOTAL_COLUMNS,
    }

    missing = (
        required
        - set(
            activity.columns
        )
    )

    if missing:
        raise ValueError(
            "Activity feature columns are missing: "
            f"{sorted(missing)}"
        )

    result = activity[
        [
            "grid_id",
            *ACTIVITY_TOTAL_COLUMNS,
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
            "Duplicate activity grid IDs were found."
        )

    for column in ACTIVITY_TOTAL_COLUMNS:
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
                f"Invalid activity values found in {column}."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"Negative activity values found in {column}."
            )

    local = result[
        "poi_count"
    ].to_numpy(
        dtype=float
    )

    within_1km = result[
        "poi_count_within_1000m"
    ].to_numpy(
        dtype=float
    )

    within_2km = result[
        "poi_count_within_2000m"
    ].to_numpy(
        dtype=float
    )

    if (
        within_1km
        < local
    ).any():
        raise ValueError(
            "1-km POI activity cannot be below local activity."
        )

    if (
        within_2km
        < within_1km
    ).any():
        raise ValueError(
            "2-km POI activity cannot be below 1-km activity."
        )

    return result


def attach_activity_features(
    baseline: pd.DataFrame,
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach activity totals to the existing spatial-CV baseline frame."""

    if baseline[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate baseline grid IDs were found."
        )

    activity = validate_activity_frame(
        activity
    )

    baseline = baseline.copy()

    baseline[
        "grid_id"
    ] = baseline[
        "grid_id"
    ].astype(str)

    merged = baseline.merge(
        activity,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        baseline
    ):
        raise ValueError(
            "Activity merge changed the baseline row count."
        )

    if merged[
        list(
            ACTIVITY_TOTAL_COLUMNS
        )
    ].isna().any().any():
        raise ValueError(
            "Not every baseline row matched activity features."
        )

    return merged


def load_analysis_frame() -> pd.DataFrame:
    """Load the fixed baseline folds and attach total activity features."""

    validate_feature_sets()

    if not ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            "Activity feature dataset not found: "
            f"{ACTIVITY_PATH}"
        )

    baseline = load_baseline_inputs()

    activity = pd.read_csv(
        ACTIVITY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    return attach_activity_features(
        baseline,
        activity,
    )


def build_model(
    model_name: str,
):
    """Build one unchanged historical baseline estimator."""

    if (
        model_name
        == "logistic_regression"
    ):
        return (
            build_logistic_pipeline()
        )

    if (
        model_name
        == "random_forest"
    ):
        return (
            build_random_forest_model()
        )

    if (
        model_name
        == "hist_gradient_boosting"
    ):
        return (
            build_gradient_boosting_model()
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def fit_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    """Fit one model using its existing class-imbalance treatment."""

    if (
        model_name
        == "hist_gradient_boosting"
    ):
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
    """Calculate ranking metrics for the extreme rare-class target."""

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
            "Both target classes are required for evaluation."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Prediction scores contain non-finite values."
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
    }


def run_configuration(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    feature_set_name: str,
    feature_columns: tuple[
        str,
        ...,
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[
        str,
        float | int | str,
    ],
]:
    """Run one model/feature-set combination on the fixed spatial folds."""

    missing = (
        set(
            feature_columns
        )
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Experiment predictors are missing: "
            f"{sorted(missing)}"
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
        "score"
    ] = np.nan

    fold_records: list[
        dict[
            str,
            float | int | str,
        ]
    ] = []

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

        if (
            train.empty
            or validation.empty
        ):
            raise ValueError(
                f"Fold {fold} is empty."
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

        if (
            len(
                np.unique(
                    y_train
                )
            )
            != 2
            or len(
                np.unique(
                    y_validation
                )
            )
            != 2
        ):
            raise ValueError(
                f"Fold {fold} does not contain both classes."
            )

        model = build_model(
            model_name
        )

        fit_model(
            model_name,
            model,
            train[
                list(
                    feature_columns
                )
            ],
            y_train,
        )

        scores = model.predict_proba(
            validation[
                list(
                    feature_columns
                )
            ]
        )[
            :,
            1,
        ]

        if not np.isfinite(
            scores
        ).all():
            raise ValueError(
                f"{model_name} fold {fold} produced invalid scores."
            )

        oof.loc[
            validation_mask,
            "score",
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
                "feature_set": (
                    feature_set_name
                ),
                "feature_set_label": (
                    FEATURE_SET_LABELS[
                        feature_set_name
                    ]
                ),
                "feature_count": len(
                    feature_columns
                ),
                "cv_fold": fold,
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
        "score"
    ].isna().any():
        raise ValueError(
            "OOF predictions are incomplete."
        )

    y_true = oof[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    scores = oof[
        "score"
    ].to_numpy(
        dtype=float
    )

    pooled = (
        calculate_ranking_metrics(
            y_true,
            scores,
        )
    )

    fold_metrics_frame = (
        pd.DataFrame(
            fold_records
        )
    )

    aggregate: dict[
        str,
        float | int | str,
    ] = {
        "model": model_name,
        "model_label": (
            MODEL_LABELS[
                model_name
            ]
        ),
        "feature_set": (
            feature_set_name
        ),
        "feature_set_label": (
            FEATURE_SET_LABELS[
                feature_set_name
            ]
        ),
        "feature_count": len(
            feature_columns
        ),
        "pooled_average_precision": (
            pooled[
                "average_precision"
            ]
        ),
        "mean_fold_average_precision": float(
            fold_metrics_frame[
                "average_precision"
            ].mean()
        ),
        "std_fold_average_precision": float(
            fold_metrics_frame[
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
            fold_metrics_frame[
                "roc_auc"
            ].mean()
        ),
        "std_fold_roc_auc": float(
            fold_metrics_frame[
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
    }

    oof[
        "model"
    ] = model_name

    oof[
        "feature_set"
    ] = feature_set_name

    return (
        oof,
        fold_metrics_frame,
        aggregate,
    )


def add_incremental_deltas(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-model deltas against the normalized-12 baseline."""

    result = metrics.copy()

    columns = (
        "pooled_average_precision",
        "mean_fold_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    )

    for column in columns:
        result[
            f"delta_{column}"
        ] = np.nan

    for model_name in MODEL_ORDER:
        model_mask = (
            result[
                "model"
            ]
            == model_name
        )

        baseline_rows = result.loc[
            model_mask
            & (
                result[
                    "feature_set"
                ]
                == "normalized_12"
            )
        ]

        if len(
            baseline_rows
        ) != 1:
            raise ValueError(
                "Expected one normalized-12 baseline for "
                f"{model_name}."
            )

        baseline = (
            baseline_rows.iloc[
                0
            ]
        )

        for column in columns:
            result.loc[
                model_mask,
                f"delta_{column}",
            ] = (
                result.loc[
                    model_mask,
                    column,
                ]
                - float(
                    baseline[
                        column
                    ]
                )
            )

    return result


def run_experiment(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run all fixed activity incremental-value configurations."""

    aggregates = []

    fold_frames = []

    oof_frames = []

    for model_name in MODEL_ORDER:
        print(
            f"Running {MODEL_LABELS[model_name]}..."
        )

        for (
            feature_set_name,
            feature_columns,
        ) in FEATURE_SETS.items():
            print(
                "  "
                f"{FEATURE_SET_LABELS[feature_set_name]}"
            )

            (
                oof,
                fold_metrics,
                aggregate,
            ) = run_configuration(
                dataframe,
                model_name=model_name,
                feature_set_name=(
                    feature_set_name
                ),
                feature_columns=(
                    feature_columns
                ),
            )

            aggregates.append(
                aggregate
            )

            fold_frames.append(
                fold_metrics
            )

            oof_frames.append(
                oof
            )

    metrics = (
        add_incremental_deltas(
            pd.DataFrame(
                aggregates
            )
        )
    )

    fold_metrics = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    oof = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    return (
        metrics,
        fold_metrics,
        oof,
    )


def create_plot(
    metrics: pd.DataFrame,
) -> None:
    """Plot pooled spatial OOF AP for every fixed feature set."""

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            11,
            6.5,
        )
    )

    x = np.arange(
        len(
            MODEL_ORDER
        )
    )

    width = 0.24

    for index, feature_set_name in enumerate(
        FEATURE_SETS
    ):
        values = []

        for model_name in MODEL_ORDER:
            row = metrics.loc[
                (
                    metrics[
                        "model"
                    ]
                    == model_name
                )
                & (
                    metrics[
                        "feature_set"
                    ]
                    == feature_set_name
                )
            ]

            if len(
                row
            ) != 1:
                raise ValueError(
                    "Unexpected metric row count while plotting."
                )

            values.append(
                float(
                    row.iloc[
                        0
                    ][
                        "pooled_average_precision"
                    ]
                )
            )

        axis.bar(
            x
            + (
                index
                - 1
            )
            * width,
            values,
            width=width,
            label=(
                FEATURE_SET_LABELS[
                    feature_set_name
                ]
            ),
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            MODEL_LABELS[
                name
            ]
            for name in MODEL_ORDER
        ]
    )

    axis.set_ylabel(
        "Pooled spatial OOF average precision"
    )

    axis.set_title(
        "Ankara OSM Activity Incremental Value"
    )

    axis.legend()

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
    """Write the activity incremental-value experiment summary."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_lines = [
        "| Model | Feature set | Features | Pooled AP | Delta AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Delta top 1% | Top 5% recall | Delta top 5% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name in MODEL_ORDER:
        for feature_set_name in FEATURE_SETS:
            row = metrics.loc[
                (
                    metrics[
                        "model"
                    ]
                    == model_name
                )
                & (
                    metrics[
                        "feature_set"
                    ]
                    == feature_set_name
                )
            ].iloc[
                0
            ]

            table_lines.append(
                "| "
                f"{row['model_label']} | "
                f"{row['feature_set_label']} | "
                f"{int(row['feature_count'])} | "
                f"{row['pooled_average_precision']:.6f} | "
                f"{row['delta_pooled_average_precision']:+.6f} | "
                f"{row['mean_fold_average_precision']:.6f} | "
                f"{row['std_fold_average_precision']:.6f} | "
                f"{row['pooled_roc_auc']:.6f} | "
                f"{row['top_1_percent_recall']:.6f} | "
                f"{row['delta_top_1_percent_recall']:+.6f} | "
                f"{row['top_5_percent_recall']:.6f} | "
                f"{row['delta_top_5_percent_recall']:+.6f} |"
            )

    delta_lines = []

    for model_name in MODEL_ORDER:
        row = metrics.loc[
            (
                metrics[
                    "model"
                ]
                == model_name
            )
            & (
                metrics[
                    "feature_set"
                ]
                == "normalized_12_plus_activity_context"
            )
        ].iloc[
            0
        ]

        delta_lines.append(
            f"- {MODEL_LABELS[model_name]}: "
            "pooled AP delta "
            f"{row['delta_pooled_average_precision']:+.6f}, "
            "top-1% recall delta "
            f"{row['delta_top_1_percent_recall']:+.6f}, "
            "top-5% recall delta "
            f"{row['delta_top_5_percent_recall']:+.6f}."
        )

    summary = f"""# Ankara OSM Activity Incremental Value

## Purpose

This experiment tests whether total OpenStreetMap urban-activity counts add
predictive ranking information beyond the deduplicated road-and-parking
baseline.

The experiment is incremental rather than a new tuned model search.

## Dataset

- Rows: {len(dataframe):,}
- Positive existing-station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km
- Activity source: OpenStreetMap mapped activity POIs

## Feature Sets

### Normalized 12

The existing deduplicated road-and-parking baseline. It excludes
`road_length_m` and `parking_area_m2` while retaining their normalized
counterparts.

### Normalized 12 + Local POI Activity

Adds:

- `poi_count`

This tests whether activity mapped inside the local 500-m cell adds information
beyond road and parking features.

### Normalized 12 + POI Activity Context

Adds:

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

The feature family is deliberately based on total POI counts and spatial scale,
not on the earlier target-association ranking of individual POI categories.
This avoids choosing activity categories merely because they looked strongest
against the same 46 positive labels later used for evaluation.

The 12-, 13-, and 15-feature configurations also keep Random Forest
`max_features="sqrt"` at three candidate predictors per split, reducing one
possible estimator-configuration confound.

## Models

The existing untuned Logistic Regression, Random Forest, and
HistGradientBoosting configurations are reused unchanged.

The same predefined 5-km spatial folds and the same class-imbalance treatments
are retained. No hyperparameter search is performed.

## Spatial OOF Results

{chr(10).join(table_lines)}

## Full Activity-Context Delta Against Normalized 12

{chr(10).join(delta_lines)}

## Interpretation Policy

Average precision is primary because only a very small fraction of Ankara grid
cells contain known existing charging stations. Top-1% and top-5% recall are
also reported because VoltSight is a candidate-ranking system.

A positive delta means the activity feature set improved spatial OOF ranking
under this fixed experiment. It is predictive evidence, not a causal estimate
of the real-world effect of urban activity on station placement.

The activity audit showed strong descriptive separation between known station
cells and non-station cells, but that same target was used to calculate those
descriptive statistics. Those SMD values are therefore context only, not
independent validation evidence.

OSM activity is a mapped urban-activity proxy. Low counts can reflect either
low activity or incomplete OSM mapping.

Only 46 positive station cells are available, so fold-level variability must
be considered alongside pooled metrics. The existing spatial-block design
reduces local train-validation dependence but does not eliminate all spatial
autocorrelation.

The historical full-14 baselines remain historical references. This experiment
uses normalized-12 as the deduplicated baseline for future feature-family
evaluation.

Category-specific POI features are intentionally deferred. They should only be
tested after this target-agnostic total-activity experiment establishes whether
the feature family has robust incremental value.

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


def save_outputs(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    """Save activity incremental-value outputs."""

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

    oof.to_csv(
        OOF_PATH,
        index=False,
        encoding="utf-8",
    )


def print_results(
    metrics: pd.DataFrame,
) -> None:
    """Print the main activity incremental-value metrics."""

    columns = [
        "model_label",
        "feature_set_label",
        "feature_count",
        "pooled_average_precision",
        "delta_pooled_average_precision",
        "mean_fold_average_precision",
        "std_fold_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
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
    """Run the Ankara OSM activity incremental-value experiment."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara OSM Activity Incremental Value"
    )

    print(
        "="
        * 70
    )

    dataframe = (
        load_analysis_frame()
    )

    (
        metrics,
        fold_metrics,
        oof,
    ) = run_experiment(
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
        "Ankara OSM activity incremental value completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
