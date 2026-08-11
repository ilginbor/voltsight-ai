from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    / "ankara_feature_redundancy_sensitivity_metrics.csv"
)

FOLD_METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_feature_redundancy_sensitivity_fold_metrics.csv"
)

RELATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_feature_redundancy_relations.csv"
)

PLOT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_feature_redundancy_sensitivity.png"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_feature_redundancy_sensitivity_summary.md"
)

FULL_FEATURE_COLUMNS = tuple(
    LOGISTIC_FEATURE_COLUMNS
)

NORMALIZED_FEATURE_COLUMNS = tuple(
    feature
    for feature in FULL_FEATURE_COLUMNS
    if feature
    not in {
        "road_length_m",
        "parking_area_m2",
    }
)

RAW_FEATURE_COLUMNS = tuple(
    feature
    for feature in FULL_FEATURE_COLUMNS
    if feature
    not in {
        "road_density_km_per_km2",
        "parking_area_ratio",
    }
)

FEATURE_SETS: dict[
    str,
    tuple[str, ...],
] = {
    "full_14": FULL_FEATURE_COLUMNS,
    "normalized_12": NORMALIZED_FEATURE_COLUMNS,
    "raw_12": RAW_FEATURE_COLUMNS,
}

FEATURE_SET_LABELS = {
    "full_14": "Full 14",
    "normalized_12": "Normalized 12",
    "raw_12": "Raw 12",
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

REDUNDANT_PAIR_DEFINITIONS = (
    (
        "road_length_m",
        "road_density_km_per_km2",
    ),
    (
        "parking_area_m2",
        "parking_area_ratio",
    ),
)


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
    """Validate the canonical and deduplicated feature-set definitions."""

    canonical = tuple(
        LOGISTIC_FEATURE_COLUMNS
    )

    if tuple(
        RANDOM_FOREST_FEATURE_COLUMNS
    ) != canonical:
        raise ValueError(
            "Random Forest features differ from the logistic baseline."
        )

    if tuple(
        GRADIENT_BOOSTING_FEATURE_COLUMNS
    ) != canonical:
        raise ValueError(
            "Gradient Boosting features differ from the logistic baseline."
        )

    if FULL_FEATURE_COLUMNS != canonical:
        raise ValueError(
            "Full feature set does not match the baseline predictor schema."
        )

    if len(FULL_FEATURE_COLUMNS) != 14:
        raise ValueError(
            "Expected 14 baseline predictors."
        )

    if len(NORMALIZED_FEATURE_COLUMNS) != 12:
        raise ValueError(
            "Normalized deduplicated set must contain 12 predictors."
        )

    if len(RAW_FEATURE_COLUMNS) != 12:
        raise ValueError(
            "Raw deduplicated set must contain 12 predictors."
        )

    if "road_length_m" in NORMALIZED_FEATURE_COLUMNS:
        raise ValueError(
            "Normalized set must drop road_length_m."
        )

    if "parking_area_m2" in NORMALIZED_FEATURE_COLUMNS:
        raise ValueError(
            "Normalized set must drop parking_area_m2."
        )

    if "road_density_km_per_km2" in RAW_FEATURE_COLUMNS:
        raise ValueError(
            "Raw set must drop road_density_km_per_km2."
        )

    if "parking_area_ratio" in RAW_FEATURE_COLUMNS:
        raise ValueError(
            "Raw set must drop parking_area_ratio."
        )


def calculate_redundancy_relations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify near-linear relationships for the two duplicate pairs."""

    records: list[
        dict[str, float | int | str]
    ] = []

    for raw_feature, normalized_feature in (
        REDUNDANT_PAIR_DEFINITIONS
    ):
        raw = pd.to_numeric(
            dataframe[raw_feature],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        normalized = pd.to_numeric(
            dataframe[normalized_feature],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(raw).all()
            or not np.isfinite(normalized).all()
        ):
            raise ValueError(
                "Non-finite values found while calculating redundancy relations."
            )

        correlation = float(
            np.corrcoef(
                raw,
                normalized,
            )[0, 1]
        )

        nonzero_mask = (
            np.abs(raw) > 1e-12
        )

        ratios = (
            normalized[nonzero_mask]
            / raw[nonzero_mask]
        )

        records.append(
            {
                "raw_feature": raw_feature,
                "normalized_feature": normalized_feature,
                "pearson_correlation": correlation,
                "nonzero_raw_rows": int(
                    nonzero_mask.sum()
                ),
                "ratio_min": float(
                    ratios.min()
                ),
                "ratio_median": float(
                    np.median(ratios)
                ),
                "ratio_max": float(
                    ratios.max()
                ),
                "ratio_std": float(
                    ratios.std()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_model(
    model_name: str,
) -> BaseEstimator:
    """Build one unchanged baseline estimator."""

    if model_name == "logistic_regression":
        return build_logistic_pipeline()

    if model_name == "random_forest":
        return build_random_forest_model()

    if model_name == "gradient_boosting":
        return build_gradient_boosting_model()

    raise ValueError(
        f"Unknown model name: {model_name}"
    )


def fit_model(
    model: BaseEstimator,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Fit with exactly the same imbalance policy as the baseline."""

    if model_name == "gradient_boosting":
        weights = (
            calculate_balanced_sample_weights(
                y_train
            )
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
        )

        return model

    model.fit(
        x_train,
        y_train,
    )

    return model


def validate_input(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    """Validate one feature set before spatial cross-validation."""

    required = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
        *feature_columns,
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Sensitivity input is missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Sensitivity input is empty."
        )

    if dataframe[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found."
        )

    if set(
        dataframe[TARGET_COLUMN].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Target must contain both classes."
        )

    if set(
        dataframe["cv_fold"].unique()
    ) != set(
        range(N_SPLITS)
    ):
        raise ValueError(
            "Unexpected spatial fold identifiers."
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
            values.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                f"Non-finite values found in {feature}."
            )


def run_single_configuration(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """Run one model/feature-set combination on fixed spatial folds."""

    validate_input(
        dataframe,
        feature_columns,
    )

    base_model = build_model(
        model_name
    )

    feature_list = list(
        feature_columns
    )

    oof_scores = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    fold_records: list[
        dict[str, float | int | str]
    ] = []

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

        if train.empty or validation.empty:
            raise ValueError(
                f"Fold {fold} has an empty split."
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

        if len(
            np.unique(y_train)
        ) != 2:
            raise ValueError(
                f"Fold {fold} training data lacks both classes."
            )

        if len(
            np.unique(y_validation)
        ) != 2:
            raise ValueError(
                f"Fold {fold} validation data lacks both classes."
            )

        model = clone(
            base_model
        )

        model = fit_model(
            model,
            model_name,
            train[feature_list],
            y_train,
        )

        scores = model.predict_proba(
            validation[feature_list]
        )[
            :,
            1,
        ]

        scores = np.asarray(
            scores,
            dtype=float,
        )

        if not np.isfinite(
            scores
        ).all():
            raise ValueError(
                "Model produced non-finite validation scores."
            )

        oof_scores[
            validation_mask.to_numpy()
        ] = scores

        fold_records.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
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

    if np.isnan(
        oof_scores
    ).any():
        raise ValueError(
            "Missing out-of-fold predictions."
        )

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
    )

    fold_metrics = pd.DataFrame(
        fold_records
    )

    top_one = (
        calculate_top_fraction_metrics(
            y_true,
            oof_scores,
            fraction=0.01,
        )
    )

    top_five = (
        calculate_top_fraction_metrics(
            y_true,
            oof_scores,
            fraction=0.05,
        )
    )

    metrics: dict[
        str,
        float | int | str,
    ] = {
        "model": model_name,
        "feature_set": feature_set_name,
        "feature_count": len(
            feature_columns
        ),
        "pooled_average_precision": float(
            average_precision_score(
                y_true,
                oof_scores,
            )
        ),
        "pooled_roc_auc": float(
            roc_auc_score(
                y_true,
                oof_scores,
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
    }

    return (
        fold_metrics,
        metrics,
    )


def run_sensitivity_analysis(
    dataframe: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    feature_set_names: tuple[str, ...] = tuple(
        FEATURE_SETS
    ),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run all selected model-by-feature-set sensitivity combinations."""

    validate_feature_definitions()

    unknown_models = (
        set(model_names)
        - set(MODEL_NAMES)
    )

    if unknown_models:
        raise ValueError(
            f"Unknown models: {sorted(unknown_models)}"
        )

    unknown_sets = (
        set(feature_set_names)
        - set(FEATURE_SETS)
    )

    if unknown_sets:
        raise ValueError(
            f"Unknown feature sets: {sorted(unknown_sets)}"
        )

    metrics_records: list[
        dict[str, float | int | str]
    ] = []

    fold_frames: list[
        pd.DataFrame
    ] = []

    for model_name in model_names:
        for feature_set_name in (
            feature_set_names
        ):
            print(
                "Running:",
                MODEL_LABELS[
                    model_name
                ],
                "/",
                FEATURE_SET_LABELS[
                    feature_set_name
                ],
            )

            fold_metrics, metrics = (
                run_single_configuration(
                    dataframe,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_columns=FEATURE_SETS[
                        feature_set_name
                    ],
                )
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
        metrics_table,
        fold_metrics_table,
        model_names=model_names,
        feature_set_names=feature_set_names,
    )

    return (
        metrics_table,
        fold_metrics_table,
    )


def validate_outputs(
    metrics_table: pd.DataFrame,
    fold_metrics_table: pd.DataFrame,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    feature_set_names: tuple[str, ...] = tuple(
        FEATURE_SETS
    ),
) -> None:
    """Validate sensitivity tables before saving."""

    expected_configurations = (
        len(model_names)
        * len(feature_set_names)
    )

    if len(
        metrics_table
    ) != expected_configurations:
        raise ValueError(
            "Unexpected sensitivity metric row count."
        )

    if len(
        fold_metrics_table
    ) != (
        expected_configurations
        * N_SPLITS
    ):
        raise ValueError(
            "Unexpected sensitivity fold metric row count."
        )

    for column in (
        "pooled_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    ):
        if not metrics_table[
            column
        ].between(
            0,
            1,
        ).all():
            raise ValueError(
                f"Invalid metric values in {column}."
            )


def save_outputs(
    metrics_table: pd.DataFrame,
    fold_metrics_table: pd.DataFrame,
    relation_table: pd.DataFrame,
) -> None:
    """Save generated tabular outputs."""

    create_output_directories()

    metrics_table.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_metrics_table.to_csv(
        FOLD_METRICS_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    relation_table.to_csv(
        RELATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_plot(
    metrics_table: pd.DataFrame,
) -> None:
    """Plot pooled AP for full and deduplicated feature sets."""

    model_order = list(
        MODEL_NAMES
    )

    feature_set_order = list(
        FEATURE_SETS
    )

    x = np.arange(
        len(model_order)
    )

    width = 0.24

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    offsets = (
        -width,
        0.0,
        width,
    )

    for offset, feature_set_name in zip(
        offsets,
        feature_set_order,
    ):
        values = []

        for model_name in model_order:
            row = metrics_table.loc[
                (
                    metrics_table[
                        "model"
                    ] == model_name
                )
                & (
                    metrics_table[
                        "feature_set"
                    ] == feature_set_name
                )
            ]

            values.append(
                float(
                    row[
                        "pooled_average_precision"
                    ].iloc[0]
                )
            )

        axis.bar(
            x + offset,
            values,
            width=width,
            label=FEATURE_SET_LABELS[
                feature_set_name
            ],
        )

    axis.set_xticks(
        x,
        [
            MODEL_LABELS[
                model_name
            ]
            for model_name in model_order
        ],
    )

    axis.set_ylabel(
        "Spatial OOF pooled average precision"
    )

    axis.set_title(
        "Ankara Feature Redundancy Sensitivity"
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


def create_summary(
    dataframe: pd.DataFrame,
    metrics_table: pd.DataFrame,
    relation_table: pd.DataFrame,
) -> None:
    """Write methodology and results as Markdown."""

    relation_lines = "\n".join(
        (
            f"- `{row.raw_feature}` vs `{row.normalized_feature}`: "
            f"Pearson {row.pearson_correlation:.12f}, "
            f"median ratio {row.ratio_median:.12g}, "
            f"ratio std {row.ratio_std:.12g}"
        )
        for row in relation_table.itertuples(
            index=False
        )
    )

    metric_lines = []

    for row in metrics_table.itertuples(
        index=False
    ):
        metric_lines.append(
            "| "
            f"{MODEL_LABELS[row.model]} | "
            f"{FEATURE_SET_LABELS[row.feature_set]} | "
            f"{int(row.feature_count)} | "
            f"{row.pooled_average_precision:.6f} | "
            f"{row.mean_fold_average_precision:.6f} | "
            f"{row.std_fold_average_precision:.6f} | "
            f"{row.pooled_roc_auc:.6f} | "
            f"{row.top_1_percent_recall:.6f} | "
            f"{row.top_5_percent_recall:.6f} |"
        )

    normalized_lines = []

    for model_name in MODEL_NAMES:
        full_row = metrics_table.loc[
            (
                metrics_table["model"]
                == model_name
            )
            & (
                metrics_table[
                    "feature_set"
                ] == "full_14"
            )
        ].iloc[0]

        normalized_row = metrics_table.loc[
            (
                metrics_table["model"]
                == model_name
            )
            & (
                metrics_table[
                    "feature_set"
                ] == "normalized_12"
            )
        ].iloc[0]

        delta = float(
            normalized_row[
                "pooled_average_precision"
            ]
            - full_row[
                "pooled_average_precision"
            ]
        )

        normalized_lines.append(
            f"- {MODEL_LABELS[model_name]}: "
            f"full AP {full_row['pooled_average_precision']:.6f}, "
            f"normalized-12 AP {normalized_row['pooled_average_precision']:.6f}, "
            f"delta {delta:+.6f}."
        )

    summary = f"""# Ankara Feature Redundancy Sensitivity

## Purpose

This experiment tests whether two near-deterministic feature pairs can be
removed without materially changing the spatially validated Ankara baseline
rankings.

The existing model hyperparameters, class-imbalance handling, and predefined
5-km spatial folds are kept unchanged.

No hyperparameter search is performed.

## Dataset

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Spatial folds: {N_SPLITS}
- Full predictors: {len(FULL_FEATURE_COLUMNS)}
- Deduplicated predictors: {len(NORMALIZED_FEATURE_COLUMNS)}

## Redundancy Audit

{relation_lines}

The relationships are treated as near-deterministic transforms for this fixed
500-m grid. Small deviations from an exact constant ratio can arise from
stored precision or upstream rounding.

## Feature Sets

### Full 14

The original leakage-safe road and parking predictor set.

### Normalized 12

Drops:

- `road_length_m`
- `parking_area_m2`

Retains their normalized counterparts:

- `road_density_km_per_km2`
- `parking_area_ratio`

This is the preferred deduplicated representation if predictive performance is
not materially worse, because the retained variables express road intensity
and parking coverage independently of raw square-metre / metre scale.

### Raw 12

Drops the normalized counterparts instead:

- `road_density_km_per_km2`
- `parking_area_ratio`

and keeps:

- `road_length_m`
- `parking_area_m2`

The raw-12 branch is a sensitivity check rather than a proposed canonical
feature set.

## Spatial OOF Results

| Model | Feature set | Features | Pooled AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Top 5% recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(metric_lines)}

## Normalized-12 Comparison

{chr(10).join(normalized_lines)}

## Interpretation Policy

This is a predictive redundancy sensitivity analysis, not a causal feature
selection procedure.

A deduplicated feature set should only replace the original 14-feature baseline
if performance remains comparable while interpretation becomes cleaner.

The `road_segment_count` relationship with road length/density is intentionally
not removed here. Although it can be highly correlated, segment count encodes a
different network characteristic and is not a deterministic scale conversion.

Likewise, main-road segment count and main-road length remain separate.

## Outputs

- `data/processed/ankara_feature_redundancy_sensitivity_metrics.csv`
- `data/processed/ankara_feature_redundancy_sensitivity_fold_metrics.csv`
- `data/processed/ankara_feature_redundancy_relations.csv`
- `docs/ankara_feature_redundancy_sensitivity.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    metrics_table: pd.DataFrame,
    relation_table: pd.DataFrame,
) -> None:
    """Print concise sensitivity diagnostics."""

    print("-" * 70)
    print("Near-deterministic feature relations")
    print(
        relation_table[
            [
                "raw_feature",
                "normalized_feature",
                "pearson_correlation",
                "ratio_median",
                "ratio_std",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Spatial OOF sensitivity")

    display = metrics_table.copy()

    display["model"] = display[
        "model"
    ].map(
        MODEL_LABELS
    )

    display["feature_set"] = display[
        "feature_set"
    ].map(
        FEATURE_SET_LABELS
    )

    print(
        display[
            [
                "model",
                "feature_set",
                "feature_count",
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


def main() -> None:
    """Run Ankara feature-redundancy sensitivity analysis."""

    print("=" * 70)
    print(
        "VoltSight - Ankara Feature Redundancy Sensitivity"
    )
    print("=" * 70)

    dataframe = load_inputs()

    relation_table = (
        calculate_redundancy_relations(
            dataframe
        )
    )

    metrics_table, fold_metrics_table = (
        run_sensitivity_analysis(
            dataframe
        )
    )

    save_outputs(
        metrics_table,
        fold_metrics_table,
        relation_table,
    )

    create_plot(
        metrics_table
    )

    create_summary(
        dataframe,
        metrics_table,
        relation_table,
    )

    print_results(
        metrics_table,
        relation_table,
    )

    print("=" * 70)
    print(
        "Ankara feature redundancy sensitivity completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
