from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from voltsight.models.analyze_ankara_activity_incremental_value import (
    ACTIVITY_PATH,
    ACTIVITY_TOTAL_COLUMNS,
    NORMALIZED_BASE_FEATURES,
    attach_activity_features,
)
from voltsight.models.train_ankara_random_forest_baseline import (
    N_SPLITS,
    TARGET_COLUMN,
    calculate_top_fraction_metrics,
    load_inputs as load_baseline_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_rf_activity_seed_stability_metrics.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_rf_activity_seed_stability_summary.md"
)

FEATURE_SETS = {
    "normalized_12": (
        NORMALIZED_BASE_FEATURES
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
    "normalized_12_plus_activity_context": (
        "Normalized 12 + total activity context"
    ),
}

RANDOM_STATES = (
    42,
    43,
    44,
    45,
    46,
)


def validate_configuration() -> None:
    """Validate the fixed Random Forest activity robustness design."""

    if len(
        NORMALIZED_BASE_FEATURES
    ) != 12:
        raise ValueError(
            "Expected 12 normalized baseline predictors."
        )

    if len(
        FEATURE_SETS[
            "normalized_12_plus_activity_context"
        ]
    ) != 15:
        raise ValueError(
            "Expected 15 predictors in the activity-context set."
        )

    if len(
        set(
            RANDOM_STATES
        )
    ) != len(
        RANDOM_STATES
    ):
        raise ValueError(
            "Random Forest seeds must be unique."
        )


def build_model(
    random_state: int,
) -> RandomForestClassifier:
    """Create the historical Random Forest baseline with one chosen seed."""

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def load_analysis_frame() -> pd.DataFrame:
    """Load the baseline spatial folds and attach total activity context."""

    validate_configuration()

    if not ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            f"Activity feature dataset not found: {ACTIVITY_PATH}"
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


def run_single_configuration(
    dataframe: pd.DataFrame,
    *,
    feature_columns: tuple[
        str,
        ...,
    ],
    random_state: int,
) -> dict[
    str,
    float | int,
]:
    """Run one complete five-fold spatial OOF Random Forest evaluation."""

    if not feature_columns:
        raise ValueError(
            "At least one predictor is required."
        )

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
            "Missing predictor columns: "
            f"{sorted(missing)}"
        )

    oof_scores = np.full(
        len(
            dataframe
        ),
        np.nan,
        dtype=float,
    )

    fold_average_precision: list[
        float
    ] = []

    for fold in range(
        N_SPLITS
    ):
        train_mask = (
            dataframe[
                "cv_fold"
            ].to_numpy(
                dtype=int
            )
            != fold
        )

        validation_mask = (
            ~train_mask
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
                f"Fold {fold} training data does not contain both classes."
            )

        if len(
            np.unique(
                y_validation
            )
        ) != 2:
            raise ValueError(
                f"Fold {fold} validation data does not contain both classes."
            )

        model = build_model(
            random_state=random_state
        )

        model.fit(
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
                f"Fold {fold} produced invalid scores."
            )

        validation_positions = (
            np.flatnonzero(
                validation_mask
            )
        )

        oof_scores[
            validation_positions
        ] = scores

        fold_average_precision.append(
            float(
                average_precision_score(
                    y_validation,
                    scores,
                )
            )
        )

    if not np.isfinite(
        oof_scores
    ).all():
        raise ValueError(
            "OOF scores are incomplete."
        )

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(
        dtype=int
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

    return {
        "pooled_average_precision": float(
            average_precision_score(
                y_true,
                oof_scores,
            )
        ),
        "mean_fold_average_precision": float(
            np.mean(
                fold_average_precision
            )
        ),
        "std_fold_average_precision": float(
            np.std(
                fold_average_precision,
                ddof=1,
            )
        ),
        "pooled_roc_auc": float(
            roc_auc_score(
                y_true,
                oof_scores,
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


def run_seed_stability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate both feature sets under the same five Random Forest seeds."""

    records: list[
        dict[
            str,
            float | int | str,
        ]
    ] = []

    for random_state in RANDOM_STATES:
        print(
            f"Random state {random_state}"
        )

        for (
            feature_set_name,
            feature_columns,
        ) in FEATURE_SETS.items():
            print(
                "  "
                f"{FEATURE_SET_LABELS[feature_set_name]}"
            )

            metrics = (
                run_single_configuration(
                    dataframe,
                    feature_columns=(
                        feature_columns
                    ),
                    random_state=(
                        random_state
                    ),
                )
            )

            records.append(
                {
                    "random_state": (
                        random_state
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
                    **metrics,
                }
            )

    return pd.DataFrame(
        records
    )


def create_aggregated_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate Random Forest diagnostics across seeds."""

    return (
        metrics.groupby(
            [
                "feature_set",
                "feature_set_label",
                "feature_count",
            ],
            as_index=False,
        )
        .agg(
            mean_pooled_ap=(
                "pooled_average_precision",
                "mean",
            ),
            std_pooled_ap=(
                "pooled_average_precision",
                "std",
            ),
            min_pooled_ap=(
                "pooled_average_precision",
                "min",
            ),
            max_pooled_ap=(
                "pooled_average_precision",
                "max",
            ),
            mean_fold_ap=(
                "mean_fold_average_precision",
                "mean",
            ),
            mean_fold_ap_std=(
                "std_fold_average_precision",
                "mean",
            ),
            mean_roc_auc=(
                "pooled_roc_auc",
                "mean",
            ),
            mean_top_1_recall=(
                "top_1_percent_recall",
                "mean",
            ),
            mean_top_5_recall=(
                "top_5_percent_recall",
                "mean",
            ),
        )
    )


def create_paired_delta_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create seed-paired activity-context minus normalized-12 deltas."""

    wide = metrics.pivot(
        index="random_state",
        columns="feature_set",
        values=[
            "pooled_average_precision",
            "mean_fold_average_precision",
            "pooled_roc_auc",
            "top_1_percent_recall",
            "top_5_percent_recall",
        ],
    )

    records = []

    for random_state in RANDOM_STATES:
        baseline = "normalized_12"

        activity = (
            "normalized_12_plus_activity_context"
        )

        records.append(
            {
                "random_state": (
                    random_state
                ),
                "baseline_pooled_ap": float(
                    wide.loc[
                        random_state,
                        (
                            "pooled_average_precision",
                            baseline,
                        ),
                    ]
                ),
                "activity_pooled_ap": float(
                    wide.loc[
                        random_state,
                        (
                            "pooled_average_precision",
                            activity,
                        ),
                    ]
                ),
                "delta_pooled_ap": float(
                    wide.loc[
                        random_state,
                        (
                            "pooled_average_precision",
                            activity,
                        ),
                    ]
                    - wide.loc[
                        random_state,
                        (
                            "pooled_average_precision",
                            baseline,
                        ),
                    ]
                ),
                "delta_mean_fold_ap": float(
                    wide.loc[
                        random_state,
                        (
                            "mean_fold_average_precision",
                            activity,
                        ),
                    ]
                    - wide.loc[
                        random_state,
                        (
                            "mean_fold_average_precision",
                            baseline,
                        ),
                    ]
                ),
                "delta_roc_auc": float(
                    wide.loc[
                        random_state,
                        (
                            "pooled_roc_auc",
                            activity,
                        ),
                    ]
                    - wide.loc[
                        random_state,
                        (
                            "pooled_roc_auc",
                            baseline,
                        ),
                    ]
                ),
                "delta_top_1_recall": float(
                    wide.loc[
                        random_state,
                        (
                            "top_1_percent_recall",
                            activity,
                        ),
                    ]
                    - wide.loc[
                        random_state,
                        (
                            "top_1_percent_recall",
                            baseline,
                        ),
                    ]
                ),
                "delta_top_5_recall": float(
                    wide.loc[
                        random_state,
                        (
                            "top_5_percent_recall",
                            activity,
                        ),
                    ]
                    - wide.loc[
                        random_state,
                        (
                            "top_5_percent_recall",
                            baseline,
                        ),
                    ]
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def create_summary(
    metrics: pd.DataFrame,
) -> None:
    """Write the Random Forest activity seed-stability summary."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated = (
        create_aggregated_table(
            metrics
        )
    )

    paired = (
        create_paired_delta_table(
            metrics
        )
    )

    table_lines = [
        "| Feature set | Features | Mean pooled AP | Seed AP std | Min AP | Max AP | Mean fold AP | Mean fold-AP std | Mean ROC-AUC | Mean top 1% recall | Mean top 5% recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in aggregated.itertuples(
        index=False
    ):
        table_lines.append(
            "| "
            f"{row.feature_set_label} | "
            f"{int(row.feature_count)} | "
            f"{row.mean_pooled_ap:.6f} | "
            f"{row.std_pooled_ap:.6f} | "
            f"{row.min_pooled_ap:.6f} | "
            f"{row.max_pooled_ap:.6f} | "
            f"{row.mean_fold_ap:.6f} | "
            f"{row.mean_fold_ap_std:.6f} | "
            f"{row.mean_roc_auc:.6f} | "
            f"{row.mean_top_1_recall:.6f} | "
            f"{row.mean_top_5_recall:.6f} |"
        )

    seed_lines = [
        (
            f"- Seed {int(row.random_state)}: "
            f"baseline AP {row.baseline_pooled_ap:.6f}, "
            f"activity AP {row.activity_pooled_ap:.6f}, "
            f"delta {row.delta_pooled_ap:+.6f}, "
            f"mean-fold AP delta {row.delta_mean_fold_ap:+.6f}, "
            f"top-1 delta {row.delta_top_1_recall:+.6f}, "
            f"top-5 delta {row.delta_top_5_recall:+.6f}"
        )
        for row in paired.itertuples(
            index=False
        )
    ]

    summary = f"""# Ankara Random Forest Activity Seed Stability

## Purpose

This robustness diagnostic checks whether the Random Forest gain from total OSM
activity context is stable across several fixed Random Forest seeds.

This is not hyperparameter tuning.

## Configuration

- Spatial folds: {N_SPLITS}
- Random Forest trees: 400
- Maximum depth: 12
- Minimum leaf samples: 5
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- Seeds: {", ".join(str(seed) for seed in RANDOM_STATES)}

The only deliberate change across repeated runs is `random_state`.

Both the 12-feature baseline and 15-feature activity-context model resolve
`sqrt(n_features)` to three candidate predictors per split.

## Feature Sets

### Normalized 12

The deduplicated road-and-parking baseline.

### Normalized 12 + Total Activity Context

Adds:

- `poi_count`
- `poi_count_within_1000m`
- `poi_count_within_2000m`

## Seed-Aggregated Results

{chr(10).join(table_lines)}

## Paired Seed Results

{chr(10).join(seed_lines)}

- Mean paired activity minus baseline pooled-AP delta:
  {paired["delta_pooled_ap"].mean():+.6f}
- Paired pooled-AP delta standard deviation:
  {paired["delta_pooled_ap"].std(ddof=1):.6f}
- Activity pooled AP higher in:
  {int((paired["delta_pooled_ap"] > 0).sum())}/{len(paired)} seeds
- Activity mean-fold AP higher in:
  {int((paired["delta_mean_fold_ap"] > 0).sum())}/{len(paired)} seeds
- Activity top-1% recall higher in:
  {int((paired["delta_top_1_recall"] > 0).sum())}/{len(paired)} seeds
- Activity top-5% recall higher in:
  {int((paired["delta_top_5_recall"] > 0).sum())}/{len(paired)} seeds

## Interpretation Policy

The preceding activity experiment showed positive pooled-AP deltas for Logistic
Regression, Random Forest, and HistGradientBoosting. Random Forest is stochastic,
and its seed-42 mean-fold AP did not improve despite the pooled-AP gain.

This diagnostic therefore focuses on whether the Random Forest activity gain is
repeatable rather than an artifact of one favorable seed.

A stable positive paired pooled-AP delta would strengthen the case for adopting
the three total-activity features as the next canonical ML feature-family
extension. Mixed or negative seed results would favor keeping them as an
experimental/contextual layer.

Only 46 positive existing-station cells are available. Seed stability cannot
replace independent external validation, and the 5-km spatial-block design
reduces but does not eliminate spatial dependence.

OSM activity remains a mapped urban-activity proxy rather than direct observed
EV demand.

## Output

- `data/processed/{METRICS_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def save_outputs(
    metrics: pd.DataFrame,
) -> None:
    """Save seed-level Random Forest metrics."""

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.sort_values(
        [
            "feature_set",
            "random_state",
        ],
        kind="stable",
    ).to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8",
    )


def print_results(
    metrics: pd.DataFrame,
) -> None:
    """Print seed-paired AP diagnostics."""

    paired = (
        create_paired_delta_table(
            metrics
        )
    )

    print(
        paired[
            [
                "random_state",
                "baseline_pooled_ap",
                "activity_pooled_ap",
                "delta_pooled_ap",
                "delta_mean_fold_ap",
                "delta_top_1_recall",
                "delta_top_5_recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()

    print(
        "Mean activity minus baseline AP delta:",
        f"{paired['delta_pooled_ap'].mean():+.6f}",
    )

    print(
        "Activity pooled AP higher in:",
        (
            f"{int((paired['delta_pooled_ap'] > 0).sum())}"
            f"/{len(paired)} seeds"
        ),
    )


def main() -> None:
    """Run the Ankara Random Forest activity seed-stability diagnostic."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara RF Activity Seed Stability"
    )

    print(
        "="
        * 70
    )

    dataframe = (
        load_analysis_frame()
    )

    metrics = (
        run_seed_stability(
            dataframe
        )
    )

    save_outputs(
        metrics
    )

    create_summary(
        metrics
    )

    print_results(
        metrics
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara RF activity seed stability completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
