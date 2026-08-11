from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from voltsight.models.train_ankara_random_forest_baseline import (
    FEATURE_COLUMNS,
    N_SPLITS,
    TARGET_COLUMN,
    calculate_top_fraction_metrics,
    load_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_rf_redundancy_seed_stability_metrics.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_rf_redundancy_seed_stability_summary.md"
)

FULL_14 = tuple(FEATURE_COLUMNS)

NORMALIZED_12 = tuple(
    feature
    for feature in FEATURE_COLUMNS
    if feature not in {
        "road_length_m",
        "parking_area_m2",
    }
)

FEATURE_SETS = {
    "full_14": FULL_14,
    "normalized_12": NORMALIZED_12,
}

RANDOM_STATES = (
    42,
    43,
    44,
    45,
    46,
)


def build_model(random_state: int) -> RandomForestClassifier:
    """Create the existing Random Forest baseline with a chosen seed."""

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def run_single_configuration(
    dataframe: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    random_state: int,
) -> dict[str, float | int]:
    """Run one complete 5-fold spatial OOF Random Forest evaluation."""

    if not feature_columns:
        raise ValueError("At least one predictor is required.")

    missing = set(feature_columns) - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Missing predictor columns: "
            f"{sorted(missing)}"
        )

    oof_scores = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    fold_ap: list[float] = []

    for fold in range(N_SPLITS):
        train_mask = (
            dataframe["cv_fold"].to_numpy(dtype=int)
            != fold
        )
        validation_mask = ~train_mask

        train = dataframe.loc[train_mask]
        validation = dataframe.loc[validation_mask]

        y_train = train[
            TARGET_COLUMN
        ].to_numpy(dtype=int)

        y_validation = validation[
            TARGET_COLUMN
        ].to_numpy(dtype=int)

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Fold {fold} training data "
                "does not contain both classes."
            )

        if len(np.unique(y_validation)) != 2:
            raise ValueError(
                f"Fold {fold} validation data "
                "does not contain both classes."
            )

        model = build_model(
            random_state=random_state
        )

        model.fit(
            train[list(feature_columns)],
            y_train,
        )

        scores = model.predict_proba(
            validation[list(feature_columns)]
        )[:, 1]

        validation_positions = np.flatnonzero(
            validation_mask
        )

        oof_scores[
            validation_positions
        ] = scores

        fold_ap.append(
            float(
                average_precision_score(
                    y_validation,
                    scores,
                )
            )
        )

    if not np.isfinite(oof_scores).all():
        raise ValueError(
            "OOF predictions contain missing "
            "or non-finite values."
        )

    y_true = dataframe[
        TARGET_COLUMN
    ].to_numpy(dtype=int)

    top_one = calculate_top_fraction_metrics(
        y_true,
        oof_scores,
        fraction=0.01,
    )

    top_five = calculate_top_fraction_metrics(
        y_true,
        oof_scores,
        fraction=0.05,
    )

    return {
        "random_state": random_state,
        "feature_count": len(feature_columns),
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
            np.mean(fold_ap)
        ),
        "std_fold_average_precision": float(
            np.std(
                fold_ap,
                ddof=1,
            )
        ),
        "top_1_percent_recall": float(
            top_one["recall"]
        ),
        "top_5_percent_recall": float(
            top_five["recall"]
        ),
    }


def run_seed_stability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate full and deduplicated predictors over fixed RF seeds."""

    records: list[
        dict[str, float | int | str]
    ] = []

    for feature_set_name, features in FEATURE_SETS.items():
        for random_state in RANDOM_STATES:
            metrics = run_single_configuration(
                dataframe,
                feature_columns=features,
                random_state=random_state,
            )

            records.append(
                {
                    "feature_set": feature_set_name,
                    **metrics,
                }
            )

    result = pd.DataFrame(records)

    expected_rows = (
        len(FEATURE_SETS)
        * len(RANDOM_STATES)
    )

    if len(result) != expected_rows:
        raise ValueError(
            "Unexpected seed-stability row count."
        )

    return result


def create_summary(
    metrics: pd.DataFrame,
) -> None:
    """Write a concise Markdown interpretation of seed stability."""

    summary_table = (
        metrics.groupby(
            "feature_set",
            as_index=False,
        )
        .agg(
            feature_count=(
                "feature_count",
                "first",
            ),
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

    wide = metrics.pivot(
        index="random_state",
        columns="feature_set",
        values="pooled_average_precision",
    )

    paired_delta = (
        wide["normalized_12"]
        - wide["full_14"]
    )

    seed_lines = "\n".join(
        (
            f"- Seed {int(seed)}: "
            f"full-14 AP {wide.loc[seed, 'full_14']:.6f}, "
            f"normalized-12 AP {wide.loc[seed, 'normalized_12']:.6f}, "
            f"delta {paired_delta.loc[seed]:+.6f}"
        )
        for seed in wide.index
    )

    table_lines = [
        "| Feature set | Features | Mean pooled AP | Seed AP std | Min AP | Max AP | Mean fold AP | Mean fold-AP std | Mean ROC-AUC | Mean top 1% recall | Mean top 5% recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_table.itertuples(
        index=False
    ):
        table_lines.append(
            "| "
            f"{row.feature_set} | "
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

    summary = f"""# Ankara Random Forest Redundancy Seed Stability

## Purpose

This diagnostic checks whether the Random Forest difference between the
original 14-feature predictor set and the normalized 12-feature deduplicated
set is stable across Random Forest seeds.

This is a robustness diagnostic, not hyperparameter tuning.

## Configuration

- Spatial folds: {N_SPLITS}
- Random Forest trees: 400
- Maximum depth: 12
- Minimum leaf samples: 5
- `max_features="sqrt"`
- `class_weight="balanced_subsample"`
- Seeds: {", ".join(str(seed) for seed in RANDOM_STATES)}

The only deliberate change across repeated runs is the Random Forest
`random_state`.

## Feature Sets

- `full_14`: original 14 leakage-safe road and parking predictors
- `normalized_12`: drops `road_length_m` and `parking_area_m2`, retaining
  `road_density_km_per_km2` and `parking_area_ratio`

## Seed-Aggregated Results

{chr(10).join(table_lines)}

## Paired Pooled-AP Differences

{seed_lines}

- Mean paired normalized-12 minus full-14 AP delta:
  {paired_delta.mean():+.6f}
- Paired delta standard deviation:
  {paired_delta.std(ddof=1):.6f}
- Normalized-12 higher in:
  {int((paired_delta > 0).sum())}/{len(paired_delta)} seeds

## Interpretation Policy

A consistent positive paired delta would indicate that the normalized
deduplicated representation is not merely benefiting from one favorable
Random Forest seed.

Because `max_features="sqrt"` randomly samples candidate predictors at each
split, duplicate or near-duplicate columns can change which latent information
is available to individual trees. This diagnostic therefore measures the
stability of that algorithm-feature-set interaction.

The result should not be interpreted as evidence that removing a variable has
a causal effect on charging-station placement.

## Output

- `data/processed/ankara_rf_redundancy_seed_stability_metrics.csv`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def save_outputs(
    metrics: pd.DataFrame,
) -> None:
    """Save tabular diagnostic results."""

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
    """Print the paired seed-level AP comparison."""

    wide = metrics.pivot(
        index="random_state",
        columns="feature_set",
        values="pooled_average_precision",
    )

    wide["normalized_minus_full"] = (
        wide["normalized_12"]
        - wide["full_14"]
    )

    print(
        wide.to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )

    print()

    print(
        "Mean normalized-12 minus full-14 AP delta:",
        f"{wide['normalized_minus_full'].mean():+.6f}",
    )

    print(
        "Normalized-12 higher in:",
        (
            f"{int((wide['normalized_minus_full'] > 0).sum())}"
            f"/{len(wide)} seeds"
        ),
    )


def main() -> None:
    """Run the Random Forest redundancy seed-stability diagnostic."""

    print("=" * 70)
    print(
        "VoltSight - Ankara RF Redundancy Seed Stability"
    )
    print("=" * 70)

    dataframe = load_inputs()

    metrics = run_seed_stability(
        dataframe
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

    print("=" * 70)
    print(
        "Ankara RF redundancy seed stability completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
