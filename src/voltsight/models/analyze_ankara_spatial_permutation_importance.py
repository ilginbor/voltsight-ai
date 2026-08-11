from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from voltsight.models.train_ankara_gradient_boosting_baseline import (
    FEATURE_COLUMNS as GB_FEATURE_COLUMNS,
    build_model as build_gradient_boosting_model,
    calculate_balanced_sample_weights,
)
from voltsight.models.train_ankara_random_forest_baseline import (
    FEATURE_COLUMNS as RF_FEATURE_COLUMNS,
    N_SPLITS,
    RANDOM_STATE,
    build_model as build_random_forest_model,
    load_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_permutation_importance.csv"
)

FOLD_DROP_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_permutation_importance_fold_drops.csv"
)

PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_spatial_permutation_importance.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_spatial_permutation_importance_summary.md"
)

TARGET_COLUMN = "has_existing_charging_station"
N_REPEATS = 5

ROAD_FEATURES = (
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
)

PARKING_FEATURES = (
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
)

FEATURE_COLUMNS = tuple(RF_FEATURE_COLUMNS)

MODEL_NAMES = (
    "Random Forest",
    "HistGradientBoosting",
)

MODEL_SEED_OFFSETS = {
    "Random Forest": 0,
    "HistGradientBoosting": 1_000_000,
}


def create_output_directories() -> None:
    """Create directories required by generated outputs."""

    IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_configuration() -> None:
    """Validate that the analysis matches the existing baseline setup."""

    if tuple(GB_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise ValueError(
            "Random Forest and Gradient Boosting feature columns differ."
        )

    combined_features = ROAD_FEATURES + PARKING_FEATURES

    if combined_features != FEATURE_COLUMNS:
        raise ValueError(
            "Road and parking feature groups do not exactly partition "
            "the baseline predictor columns."
        )

    if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("Duplicate predictor columns were found.")

    if N_REPEATS < 1:
        raise ValueError("At least one permutation repeat is required.")


def feature_group_for(feature: str) -> str:
    """Return the feature-family label used in the analysis."""

    if feature in ROAD_FEATURES:
        return "road"

    if feature in PARKING_FEATURES:
        return "parking"

    raise ValueError(f"Unknown baseline feature: {feature}")


def permutation_seed(
    model_name: str,
    fold: int,
    feature_index: int,
    repeat: int,
) -> int:
    """Create a deterministic seed for one fold-feature permutation."""

    if model_name not in MODEL_SEED_OFFSETS:
        raise ValueError(f"Unknown model: {model_name}")

    return int(
        RANDOM_STATE
        + MODEL_SEED_OFFSETS[model_name]
        + fold * 10_000
        + feature_index * 100
        + repeat
    )


def permute_feature_values(
    values: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return a deterministic permutation while preserving all values."""

    result = np.asarray(values).copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(result)

    return result


def calculate_rank_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate ranking metrics used for permutation degradation."""

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Both classes are required for evaluation.")

    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores contain non-finite values.")

    return {
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
    }


def build_model(model_name: str):
    """Create a clone of the corresponding fixed baseline estimator."""

    if model_name == "Random Forest":
        return clone(build_random_forest_model())

    if model_name == "HistGradientBoosting":
        return clone(build_gradient_boosting_model())

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    """Fit one baseline model using its original imbalance handling."""

    if model_name == "Random Forest":
        model.fit(x_train, y_train)
        return

    if model_name == "HistGradientBoosting":
        sample_weights = calculate_balanced_sample_weights(y_train)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
        )
        return

    raise ValueError(f"Unknown model: {model_name}")


def predict_positive_scores(
    model,
    x: pd.DataFrame,
) -> np.ndarray:
    """Return positive-class probability scores from a fitted model."""

    scores = np.asarray(
        model.predict_proba(x)[:, 1],
        dtype=float,
    )

    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite prediction scores.")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Model produced scores outside 0-1.")

    return scores


def summarize_feature_importance(
    *,
    model_name: str,
    y_true: np.ndarray,
    baseline_oof_scores: np.ndarray,
    permuted_oof_scores: dict[tuple[str, int], np.ndarray],
    fold_records: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate pooled and spatial-fold permutation degradation."""

    baseline_metrics = calculate_rank_metrics(
        y_true,
        baseline_oof_scores,
    )

    records: list[dict[str, float | int | str]] = []

    for feature in FEATURE_COLUMNS:
        pooled_ap_drops: list[float] = []
        pooled_roc_drops: list[float] = []
        pooled_permuted_aps: list[float] = []
        pooled_permuted_rocs: list[float] = []

        for repeat in range(N_REPEATS):
            scores = permuted_oof_scores[(feature, repeat)]

            if np.isnan(scores).any():
                raise ValueError(
                    "Missing pooled permuted scores for "
                    f"{model_name} / {feature} / repeat {repeat}."
                )

            metrics = calculate_rank_metrics(
                y_true,
                scores,
            )

            pooled_permuted_aps.append(
                metrics["average_precision"]
            )
            pooled_permuted_rocs.append(
                metrics["roc_auc"]
            )
            pooled_ap_drops.append(
                baseline_metrics["average_precision"]
                - metrics["average_precision"]
            )
            pooled_roc_drops.append(
                baseline_metrics["roc_auc"]
                - metrics["roc_auc"]
            )

        feature_fold_records = fold_records.loc[
            fold_records["feature"].eq(feature)
        ]

        if len(feature_fold_records) != N_SPLITS * N_REPEATS:
            raise ValueError(
                "Unexpected fold-repeat record count for "
                f"{model_name} / {feature}."
            )

        fold_means = (
            feature_fold_records.groupby("cv_fold", sort=True)[
                ["ap_drop", "roc_auc_drop"]
            ]
            .mean()
        )

        records.append(
            {
                "model": model_name,
                "feature": feature,
                "feature_group": feature_group_for(feature),
                "permutation_repeats": N_REPEATS,
                "baseline_pooled_average_precision": (
                    baseline_metrics["average_precision"]
                ),
                "mean_permuted_pooled_average_precision": float(
                    np.mean(pooled_permuted_aps)
                ),
                "mean_pooled_ap_drop": float(
                    np.mean(pooled_ap_drops)
                ),
                "std_pooled_ap_drop": float(
                    np.std(pooled_ap_drops, ddof=1)
                    if N_REPEATS > 1
                    else 0.0
                ),
                "mean_fold_ap_drop": float(
                    fold_means["ap_drop"].mean()
                ),
                "std_fold_ap_drop": float(
                    fold_means["ap_drop"].std(ddof=1)
                ),
                "baseline_pooled_roc_auc": (
                    baseline_metrics["roc_auc"]
                ),
                "mean_permuted_pooled_roc_auc": float(
                    np.mean(pooled_permuted_rocs)
                ),
                "mean_pooled_roc_auc_drop": float(
                    np.mean(pooled_roc_drops)
                ),
                "std_pooled_roc_auc_drop": float(
                    np.std(pooled_roc_drops, ddof=1)
                    if N_REPEATS > 1
                    else 0.0
                ),
                "mean_fold_roc_auc_drop": float(
                    fold_means["roc_auc_drop"].mean()
                ),
                "std_fold_roc_auc_drop": float(
                    fold_means["roc_auc_drop"].std(ddof=1)
                ),
            }
        )

    return pd.DataFrame(records)


def run_model_permutation_importance(
    dataframe: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run within-validation-fold permutation importance for one model."""

    dataframe = dataframe.reset_index(drop=True)
    feature_list = list(FEATURE_COLUMNS)

    baseline_oof_scores = np.full(
        len(dataframe),
        np.nan,
        dtype=float,
    )

    permuted_oof_scores = {
        (feature, repeat): np.full(
            len(dataframe),
            np.nan,
            dtype=float,
        )
        for feature in FEATURE_COLUMNS
        for repeat in range(N_REPEATS)
    }

    fold_records: list[dict[str, float | int | str]] = []

    for fold in range(N_SPLITS):
        train_mask = dataframe["cv_fold"].ne(fold).to_numpy()
        validation_mask = dataframe["cv_fold"].eq(fold).to_numpy()

        train = dataframe.loc[train_mask]
        validation = dataframe.loc[validation_mask]

        if train.empty or validation.empty:
            raise ValueError(f"Fold {fold} is empty.")

        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_validation = validation[TARGET_COLUMN].to_numpy(dtype=int)

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Fold {fold} training data does not contain both classes."
            )

        if len(np.unique(y_validation)) != 2:
            raise ValueError(
                f"Fold {fold} validation data does not contain both classes."
            )

        x_train = train[feature_list]
        x_validation = validation[feature_list]

        model = build_model(model_name)
        fit_model(
            model_name,
            model,
            x_train,
            y_train,
        )

        baseline_scores = predict_positive_scores(
            model,
            x_validation,
        )

        validation_indices = np.flatnonzero(validation_mask)
        baseline_oof_scores[validation_indices] = baseline_scores

        baseline_metrics = calculate_rank_metrics(
            y_validation,
            baseline_scores,
        )

        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            original_values = x_validation[feature].to_numpy(copy=True)

            for repeat in range(N_REPEATS):
                seed = permutation_seed(
                    model_name,
                    fold,
                    feature_index,
                    repeat,
                )

                x_permuted = x_validation.copy()
                x_permuted.loc[:, feature] = permute_feature_values(
                    original_values,
                    seed=seed,
                )

                permuted_scores = predict_positive_scores(
                    model,
                    x_permuted,
                )

                permuted_oof_scores[
                    (feature, repeat)
                ][validation_indices] = permuted_scores

                permuted_metrics = calculate_rank_metrics(
                    y_validation,
                    permuted_scores,
                )

                fold_records.append(
                    {
                        "model": model_name,
                        "cv_fold": fold,
                        "feature": feature,
                        "feature_group": feature_group_for(feature),
                        "repeat": repeat,
                        "validation_rows": len(validation),
                        "validation_positives": int(y_validation.sum()),
                        "baseline_average_precision": (
                            baseline_metrics["average_precision"]
                        ),
                        "permuted_average_precision": (
                            permuted_metrics["average_precision"]
                        ),
                        "ap_drop": (
                            baseline_metrics["average_precision"]
                            - permuted_metrics["average_precision"]
                        ),
                        "baseline_roc_auc": baseline_metrics["roc_auc"],
                        "permuted_roc_auc": permuted_metrics["roc_auc"],
                        "roc_auc_drop": (
                            baseline_metrics["roc_auc"]
                            - permuted_metrics["roc_auc"]
                        ),
                    }
                )

    if np.isnan(baseline_oof_scores).any():
        raise ValueError(
            f"Missing baseline OOF scores for {model_name}."
        )

    fold_dataframe = pd.DataFrame(fold_records)

    importance = summarize_feature_importance(
        model_name=model_name,
        y_true=dataframe[TARGET_COLUMN].to_numpy(dtype=int),
        baseline_oof_scores=baseline_oof_scores,
        permuted_oof_scores=permuted_oof_scores,
        fold_records=fold_dataframe,
    )

    return importance, fold_dataframe


def validate_outputs(
    dataframe: pd.DataFrame,
    importance: pd.DataFrame,
    fold_drops: pd.DataFrame,
) -> None:
    """Validate permutation-importance outputs before saving."""

    expected_importance_rows = len(MODEL_NAMES) * len(FEATURE_COLUMNS)

    if len(importance) != expected_importance_rows:
        raise ValueError("Unexpected importance row count.")

    if importance[["model", "feature"]].duplicated().any():
        raise ValueError("Duplicate model-feature importance rows found.")

    expected_fold_rows = (
        len(MODEL_NAMES)
        * N_SPLITS
        * len(FEATURE_COLUMNS)
        * N_REPEATS
    )

    if len(fold_drops) != expected_fold_rows:
        raise ValueError("Unexpected fold-drop row count.")

    if fold_drops[
        ["model", "cv_fold", "feature", "repeat"]
    ].duplicated().any():
        raise ValueError("Duplicate fold permutation records found.")

    numeric_columns = [
        column
        for column in importance.columns
        if column not in {"model", "feature", "feature_group"}
    ]

    numeric_values = importance[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Importance output contains non-finite values.")

    fold_numeric_columns = [
        "baseline_average_precision",
        "permuted_average_precision",
        "ap_drop",
        "baseline_roc_auc",
        "permuted_roc_auc",
        "roc_auc_drop",
    ]

    if not np.isfinite(
        fold_drops[fold_numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Fold-drop output contains non-finite values.")

    if set(dataframe["cv_fold"].unique()) != set(range(N_SPLITS)):
        raise ValueError("Unexpected spatial fold identifiers.")


def save_outputs(
    importance: pd.DataFrame,
    fold_drops: pd.DataFrame,
) -> None:
    """Save tabular permutation-importance results."""

    importance.sort_values(
        ["model", "mean_pooled_ap_drop", "feature"],
        ascending=[True, False, True],
        kind="stable",
    ).to_csv(
        IMPORTANCE_PATH,
        index=False,
        encoding="utf-8",
    )

    fold_drops.sort_values(
        ["model", "cv_fold", "feature", "repeat"],
        kind="stable",
    ).to_csv(
        FOLD_DROP_PATH,
        index=False,
        encoding="utf-8",
    )


def create_importance_plot(
    importance: pd.DataFrame,
) -> None:
    """Plot pooled AP degradation for both nonlinear baseline models."""

    ordering = (
        importance.groupby("feature")["mean_pooled_ap_drop"]
        .max()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    figure, axis = plt.subplots(figsize=(12, 9))

    y_positions = np.arange(len(ordering), dtype=float)
    bar_height = 0.36

    for offset, model_name in zip(
        (-bar_height / 2, bar_height / 2),
        MODEL_NAMES,
    ):
        model_values = (
            importance.loc[importance["model"].eq(model_name)]
            .set_index("feature")
            .reindex(ordering)["mean_pooled_ap_drop"]
            .to_numpy(dtype=float)
        )

        axis.barh(
            y_positions + offset,
            model_values,
            height=bar_height,
            label=model_name,
        )

    axis.axvline(0, linewidth=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(ordering)
    axis.set_xlabel("Mean pooled AP drop after validation-fold permutation")
    axis.set_title("Ankara Spatial Permutation Importance")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def format_importance_table(
    importance: pd.DataFrame,
) -> str:
    """Create a Markdown table ordered by model and AP degradation."""

    lines = [
        "| Model | Feature | Group | Pooled AP drop | Fold AP drop | Fold AP std | ROC-AUC drop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for model_name in MODEL_NAMES:
        model_rows = (
            importance.loc[importance["model"].eq(model_name)]
            .sort_values(
                ["mean_pooled_ap_drop", "feature"],
                ascending=[False, True],
                kind="stable",
            )
        )

        for row in model_rows.itertuples(index=False):
            lines.append(
                "| "
                f"{row.model} | `{row.feature}` | {row.feature_group} | "
                f"{row.mean_pooled_ap_drop:.6f} | "
                f"{row.mean_fold_ap_drop:.6f} | "
                f"{row.std_fold_ap_drop:.6f} | "
                f"{row.mean_pooled_roc_auc_drop:.6f} |"
            )

    return "\n".join(lines)


def create_summary(
    dataframe: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Create Markdown documentation for the analysis."""

    table = format_importance_table(importance)

    baseline_lines = []

    for model_name in MODEL_NAMES:
        model_rows = importance.loc[
            importance["model"].eq(model_name)
        ]

        baseline_lines.append(
            f"- {model_name}: pooled AP "
            f"{model_rows['baseline_pooled_average_precision'].iloc[0]:.6f}, "
            "pooled ROC-AUC "
            f"{model_rows['baseline_pooled_roc_auc'].iloc[0]:.6f}"
        )

    top_lines = []

    for model_name in MODEL_NAMES:
        top_rows = (
            importance.loc[importance["model"].eq(model_name)]
            .sort_values(
                ["mean_pooled_ap_drop", "feature"],
                ascending=[False, True],
                kind="stable",
            )
            .head(5)
        )

        formatted = ", ".join(
            f"`{row.feature}` ({row.mean_pooled_ap_drop:+.6f})"
            for row in top_rows.itertuples(index=False)
        )

        top_lines.append(
            f"- {model_name}: {formatted}"
        )

    summary = f"""# Ankara Spatial Permutation Importance

## Purpose

This analysis measures validation-set dependence on individual road and
parking predictors for the two nonlinear Ankara baselines.

The Random Forest and HistGradientBoosting configurations are reused without
hyperparameter tuning. The same predefined 5-km spatial cross-validation
folds are used throughout.

## Dataset

- Rows: {len(dataframe):,}
- Positive station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Predictor features: {len(FEATURE_COLUMNS)}
- Spatial folds: {N_SPLITS}
- Permutation repeats per feature per fold: {N_REPEATS}

## Method

For each model and spatial fold:

1. Fit the baseline model on the other four folds.
2. Score the untouched validation fold and calculate baseline AP / ROC-AUC.
3. Permute one predictor only inside that validation fold.
4. Re-score the already-fitted model without refitting it.
5. Record the decrease in AP and ROC-AUC.
6. Repeat each feature permutation {N_REPEATS} times with deterministic seeds.

The pooled metric also reconstructs a complete permuted OOF score vector for
each feature and repeat before calculating the pooled degradation.

A positive drop means that shuffling the feature reduced validation ranking
performance. A value near zero indicates little measurable dependence under
this experiment. Negative values are retained rather than clipped because
permutation noise or correlated predictors can occasionally improve the
metric by chance.

## Baseline Spatial OOF Metrics

{chr(10).join(baseline_lines)}

## Highest Pooled-AP Degradation

{chr(10).join(top_lines)}

## Full Results

{table}

## Interpretation Policy

Permutation importance is model- and dataset-dependent. It is not a causal
estimate of the real-world effect of a road or parking variable on charging
station placement.

Correlated predictors can substitute for one another, reducing the apparent
importance of each individual variable. Sparse or incomplete OSM parking
coverage can also affect the measured parking importance.

The procedure uses validation-fold permutation rather than training-set
impurity importance, so the reported degradation is evaluated on held-out
spatial folds. However, neighboring spatial blocks can still occur in
different folds; the existing spatial CV design reduces local dependence but
does not eliminate all possible spatial autocorrelation.

## Outputs

- `data/processed/ankara_spatial_permutation_importance.csv`
- `data/processed/ankara_spatial_permutation_importance_fold_drops.csv`
- `docs/ankara_spatial_permutation_importance.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    importance: pd.DataFrame,
) -> None:
    """Print the highest AP-degradation features for each model."""

    print("-" * 70)

    for model_name in MODEL_NAMES:
        print(model_name)

        model_rows = (
            importance.loc[importance["model"].eq(model_name)]
            .sort_values(
                ["mean_pooled_ap_drop", "feature"],
                ascending=[False, True],
                kind="stable",
            )
            .head(10)
        )

        print(
            model_rows[
                [
                    "feature",
                    "feature_group",
                    "mean_pooled_ap_drop",
                    "mean_fold_ap_drop",
                    "std_fold_ap_drop",
                    "mean_pooled_roc_auc_drop",
                ]
            ].to_string(index=False)
        )
        print()


def main() -> None:
    """Run Ankara spatial permutation importance."""

    print("=" * 70)
    print("VoltSight - Ankara Spatial Permutation Importance")
    print("=" * 70)

    validate_configuration()
    create_output_directories()

    dataframe = load_inputs().reset_index(drop=True)

    importance_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        print(f"Running {model_name}...")

        importance, fold_drops = run_model_permutation_importance(
            dataframe,
            model_name,
        )

        importance_frames.append(importance)
        fold_frames.append(fold_drops)

    importance = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    fold_drops = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    validate_outputs(
        dataframe,
        importance,
        fold_drops,
    )

    save_outputs(
        importance,
        fold_drops,
    )

    create_importance_plot(importance)
    create_summary(dataframe, importance)
    print_results(importance)

    print("=" * 70)
    print("Ankara spatial permutation importance completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
