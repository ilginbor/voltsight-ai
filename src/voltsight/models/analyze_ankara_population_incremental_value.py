from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
POPULATION_PATH = PROJECT_ROOT / "data" / "processed" / "ankara_grid_population_features.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ankara_population_incremental_value_metrics.csv"
FOLD_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ankara_population_incremental_value_fold_metrics.csv"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "ankara_population_incremental_value_oof_predictions.csv"
PLOT_PATH = PROJECT_ROOT / "docs" / "ankara_population_incremental_value.png"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "ankara_population_incremental_value_summary.md"

NORMALIZED_BASE_FEATURES = tuple(
    feature
    for feature in BASELINE_FEATURE_COLUMNS
    if feature not in {"road_length_m", "parking_area_m2"}
)

POPULATION_COLUMNS = (
    "population_count",
    "population_density_per_km2",
    "population_within_1000m",
    "population_within_2000m",
)

FEATURE_SETS = {
    "normalized_12": NORMALIZED_BASE_FEATURES,
    "normalized_12_plus_local_population": NORMALIZED_BASE_FEATURES + ("population_count",),
    "normalized_12_plus_population_context": NORMALIZED_BASE_FEATURES
    + (
        "population_count",
        "population_within_1000m",
        "population_within_2000m",
    ),
}

FEATURE_SET_LABELS = {
    "normalized_12": "Normalized 12",
    "normalized_12_plus_local_population": "Normalized 12 + local population",
    "normalized_12_plus_population_context": "Normalized 12 + local + 1 km + 2 km population",
}

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


def validate_feature_sets() -> None:
    if len(NORMALIZED_BASE_FEATURES) != 12:
        raise ValueError("Expected 12 normalized baseline predictors.")

    expected_counts = {
        "normalized_12": 12,
        "normalized_12_plus_local_population": 13,
        "normalized_12_plus_population_context": 15,
    }

    for name, expected in expected_counts.items():
        if len(FEATURE_SETS[name]) != expected:
            raise ValueError(f"{name} has an unexpected predictor count.")

    for features in FEATURE_SETS.values():
        if {
            "population_count",
            "population_density_per_km2",
        }.issubset(features):
            raise ValueError(
                "population_count and population_density_per_km2 must not be used together."
            )


def validate_population_frame(population: pd.DataFrame) -> pd.DataFrame:
    required = {"grid_id", *POPULATION_COLUMNS}
    missing = required - set(population.columns)
    if missing:
        raise ValueError(f"Population feature columns are missing: {sorted(missing)}")

    population = population[["grid_id", *POPULATION_COLUMNS]].copy()
    population["grid_id"] = population["grid_id"].astype(str)

    if population["grid_id"].duplicated().any():
        raise ValueError("Duplicate population grid IDs were found.")

    for column in POPULATION_COLUMNS:
        population[column] = pd.to_numeric(population[column], errors="coerce")
        values = population[column].to_numpy(dtype=float)
        if population[column].isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Invalid population values found in {column}.")
        if (values < -1e-8).any():
            raise ValueError(f"Negative population values found in {column}.")

    expected_density = population["population_count"].to_numpy(dtype=float) / 0.25
    if not np.allclose(
        population["population_density_per_km2"].to_numpy(dtype=float),
        expected_density,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError(
            "Population density is not the expected deterministic transform of local population."
        )

    local = population["population_count"].to_numpy(dtype=float)
    within_1km = population["population_within_1000m"].to_numpy(dtype=float)
    within_2km = population["population_within_2000m"].to_numpy(dtype=float)

    if (within_1km + 1e-8 < local).any():
        raise ValueError("1-km population cannot be below local population.")
    if (within_2km + 1e-8 < within_1km).any():
        raise ValueError("2-km population cannot be below 1-km population.")

    return population


def attach_population_features(
    baseline: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    if baseline["grid_id"].duplicated().any():
        raise ValueError("Duplicate baseline grid IDs were found.")

    population = validate_population_frame(population)
    baseline = baseline.copy()
    baseline["grid_id"] = baseline["grid_id"].astype(str)

    merged = baseline.merge(
        population,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(baseline):
        raise ValueError("Population merge changed the baseline row count.")

    if merged[list(POPULATION_COLUMNS)].isna().any().any():
        raise ValueError("Not every baseline row matched population features.")

    return merged


def load_analysis_frame() -> pd.DataFrame:
    validate_feature_sets()
    if not POPULATION_PATH.exists():
        raise FileNotFoundError(f"Population feature dataset not found: {POPULATION_PATH}")

    baseline = load_baseline_inputs()
    population = pd.read_csv(POPULATION_PATH, dtype={"grid_id": str})
    return attach_population_features(baseline, population)


def build_model(model_name: str):
    if model_name == "logistic_regression":
        return build_logistic_pipeline()
    if model_name == "random_forest":
        return build_random_forest_model()
    if model_name == "hist_gradient_boosting":
        return build_gradient_boosting_model()
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model_name: str,
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    if model_name == "hist_gradient_boosting":
        model.fit(
            x_train,
            y_train,
            sample_weight=calculate_balanced_sample_weights(y_train),
        )
        return
    model.fit(x_train, y_train)


def calculate_ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("Both target classes are required for evaluation.")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores contain non-finite values.")

    top_one = calculate_top_fraction_metrics(y_true, scores, fraction=0.01)
    top_five = calculate_top_fraction_metrics(y_true, scores, fraction=0.05)

    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "top_1_percent_recall": float(top_one["recall"]),
        "top_5_percent_recall": float(top_five["recall"]),
    }


def run_configuration(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    missing = set(feature_columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Experiment predictors are missing: {sorted(missing)}")

    oof = dataframe[
        ["grid_id", TARGET_COLUMN, "spatial_block_id", "cv_fold"]
    ].copy()
    oof["score"] = np.nan
    fold_records: list[dict[str, float | int | str]] = []

    for fold in range(N_SPLITS):
        train_mask = dataframe["cv_fold"] != fold
        validation_mask = dataframe["cv_fold"] == fold
        train = dataframe.loc[train_mask]
        validation = dataframe.loc[validation_mask]

        if train.empty or validation.empty:
            raise ValueError(f"Fold {fold} is empty.")

        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_validation = validation[TARGET_COLUMN].to_numpy(dtype=int)

        if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
            raise ValueError(f"Fold {fold} does not contain both classes.")

        model = build_model(model_name)
        fit_model(
            model_name,
            model,
            train[list(feature_columns)],
            y_train,
        )

        scores = model.predict_proba(validation[list(feature_columns)])[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError(f"{model_name} fold {fold} produced invalid scores.")

        oof.loc[validation_mask, "score"] = scores
        fold_metrics = calculate_ranking_metrics(y_validation, scores)

        fold_records.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature_set": feature_set_name,
                "feature_set_label": FEATURE_SET_LABELS[feature_set_name],
                "feature_count": len(feature_columns),
                "cv_fold": fold,
                "validation_rows": len(validation),
                "validation_positives": int(y_validation.sum()),
                **fold_metrics,
            }
        )

    if oof["score"].isna().any():
        raise ValueError("OOF predictions are incomplete.")

    y_true = oof[TARGET_COLUMN].to_numpy(dtype=int)
    scores = oof["score"].to_numpy(dtype=float)
    pooled = calculate_ranking_metrics(y_true, scores)
    fold_metrics_frame = pd.DataFrame(fold_records)

    aggregate: dict[str, float | int | str] = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "feature_set": feature_set_name,
        "feature_set_label": FEATURE_SET_LABELS[feature_set_name],
        "feature_count": len(feature_columns),
        "pooled_average_precision": pooled["average_precision"],
        "mean_fold_average_precision": float(
            fold_metrics_frame["average_precision"].mean()
        ),
        "std_fold_average_precision": float(
            fold_metrics_frame["average_precision"].std(ddof=1)
        ),
        "pooled_roc_auc": pooled["roc_auc"],
        "mean_fold_roc_auc": float(fold_metrics_frame["roc_auc"].mean()),
        "std_fold_roc_auc": float(fold_metrics_frame["roc_auc"].std(ddof=1)),
        "top_1_percent_recall": pooled["top_1_percent_recall"],
        "top_5_percent_recall": pooled["top_5_percent_recall"],
    }

    oof["model"] = model_name
    oof["feature_set"] = feature_set_name
    return oof, fold_metrics_frame, aggregate


def add_incremental_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    columns = (
        "pooled_average_precision",
        "mean_fold_average_precision",
        "pooled_roc_auc",
        "top_1_percent_recall",
        "top_5_percent_recall",
    )

    for column in columns:
        result[f"delta_{column}"] = np.nan

    for model_name in MODEL_ORDER:
        model_mask = result["model"] == model_name
        baseline_rows = result.loc[
            model_mask & (result["feature_set"] == "normalized_12")
        ]
        if len(baseline_rows) != 1:
            raise ValueError(f"Expected one normalized-12 baseline for {model_name}.")
        baseline = baseline_rows.iloc[0]

        for column in columns:
            result.loc[model_mask, f"delta_{column}"] = (
                result.loc[model_mask, column] - float(baseline[column])
            )

    return result


def run_experiment(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregates = []
    fold_frames = []
    oof_frames = []

    for model_name in MODEL_ORDER:
        print(f"Running {MODEL_LABELS[model_name]}...")
        for feature_set_name, feature_columns in FEATURE_SETS.items():
            print(f"  {FEATURE_SET_LABELS[feature_set_name]}")
            oof, fold_metrics, aggregate = run_configuration(
                dataframe,
                model_name=model_name,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
            )
            aggregates.append(aggregate)
            fold_frames.append(fold_metrics)
            oof_frames.append(oof)

    metrics = add_incremental_deltas(pd.DataFrame(aggregates))
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    return metrics, fold_metrics, oof


def create_plot(metrics: pd.DataFrame) -> None:
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    for index, feature_set_name in enumerate(FEATURE_SETS):
        values = []
        for model_name in MODEL_ORDER:
            row = metrics.loc[
                (metrics["model"] == model_name)
                & (metrics["feature_set"] == feature_set_name)
            ]
            if len(row) != 1:
                raise ValueError("Unexpected metric row count while plotting.")
            values.append(float(row.iloc[0]["pooled_average_precision"]))

        axis.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=FEATURE_SET_LABELS[feature_set_name],
        )

    axis.set_xticks(x)
    axis.set_xticklabels([MODEL_LABELS[name] for name in MODEL_ORDER])
    axis.set_ylabel("Pooled spatial OOF average precision")
    axis.set_title("Ankara Population Incremental Value")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_summary(dataframe: pd.DataFrame, metrics: pd.DataFrame) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    table_lines = [
        "| Model | Feature set | Features | Pooled AP | Delta AP | Mean fold AP | Fold AP std | ROC-AUC | Top 1% recall | Delta top 1% | Top 5% recall | Delta top 5% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name in MODEL_ORDER:
        for feature_set_name in FEATURE_SETS:
            row = metrics.loc[
                (metrics["model"] == model_name)
                & (metrics["feature_set"] == feature_set_name)
            ].iloc[0]
            table_lines.append(
                "| "
                f"{row['model_label']} | {row['feature_set_label']} | "
                f"{int(row['feature_count'])} | {row['pooled_average_precision']:.6f} | "
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
            (metrics["model"] == model_name)
            & (metrics["feature_set"] == "normalized_12_plus_population_context")
        ].iloc[0]
        delta_lines.append(
            f"- {MODEL_LABELS[model_name]}: pooled AP delta "
            f"{row['delta_pooled_average_precision']:+.6f}, top-1% recall delta "
            f"{row['delta_top_1_percent_recall']:+.6f}, top-5% recall delta "
            f"{row['delta_top_5_percent_recall']:+.6f}."
        )

    summary = f"""# Ankara Population Incremental Value

## Purpose

This experiment tests whether WorldPop-derived residential population adds
predictive ranking information beyond the deduplicated road-and-parking
baseline.

The experiment is incremental rather than a new tuned model search.

## Dataset

- Rows: {len(dataframe):,}
- Positive existing-station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Spatial folds: {N_SPLITS}
- Spatial block size: 5 km
- Population source: WorldPop 2025 constrained R2024B

## Feature Sets

### Normalized 12

The previously audited deduplicated road-and-parking baseline. It excludes
`road_length_m` and `parking_area_m2` while retaining their normalized
counterparts.

### Normalized 12 + Local Population

Adds `population_count` only.

`population_density_per_km2` is intentionally excluded because every analysis
cell is 0.25 km2, making density a deterministic scale transform of local
population count.

### Normalized 12 + Population Context

Adds:

- `population_count`
- `population_within_1000m`
- `population_within_2000m`

The neighborhood variables test whether surrounding residential demand adds
information beyond the local 500-m cell.

## Models

The existing untuned Logistic Regression, Random Forest, and
HistGradientBoosting configurations are reused unchanged. The same predefined
5-km spatial folds and the same class-imbalance treatments are retained. No
hyperparameter search is performed.

## Spatial OOF Results

{chr(10).join(table_lines)}

## Full Population-Context Delta Against Normalized 12

{chr(10).join(delta_lines)}

## Interpretation Policy

Average precision is primary because only a very small fraction of Ankara grid
cells contain known existing charging stations. Top-1% and top-5% recall are
also reported because VoltSight is a candidate-ranking system.

A positive delta means the population feature set improved spatial OOF ranking
under this experiment. It is predictive evidence, not a causal estimate of the
real-world effect of population on station placement.

Population represents modeled residential demand only. It does not directly
capture employment, commuting, retail activity, tourism, traffic volume,
vehicle ownership, or electricity-grid capacity.

Only 46 positive station cells are available, so fold-level variability must
be considered alongside pooled metrics. The existing spatial block design
reduces local dependence but does not eliminate all spatial autocorrelation.

The historical full-14 baselines remain historical references. This experiment
uses normalized-12 as the deduplicated baseline for future feature-family
evaluation.

## Outputs

- `data/processed/{METRICS_PATH.name}`
- `data/processed/{FOLD_METRICS_PATH.name}`
- `data/processed/{OOF_PATH.name}`
- `docs/{PLOT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def save_outputs(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8")
    fold_metrics.to_csv(FOLD_METRICS_PATH, index=False, encoding="utf-8")
    oof.to_csv(OOF_PATH, index=False, encoding="utf-8")


def print_results(metrics: pd.DataFrame) -> None:
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
        metrics[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main() -> None:
    print("=" * 70)
    print("VoltSight - Ankara Population Incremental Value")
    print("=" * 70)

    dataframe = load_analysis_frame()
    metrics, fold_metrics, oof = run_experiment(dataframe)
    save_outputs(metrics, fold_metrics, oof)
    create_plot(metrics)
    create_summary(dataframe, metrics)
    print_results(metrics)

    print("=" * 70)
    print("Ankara population incremental value completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
