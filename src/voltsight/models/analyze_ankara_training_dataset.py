from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

FEATURE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_training_feature_summary.csv"
)

CLASS_BALANCE_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_training_class_balance.png"
)

FEATURE_PROFILE_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_training_positive_feature_profile.png"
)

CORRELATION_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_training_feature_correlation.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_training_eda_summary.md"
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


def create_output_directories() -> None:
    """Create all output directories."""

    FEATURE_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_training_dataset() -> pd.DataFrame:
    """Load and validate leakage-safe Ankara training data."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Ankara training dataset was not found:\n"
            f"{INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        dtype={
            "grid_id": str,
        },
    )

    if dataframe.empty:
        raise ValueError(
            "The Ankara training dataset is empty."
        )

    required_columns = {
        "grid_id",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate training grid IDs were found."
        )

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    target_values = set(
        dataframe[TARGET_COLUMN]
        .astype(int)
        .unique()
    )

    if target_values != {0, 1}:
        raise ValueError(
            "Training target must contain both 0 and 1."
        )

    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN]
        .astype(int)
    )

    for column in FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if dataframe[column].isna().any():
            raise ValueError(
                f"Missing values found in {column}."
            )

        values = dataframe[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    print(
        "Loaded training rows: "
        f"{len(dataframe):,}"
    )

    print(
        "Feature count: "
        f"{len(FEATURE_COLUMNS):,}"
    )

    return dataframe


def empirical_percentile(
    reference_values: np.ndarray,
    value: float,
) -> float:
    """Return value percentile relative to a reference distribution."""

    if reference_values.size == 0:
        raise ValueError(
            "Reference distribution is empty."
        )

    return float(
        np.mean(
            reference_values <= value
        )
        * 100.0
    )


def calculate_feature_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compare positive and negative feature distributions."""

    positives = dataframe.loc[
        dataframe[TARGET_COLUMN] == 1
    ]

    negatives = dataframe.loc[
        dataframe[TARGET_COLUMN] == 0
    ]

    records: list[dict[str, float | str]] = []

    for feature in FEATURE_COLUMNS:
        positive_values = (
            positives[feature]
            .to_numpy(dtype=float)
        )

        negative_values = (
            negatives[feature]
            .to_numpy(dtype=float)
        )

        positive_median = float(
            np.median(
                positive_values
            )
        )

        negative_median = float(
            np.median(
                negative_values
            )
        )

        positive_mean = float(
            np.mean(
                positive_values
            )
        )

        negative_mean = float(
            np.mean(
                negative_values
            )
        )

        positive_median_percentile = (
            empirical_percentile(
                negative_values,
                positive_median,
            )
        )

        negative_std = float(
            np.std(
                negative_values,
                ddof=0,
            )
        )

        if negative_std > 0:
            standardized_mean_difference = (
                positive_mean
                - negative_mean
            ) / negative_std
        else:
            standardized_mean_difference = 0.0

        records.append(
            {
                "feature": feature,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "positive_median": positive_median,
                "negative_median": negative_median,
                "positive_median_percentile_vs_negative": (
                    positive_median_percentile
                ),
                "standardized_mean_difference": (
                    standardized_mean_difference
                ),
            }
        )

    summary = pd.DataFrame(
        records
    )

    summary[
        "absolute_standardized_difference"
    ] = (
        summary[
            "standardized_mean_difference"
        ]
        .abs()
    )

    summary = summary.sort_values(
        by=[
            "absolute_standardized_difference",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return summary


def validate_feature_summary(
    summary: pd.DataFrame,
) -> None:
    """Validate generated feature statistics."""

    if len(summary) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unexpected feature-summary row count."
        )

    if summary["feature"].duplicated().any():
        raise ValueError(
            "Duplicate feature names found."
        )

    numeric_columns = [
        column
        for column in summary.columns
        if column != "feature"
    ]

    for column in numeric_columns:
        values = summary[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite summary values in {column}."
            )

    percentile_values = summary[
        "positive_median_percentile_vs_negative"
    ]

    if not percentile_values.between(
        0,
        100,
    ).all():
        raise ValueError(
            "Feature percentile is outside 0-100."
        )

    print(
        "Feature summary validation completed successfully."
    )


def create_class_balance_plot(
    dataframe: pd.DataFrame,
) -> None:
    """Create class-distribution chart."""

    class_counts = (
        dataframe[
            TARGET_COLUMN
        ]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0,
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    bars = axis.bar(
        [
            "No existing station",
            "Existing station",
        ],
        class_counts.values,
    )

    axis.set_title(
        "Ankara Existing-Station Training Class Distribution"
    )

    axis.set_ylabel(
        "Grid cells"
    )

    for bar, count in zip(
        bars,
        class_counts.values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        CLASS_BALANCE_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_feature_profile_plot(
    feature_summary: pd.DataFrame,
) -> None:
    """Plot positive medians as percentiles of negative cells."""

    profile = feature_summary.sort_values(
        "positive_median_percentile_vs_negative",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.barh(
        profile["feature"],
        profile[
            "positive_median_percentile_vs_negative"
        ],
    )

    axis.axvline(
        50,
        linestyle="--",
        linewidth=1.2,
        label="Negative-cell median",
    )

    axis.set_xlim(
        0,
        100,
    )

    axis.set_xlabel(
        "Positive median percentile within negative-cell distribution"
    )

    axis.set_title(
        "Ankara Existing-Station Feature Profile"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FEATURE_PROFILE_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_correlation_plot(
    dataframe: pd.DataFrame,
) -> None:
    """Create predictor correlation matrix."""

    correlation = (
        dataframe[
            list(FEATURE_COLUMNS)
        ]
        .corr(
            method="spearman"
        )
    )

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    image = axis.imshow(
        correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    axis.set_xticks(
        range(
            len(FEATURE_COLUMNS)
        )
    )

    axis.set_yticks(
        range(
            len(FEATURE_COLUMNS)
        )
    )

    axis.set_xticklabels(
        FEATURE_COLUMNS,
        rotation=90,
        fontsize=8,
    )

    axis.set_yticklabels(
        FEATURE_COLUMNS,
        fontsize=8,
    )

    axis.set_title(
        "Ankara Training Feature Spearman Correlation"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Spearman correlation"
    )

    figure.tight_layout()

    figure.savefig(
        CORRELATION_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_feature_summary(
    feature_summary: pd.DataFrame,
) -> None:
    """Save feature statistics CSV."""

    if FEATURE_SUMMARY_PATH.exists():
        FEATURE_SUMMARY_PATH.unlink()

    feature_summary.to_csv(
        FEATURE_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> None:
    """Create Ankara training EDA summary."""

    positive_count = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )

    negative_count = (
        len(dataframe)
        - positive_count
    )

    positive_rate = (
        positive_count
        / len(dataframe)
        * 100
    )

    imbalance_ratio = (
        negative_count
        / positive_count
    )

    strongest = feature_summary.head(
        8
    )

    strongest_lines = "\n".join(
        (
            f"- `{row.feature}`: "
            f"positive median percentile "
            f"{row.positive_median_percentile_vs_negative:.2f}, "
            f"standardized mean difference "
            f"{row.standardized_mean_difference:.3f}"
        )
        for row in strongest.itertuples(
            index=False
        )
    )

    summary = f"""# Ankara Training Dataset EDA

## Dataset

- Rows: {len(dataframe):,}
- Predictor features: {len(FEATURE_COLUMNS):,}
- Existing-station grid cells: {positive_count:,}
- Non-station grid cells: {negative_count:,}
- Positive prevalence: {positive_rate:.4f}%
- Negative-to-positive ratio: {imbalance_ratio:,.2f}:1
- Missing predictor values: 0

## Interpretation

The existing-station learning problem is extremely imbalanced.

Accuracy is therefore not an appropriate primary evaluation metric.
A classifier predicting every cell as negative would achieve a very
high apparent accuracy while detecting no charging-station cells.

Future model evaluation should emphasize:

- precision
- recall
- F1
- average precision / PR-AUC
- ROC-AUC as a secondary metric
- spatial cross-validation stability

## Strongest Univariate Differences

{strongest_lines}

The standardized mean difference is used only as a descriptive
screening statistic. It is not interpreted as causal importance.

The positive-median percentile reports where the median existing-
station cell falls inside the distribution of non-station cells.

Values far above 50 indicate that existing-station cells tend to have
higher feature values; values far below 50 indicate lower values.

## Generated Outputs

- `data/processed/ankara_training_feature_summary.csv`
- `docs/ankara_training_class_balance.png`
- `docs/ankara_training_positive_feature_profile.png`
- `docs/ankara_training_feature_correlation.png`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_statistics(
    dataframe: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> None:
    """Print key EDA statistics."""

    positive_count = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )

    negative_count = (
        len(dataframe)
        - positive_count
    )

    print("-" * 70)

    print(
        "Training rows: "
        f"{len(dataframe):,}"
    )

    print(
        "Positive rows: "
        f"{positive_count:,}"
    )

    print(
        "Negative rows: "
        f"{negative_count:,}"
    )

    print(
        "Positive prevalence: "
        f"{positive_count / len(dataframe) * 100:.4f}%"
    )

    print(
        "Negative / positive ratio: "
        f"{negative_count / positive_count:,.2f}:1"
    )

    print()

    print(
        "Top feature differences:"
    )

    print(
        feature_summary[
            [
                "feature",
                "positive_median_percentile_vs_negative",
                "standardized_mean_difference",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


def main() -> None:
    """Run Ankara training-data exploratory analysis."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Training Dataset EDA"
    )

    print("=" * 70)

    create_output_directories()

    dataframe = (
        load_training_dataset()
    )

    feature_summary = (
        calculate_feature_summary(
            dataframe
        )
    )

    validate_feature_summary(
        feature_summary
    )

    save_feature_summary(
        feature_summary
    )

    create_class_balance_plot(
        dataframe
    )

    create_feature_profile_plot(
        feature_summary
    )

    create_correlation_plot(
        dataframe
    )

    create_summary(
        dataframe,
        feature_summary,
    )

    print_statistics(
        dataframe,
        feature_summary,
    )

    print("=" * 70)

    print(
        "Ankara training dataset EDA "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
