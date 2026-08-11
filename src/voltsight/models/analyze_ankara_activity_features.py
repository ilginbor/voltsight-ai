from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ACTIVITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_activity_features.csv"
)

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

POPULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_population_features.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_activity_feature_audit_summary.md"
)

DISTRIBUTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_feature_audit_distributions.csv"
)

TARGET_COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_feature_target_comparison.csv"
)

CORRELATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_feature_correlations.csv"
)

REDUNDANCY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_activity_feature_redundancy_pairs.csv"
)

CORRELATION_PLOT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_activity_feature_correlations.png"
)

TARGET_COLUMN = "has_existing_charging_station"

ACTIVITY_FEATURE_COLUMNS = (
    "poi_count",
    "retail_commercial_count",
    "education_count",
    "healthcare_count",
    "transport_activity_count",
    "poi_count_within_1000m",
    "poi_count_within_2000m",
    "retail_commercial_within_1000m",
    "education_within_1000m",
    "healthcare_within_1000m",
    "transport_activity_within_1000m",
)

OPTIONAL_POPULATION_COLUMNS = (
    "population_count",
    "population_within_1000m",
    "population_within_2000m",
)

REDUNDANCY_THRESHOLD = 0.90


def validate_activity_frame(
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the Ankara activity feature table."""

    required = {
        "grid_id",
        *ACTIVITY_FEATURE_COLUMNS,
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
            *ACTIVITY_FEATURE_COLUMNS,
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

    for column in ACTIVITY_FEATURE_COLUMNS:
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
                f"Invalid values found in {column}."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"Negative values found in {column}."
            )

    if (
        result[
            "poi_count_within_1000m"
        ]
        < result[
            "poi_count"
        ]
    ).any():
        raise ValueError(
            "1-km POI count cannot be below local POI count."
        )

    if (
        result[
            "poi_count_within_2000m"
        ]
        < result[
            "poi_count_within_1000m"
        ]
    ).any():
        raise ValueError(
            "2-km POI count cannot be below 1-km POI count."
        )

    return result


def load_inputs() -> pd.DataFrame:
    """Load activity features, training labels, and optional population context."""

    if not ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            f"Activity features not found: {ACTIVITY_PATH}"
        )

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_PATH}"
        )

    activity = validate_activity_frame(
        pd.read_csv(
            ACTIVITY_PATH,
            dtype={
                "grid_id": str,
            },
        )
    )

    training = pd.read_csv(
        TRAINING_PATH,
        dtype={
            "grid_id": str,
        },
    )

    required_training = {
        "grid_id",
        TARGET_COLUMN,
    }

    missing_training = (
        required_training
        - set(
            training.columns
        )
    )

    if missing_training:
        raise ValueError(
            "Training columns are missing: "
            f"{sorted(missing_training)}"
        )

    if training[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate training grid IDs were found."
        )

    training[
        TARGET_COLUMN
    ] = pd.to_numeric(
        training[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    if training[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            "Training target contains invalid values."
        )

    training[
        TARGET_COLUMN
    ] = training[
        TARGET_COLUMN
    ].astype(int)

    if set(
        training[
            TARGET_COLUMN
        ].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Training target must contain both 0 and 1."
        )

    merged = training[
        [
            "grid_id",
            TARGET_COLUMN,
        ]
    ].merge(
        activity,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        training
    ):
        raise ValueError(
            "Not every training row matched an activity feature row."
        )

    if POPULATION_PATH.exists():
        population = pd.read_csv(
            POPULATION_PATH,
            dtype={
                "grid_id": str,
            },
        )

        required_population = {
            "grid_id",
            *OPTIONAL_POPULATION_COLUMNS,
        }

        missing_population = (
            required_population
            - set(
                population.columns
            )
        )

        if missing_population:
            raise ValueError(
                "Population comparison columns are missing: "
                f"{sorted(missing_population)}"
            )

        if population[
            "grid_id"
        ].duplicated().any():
            raise ValueError(
                "Duplicate population grid IDs were found."
            )

        merged = merged.merge(
            population[
                [
                    "grid_id",
                    *OPTIONAL_POPULATION_COLUMNS,
                ]
            ],
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

        if merged[
            list(
                OPTIONAL_POPULATION_COLUMNS
            )
        ].isna().any().any():
            raise ValueError(
                "Population comparison merge contains missing values."
            )

    return merged


def safe_standardized_mean_difference(
    positive: pd.Series,
    negative: pd.Series,
) -> float:
    """Calculate a descriptive pooled-standard-deviation mean difference."""

    positive_values = positive.to_numpy(
        dtype=float
    )

    negative_values = negative.to_numpy(
        dtype=float
    )

    positive_variance = float(
        np.var(
            positive_values,
            ddof=1,
        )
    )

    negative_variance = float(
        np.var(
            negative_values,
            ddof=1,
        )
    )

    pooled_variance = (
        positive_variance
        + negative_variance
    ) / 2.0

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance
        <= 0
    ):
        return 0.0

    return float(
        (
            positive_values.mean()
            - negative_values.mean()
        )
        / np.sqrt(
            pooled_variance
        )
    )


def create_distribution_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sparsity and upper-tail distributions for activity features."""

    records: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    for column in ACTIVITY_FEATURE_COLUMNS:
        series = dataframe[
            column
        ]

        records.append(
            {
                "feature": column,
                "zero_count": int(
                    (
                        series
                        == 0
                    ).sum()
                ),
                "nonzero_count": int(
                    (
                        series
                        > 0
                    ).sum()
                ),
                "nonzero_fraction": float(
                    (
                        series
                        > 0
                    ).mean()
                ),
                "mean": float(
                    series.mean()
                ),
                "p50": float(
                    series.quantile(
                        0.50
                    )
                ),
                "p90": float(
                    series.quantile(
                        0.90
                    )
                ),
                "p95": float(
                    series.quantile(
                        0.95
                    )
                ),
                "p99": float(
                    series.quantile(
                        0.99
                    )
                ),
                "maximum": float(
                    series.max()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def create_target_comparison(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Compare activity distributions in known station and non-station cells."""

    positive_rows = dataframe.loc[
        dataframe[
            TARGET_COLUMN
        ]
        == 1
    ]

    negative_rows = dataframe.loc[
        dataframe[
            TARGET_COLUMN
        ]
        == 0
    ]

    records: list[
        dict[
            str,
            float,
        ]
    ] = []

    for column in ACTIVITY_FEATURE_COLUMNS:
        positive = positive_rows[
            column
        ]

        negative = negative_rows[
            column
        ]

        records.append(
            {
                "feature": column,
                "positive_mean": float(
                    positive.mean()
                ),
                "negative_mean": float(
                    negative.mean()
                ),
                "positive_median": float(
                    positive.median()
                ),
                "negative_median": float(
                    negative.median()
                ),
                "positive_nonzero_fraction": float(
                    (
                        positive
                        > 0
                    ).mean()
                ),
                "negative_nonzero_fraction": float(
                    (
                        negative
                        > 0
                    ).mean()
                ),
                "standardized_mean_difference": (
                    safe_standardized_mean_difference(
                        positive,
                        negative,
                    )
                ),
            }
        )

    result = pd.DataFrame(
        records
    )

    return result.sort_values(
        "standardized_mean_difference",
        ascending=False,
        kind="stable",
    ).reset_index(
        drop=True
    )


def create_correlation_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create Spearman correlations among activity and optional population features."""

    columns = list(
        ACTIVITY_FEATURE_COLUMNS
    )

    columns.extend(
        column
        for column in OPTIONAL_POPULATION_COLUMNS
        if column
        in dataframe.columns
    )

    return dataframe[
        columns
    ].corr(
        method="spearman"
    )


def create_redundancy_pairs(
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    """List highly correlated activity-feature pairs once each."""

    activity_columns = list(
        ACTIVITY_FEATURE_COLUMNS
    )

    records: list[
        dict[
            str,
            float,
        ]
    ] = []

    for left_index, left_column in enumerate(
        activity_columns
    ):
        for right_column in activity_columns[
            left_index
            + 1:
        ]:
            value = float(
                correlation.loc[
                    left_column,
                    right_column,
                ]
            )

            if (
                np.isfinite(
                    value
                )
                and abs(
                    value
                )
                >= REDUNDANCY_THRESHOLD
            ):
                records.append(
                    {
                        "feature_a": left_column,
                        "feature_b": right_column,
                        "spearman_correlation": value,
                        "absolute_spearman_correlation": abs(
                            value
                        ),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "spearman_correlation",
                "absolute_spearman_correlation",
            ]
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "absolute_spearman_correlation",
            ascending=False,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def create_correlation_plot(
    correlation: pd.DataFrame,
) -> None:
    """Plot the activity-feature Spearman correlation matrix."""

    CORRELATION_PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    activity_correlation = correlation.loc[
        list(
            ACTIVITY_FEATURE_COLUMNS
        ),
        list(
            ACTIVITY_FEATURE_COLUMNS
        ),
    ]

    figure, axis = plt.subplots(
        figsize=(
            12,
            10,
        )
    )

    image = axis.imshow(
        activity_correlation.to_numpy(
            dtype=float
        ),
        vmin=-1,
        vmax=1,
    )

    axis.set_xticks(
        np.arange(
            len(
                ACTIVITY_FEATURE_COLUMNS
            )
        )
    )

    axis.set_yticks(
        np.arange(
            len(
                ACTIVITY_FEATURE_COLUMNS
            )
        )
    )

    axis.set_xticklabels(
        ACTIVITY_FEATURE_COLUMNS,
        rotation=70,
        ha="right",
    )

    axis.set_yticklabels(
        ACTIVITY_FEATURE_COLUMNS
    )

    axis.set_title(
        "Ankara Activity Features - Spearman Correlation"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Spearman correlation",
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


def save_outputs(
    distribution: pd.DataFrame,
    target_comparison: pd.DataFrame,
    correlation: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> None:
    """Save activity audit tables."""

    DISTRIBUTION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution.to_csv(
        DISTRIBUTION_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    target_comparison.to_csv(
        TARGET_COMPARISON_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    correlation.to_csv(
        CORRELATION_OUTPUT_PATH,
        index=True,
        encoding="utf-8",
    )

    redundancy.to_csv(
        REDUNDANCY_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
    distribution: pd.DataFrame,
    target_comparison: pd.DataFrame,
    correlation: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> None:
    """Write the human-readable Ankara activity audit summary."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_lines = [
        "| Feature | Nonzero cells | Nonzero % | Median | P90 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in distribution.itertuples(
        index=False
    ):
        distribution_lines.append(
            "| "
            f"{row.feature} | "
            f"{row.nonzero_count:,} | "
            f"{row.nonzero_fraction:.2%} | "
            f"{row.p50:.2f} | "
            f"{row.p90:.2f} | "
            f"{row.p95:.2f} | "
            f"{row.p99:.2f} | "
            f"{row.maximum:.0f} |"
        )

    target_lines = [
        "| Feature | Positive median | Negative median | Positive nonzero | Negative nonzero | SMD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in target_comparison.itertuples(
        index=False
    ):
        target_lines.append(
            "| "
            f"{row.feature} | "
            f"{row.positive_median:.2f} | "
            f"{row.negative_median:.2f} | "
            f"{row.positive_nonzero_fraction:.2%} | "
            f"{row.negative_nonzero_fraction:.2%} | "
            f"{row.standardized_mean_difference:+.4f} |"
        )

    if redundancy.empty:
        redundancy_lines = [
            "- No activity-feature pair reached "
            f"|Spearman| >= {REDUNDANCY_THRESHOLD:.2f}."
        ]
    else:
        redundancy_lines = [
            (
                f"- `{row.feature_a}` ↔ `{row.feature_b}`: "
                f"{row.spearman_correlation:+.4f}"
            )
            for row in redundancy.itertuples(
                index=False
            )
        ]

    population_lines: list[
        str
    ] = []

    available_population = [
        column
        for column in OPTIONAL_POPULATION_COLUMNS
        if column
        in dataframe.columns
    ]

    if available_population:
        population_lines.append(
            "Spearman correlation with the optional WorldPop context:"
        )

        for activity_column in ACTIVITY_FEATURE_COLUMNS:
            correlations = [
                (
                    population_column,
                    float(
                        correlation.loc[
                            activity_column,
                            population_column,
                        ]
                    ),
                )
                for population_column in available_population
            ]

            strongest_column, strongest_value = max(
                correlations,
                key=lambda item: abs(
                    item[
                        1
                    ]
                ),
            )

            population_lines.append(
                f"- `{activity_column}` strongest population association: "
                f"`{strongest_column}` {strongest_value:+.4f}"
            )

    summary = f"""# Ankara Activity Feature Audit

## Purpose

This audit evaluates coverage, sparsity, redundancy, and descriptive association
with known charging-station cells before OSM activity features are used in any
machine-learning or suitability model.

## Dataset

- Grid/training rows: {len(dataframe):,}
- Positive existing-station cells: {int(dataframe[TARGET_COLUMN].sum()):,}
- Negative cells: {int((dataframe[TARGET_COLUMN] == 0).sum()):,}
- Activity features: {len(ACTIVITY_FEATURE_COLUMNS)}

## Coverage and Distribution

{chr(10).join(distribution_lines)}

Sparse zero-heavy local counts are expected because OSM activity mapping is
concentrated in settlements and because the study area covers the full Ankara
province.

## Existing-Station Descriptive Comparison

{chr(10).join(target_lines)}

SMD is a descriptive standardized mean difference between known station cells
and non-station cells. It is not a causal effect and should not be interpreted
as feature importance.

Only {int(dataframe[TARGET_COLUMN].sum()):,} positive cells are available, so
positive-group estimates are inherently noisy.

## Activity-Feature Redundancy

Pairs at or above an absolute Spearman correlation of
{REDUNDANCY_THRESHOLD:.2f}:

{chr(10).join(redundancy_lines)}

Nested local / 1-km / 2-km counts can be strongly correlated without being
mathematically identical. High correlation is a reason to prefer a parsimonious
feature set in downstream ML experiments.

## Population Overlap

{chr(10).join(population_lines) if population_lines else "WorldPop comparison was not available."}

Population correlation is important because both feature families may encode
urbanization. A strong relationship does not make either feature invalid, but
it reduces the case for treating them as independent evidence.

## Interpretation Policy

OSM POI coverage is spatially heterogeneous. Zero or low counts may reflect
either low mapped urban activity or incomplete OSM mapping.

Activity counts are therefore treated as mapped urban-activity proxies rather
than direct observations of EV demand, trips, employment, retail turnover, or
traffic.

This audit is descriptive. The next evidence step is incremental evaluation
under the existing 5-km spatial block cross-validation design, with average
precision as the primary metric because the target is extremely imbalanced.

## Outputs

- `data/processed/{DISTRIBUTION_OUTPUT_PATH.name}`
- `data/processed/{TARGET_COMPARISON_OUTPUT_PATH.name}`
- `data/processed/{CORRELATION_OUTPUT_PATH.name}`
- `data/processed/{REDUNDANCY_OUTPUT_PATH.name}`
- `docs/{CORRELATION_PLOT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    distribution: pd.DataFrame,
    target_comparison: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> None:
    """Print the main activity-audit diagnostics."""

    print(
        "-"
        * 70
    )

    print(
        "Activity coverage:"
    )

    print(
        distribution[
            [
                "feature",
                "nonzero_count",
                "nonzero_fraction",
                "p50",
                "p95",
                "p99",
                "maximum",
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
        "Target comparison:"
    )

    print(
        target_comparison[
            [
                "feature",
                "positive_median",
                "negative_median",
                "positive_nonzero_fraction",
                "negative_nonzero_fraction",
                "standardized_mean_difference",
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
        "High-correlation activity pairs:",
        len(
            redundancy
        ),
    )


def main() -> None:
    """Run the Ankara OSM activity feature audit."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Activity Feature Audit"
    )

    print(
        "="
        * 70
    )

    dataframe = load_inputs()

    distribution = (
        create_distribution_table(
            dataframe
        )
    )

    target_comparison = (
        create_target_comparison(
            dataframe
        )
    )

    correlation = (
        create_correlation_table(
            dataframe
        )
    )

    redundancy = (
        create_redundancy_pairs(
            correlation
        )
    )

    save_outputs(
        distribution,
        target_comparison,
        correlation,
        redundancy,
    )

    create_correlation_plot(
        correlation
    )

    create_summary(
        dataframe,
        distribution,
        target_comparison,
        correlation,
        redundancy,
    )

    print_results(
        distribution,
        target_comparison,
        redundancy,
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara activity feature audit completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
