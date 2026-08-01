from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.csv"
)

TRAINING_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_existing_station_training_dataset.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_site_dataset.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_model_dataset_summary.md"
)

TARGET_COLUMN = "has_existing_charging_station"

IDENTIFIER_COLUMNS = [
    "grid_id",
    "district",
    "city",
    "center_longitude",
    "center_latitude",
]

CONSTANT_METADATA_COLUMNS = [
    "grid_size_m",
    "cell_area_m2",
]

ROAD_FEATURE_COLUMNS = [
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
]

PARKING_FEATURE_COLUMNS = [
    "parking_count",
    "known_parking_capacity",
    "parking_capacity_record_count",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "parking_area_ratio",
]

CHARGING_CONTEXT_COLUMNS = [
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
]

CHARGING_LEAKAGE_COLUMNS = [
    "charging_station_count",
    "known_charging_capacity",
    "charging_capacity_record_count",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
]

TRAINING_FEATURE_COLUMNS = (
    ROAD_FEATURE_COLUMNS
    + PARKING_FEATURE_COLUMNS
)

TRAINING_OUTPUT_COLUMNS = (
    IDENTIFIER_COLUMNS
    + CONSTANT_METADATA_COLUMNS
    + TRAINING_FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

CANDIDATE_OUTPUT_COLUMNS = (
    IDENTIFIER_COLUMNS
    + CONSTANT_METADATA_COLUMNS
    + ROAD_FEATURE_COLUMNS
    + PARKING_FEATURE_COLUMNS
    + CHARGING_CONTEXT_COLUMNS
)


def create_output_directories() -> None:
    """Create directories required by the model-dataset pipeline."""

    directories = {
        TRAINING_OUTPUT_PATH.parent,
        CANDIDATE_OUTPUT_PATH.parent,
        SUMMARY_OUTPUT_PATH.parent,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_input_file() -> None:
    """Ensure that the final grid-feature CSV exists."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "The charging-feature CSV was not found:\n"
            f"{INPUT_PATH}"
        )

    if INPUT_PATH.stat().st_size == 0:
        raise ValueError(
            "The charging-feature CSV is empty."
        )


def required_input_columns() -> set[str]:
    """Return every source column required by this pipeline."""

    return set(
        IDENTIFIER_COLUMNS
        + CONSTANT_METADATA_COLUMNS
        + ROAD_FEATURE_COLUMNS
        + PARKING_FEATURE_COLUMNS
        + CHARGING_CONTEXT_COLUMNS
        + CHARGING_LEAKAGE_COLUMNS
        + [TARGET_COLUMN]
    )


def load_source_dataset() -> pd.DataFrame:
    """Load and validate the final grid-level feature dataset."""

    dataframe = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError(
            "The source model dataset contains no rows."
        )

    missing_columns = (
        required_input_columns()
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The source dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe["grid_id"].isna().any():
        raise ValueError(
            "The source dataset contains missing grid IDs."
        )

    if dataframe["grid_id"].duplicated().any():
        duplicate_ids = (
            dataframe.loc[
                dataframe["grid_id"].duplicated(
                    keep=False
                ),
                "grid_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate grid IDs were found: "
            f"{duplicate_ids[:10]}"
        )

    if dataframe.isna().any().any():
        missing_counts = (
            dataframe.isna()
            .sum()
        )

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            "Missing values were found:\n"
            f"{missing_counts.to_string()}"
        )

    return dataframe


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Convert selected columns to finite numeric values."""

    result = dataframe.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if result[column].isna().any():
            raise ValueError(
                f"Column {column!r} contains "
                "non-numeric values."
            )

        values = result[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column {column!r} contains "
                "non-finite values."
            )

    return result


def normalize_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the existing-station target to zero and one."""

    result = dataframe.copy()

    result[TARGET_COLUMN] = pd.to_numeric(
        result[TARGET_COLUMN],
        errors="coerce",
    )

    if result[TARGET_COLUMN].isna().any():
        raise ValueError(
            "The target column contains invalid values."
        )

    unique_values = set(
        result[TARGET_COLUMN]
        .astype(int)
        .unique()
        .tolist()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "The target must contain only zero and one. "
            f"Found: {sorted(unique_values)}"
        )

    result[TARGET_COLUMN] = (
        result[TARGET_COLUMN]
        .astype(int)
    )

    return result


def prepare_training_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Create a leakage-safe supervised-learning dataset."""

    training = source[
        TRAINING_OUTPUT_COLUMNS
    ].copy()

    numeric_columns = [
        column
        for column in TRAINING_OUTPUT_COLUMNS
        if column not in {
            "grid_id",
            "district",
            "city",
            "nearest_main_road_type",
        }
    ]

    training = convert_numeric_columns(
        training,
        numeric_columns,
    )

    training = normalize_target(
        training
    )

    positive_count = int(
        training[TARGET_COLUMN].sum()
    )

    negative_count = (
        len(training)
        - positive_count
    )

    if positive_count == 0:
        raise ValueError(
            "The training dataset has no positive labels."
        )

    if negative_count == 0:
        raise ValueError(
            "The training dataset has no negative labels."
        )

    return training.sort_values(
        by="grid_id"
    ).reset_index(
        drop=True
    )


def prepare_candidate_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Create candidate cells that do not contain a station."""

    normalized = normalize_target(
        source
    )

    candidate_rows = normalized.loc[
        normalized[TARGET_COLUMN].eq(0),
        CANDIDATE_OUTPUT_COLUMNS,
    ].copy()

    numeric_columns = [
        column
        for column in CANDIDATE_OUTPUT_COLUMNS
        if column not in {
            "grid_id",
            "district",
            "city",
            "nearest_main_road_type",
        }
    ]

    candidate_rows = convert_numeric_columns(
        candidate_rows,
        numeric_columns,
    )

    if candidate_rows.empty:
        raise ValueError(
            "No candidate grid cells were found."
        )

    if candidate_rows["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate grid IDs were found."
        )

    return candidate_rows.sort_values(
        by="grid_id"
    ).reset_index(
        drop=True
    )


def validate_output_relationships(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Validate row counts and leakage exclusions."""

    if len(training) != len(source):
        raise ValueError(
            "Training row count does not match "
            "the source row count."
        )

    positive_count = int(
        training[TARGET_COLUMN].sum()
    )

    expected_candidate_count = (
        len(source)
        - positive_count
    )

    if len(candidates) != expected_candidate_count:
        raise ValueError(
            "Candidate row count is incorrect. "
            f"Expected {expected_candidate_count}, "
            f"found {len(candidates)}."
        )

    leaked_training_columns = (
        set(CHARGING_LEAKAGE_COLUMNS)
        & set(training.columns)
    )

    if leaked_training_columns:
        raise ValueError(
            "Charging-derived leakage columns were "
            "included in the supervised dataset: "
            f"{sorted(leaked_training_columns)}"
        )

    if TARGET_COLUMN in candidates.columns:
        raise ValueError(
            "The candidate dataset must not contain "
            "the supervised target."
        )

    training_ids = set(
        training["grid_id"].astype(str)
    )

    candidate_ids = set(
        candidates["grid_id"].astype(str)
    )

    if not candidate_ids.issubset(
        training_ids
    ):
        raise ValueError(
            "Candidate IDs are not a subset of "
            "the source grid IDs."
        )


def save_outputs(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Save model-ready CSV outputs."""

    training.to_csv(
        TRAINING_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Training dataset saved:"
    )
    print(TRAINING_OUTPUT_PATH)

    print()

    print(
        "Candidate-site dataset saved:"
    )
    print(CANDIDATE_OUTPUT_PATH)


def create_summary(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Create a reproducibility and modeling summary."""

    positive_count = int(
        training[TARGET_COLUMN].sum()
    )

    negative_count = (
        len(training)
        - positive_count
    )

    positive_rate = (
        positive_count
        / len(training)
        * 100
    )

    training_feature_lines = "\n".join(
        f"- `{column}`"
        for column in TRAINING_FEATURE_COLUMNS
    )

    candidate_context_lines = "\n".join(
        f"- `{column}`"
        for column in CHARGING_CONTEXT_COLUMNS
    )

    leakage_lines = "\n".join(
        f"- `{column}`"
        for column in CHARGING_LEAKAGE_COLUMNS
    )

    summary = f"""# Çankaya Model Dataset Summary

## Source

- Source file: `{INPUT_PATH.name}`
- Generated at: {datetime.now(timezone.utc).isoformat()}
- Source grid rows: {len(source):,}
- Source columns: {len(source.columns):,}
- Missing source values: {int(source.isna().sum().sum()):,}

## Supervised Training Dataset

- Output: `data/processed/{TRAINING_OUTPUT_PATH.name}`
- Rows: {len(training):,}
- Columns: {len(training.columns):,}
- Target: `{TARGET_COLUMN}`
- Positive rows: {positive_count:,}
- Negative rows: {negative_count:,}
- Positive-class rate: {positive_rate:.4f}%

The supervised dataset excludes charging-derived columns that would
directly reveal the target. It retains road and parking characteristics
for future experiments.

### Training Features

{training_feature_lines}

## Candidate-Site Dataset

- Output: `data/processed/{CANDIDATE_OUTPUT_PATH.name}`
- Candidate rows: {len(candidates):,}
- Columns: {len(candidates.columns):,}
- Existing-station cells excluded: {positive_count:,}

The candidate dataset contains only grid cells without an existing
charging station. Charging proximity and neighborhood-count features
are retained because they can help identify underserved areas and
measure existing infrastructure coverage.

### Charging Context Features

{candidate_context_lines}

## Leakage Controls

The following charging-derived fields are not included in the
supervised existing-station training matrix:

{leakage_lines}

## Modeling Limitation

Only {positive_count:,} of {len(training):,} grid cells contain an
existing charging station. The positive-class rate is
{positive_rate:.4f}%. This is an extremely imbalanced target and is not
sufficient by itself for a reliable production classifier.

The candidate-site dataset should first be used for explainable
suitability scoring, ranking, clustering or weak-supervision
experiments. Additional verified stations, utilization data or expert
labels are required before treating the supervised target as strong
ground truth.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()

    print(
        "Model dataset summary saved:"
    )
    print(SUMMARY_OUTPUT_PATH)


def print_statistics(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Print final model-dataset statistics."""

    positive_count = int(
        training[TARGET_COLUMN].sum()
    )

    positive_rate = (
        positive_count
        / len(training)
        * 100
    )

    print("-" * 70)

    print(
        f"Source row count: {len(source):,}"
    )

    print(
        f"Training row count: {len(training):,}"
    )

    print(
        f"Candidate-site row count: {len(candidates):,}"
    )

    print(
        f"Existing-station target count: "
        f"{positive_count:,}"
    )

    print(
        f"Existing-station target rate: "
        f"{positive_rate:.4f}%"
    )

    print(
        "Training feature count: "
        f"{len(TRAINING_FEATURE_COLUMNS):,}"
    )

    print(
        "Candidate context feature count: "
        f"{len(CHARGING_CONTEXT_COLUMNS):,}"
    )


def main() -> None:
    """Create leakage-safe training and candidate-site datasets."""

    print("=" * 70)

    print(
        "VoltSight - Çankaya Model Dataset Pipeline"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_file()

    source = load_source_dataset()

    training = prepare_training_dataset(
        source
    )

    candidates = prepare_candidate_dataset(
        source
    )

    validate_output_relationships(
        source,
        training,
        candidates,
    )

    save_outputs(
        training,
        candidates,
    )

    create_summary(
        source,
        training,
        candidates,
    )

    print_statistics(
        source,
        training,
        candidates,
    )

    print("=" * 70)

    print(
        "Model dataset pipeline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
