from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voltsight.core.ankara_ml_features import (
    CHARGING_CONTEXT_COLUMNS,
    CHARGING_LEAKAGE_COLUMNS,
    HISTORICAL_FULL_14_FEATURE_COLUMNS,
    PARKING_FEATURE_COLUMNS,
    ROAD_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_charging_features.csv"
)

TRAINING_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_site_dataset.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_model_dataset_summary.md"
)


IDENTIFIER_COLUMNS = (
    "grid_id",
)

CONSTANT_METADATA_COLUMNS = (
    "cell_area_m2",
)

TRAINING_FEATURE_COLUMNS = (
    HISTORICAL_FULL_14_FEATURE_COLUMNS
)

TRAINING_OUTPUT_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *TRAINING_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

CANDIDATE_OUTPUT_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *ROAD_FEATURE_COLUMNS,
    *PARKING_FEATURE_COLUMNS,
    *CHARGING_CONTEXT_COLUMNS,
)


def create_output_directories() -> None:
    """Create output directories."""

    TRAINING_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_input_file() -> None:
    """Ensure the complete Ankara feature CSV exists."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Ankara charging feature CSV was not found:\n"
            f"{INPUT_PATH}"
        )


def required_input_columns() -> set[str]:
    """Return columns required to build model datasets."""

    return {
        *IDENTIFIER_COLUMNS,
        *CONSTANT_METADATA_COLUMNS,
        *ROAD_FEATURE_COLUMNS,
        *PARKING_FEATURE_COLUMNS,
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
        TARGET_COLUMN,
    }


def load_source_dataset() -> pd.DataFrame:
    """Load and validate the complete Ankara feature dataset."""

    source = pd.read_csv(
        INPUT_PATH,
        dtype={
            "grid_id": str,
        },
    )

    if source.empty:
        raise ValueError(
            "The Ankara feature dataset is empty."
        )

    missing_columns = (
        required_input_columns()
        - set(source.columns)
    )

    if missing_columns:
        raise ValueError(
            "The Ankara source dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if source["grid_id"].isna().any():
        raise ValueError(
            "Missing grid IDs were found."
        )

    source["grid_id"] = (
        source["grid_id"]
        .astype(str)
        .str.strip()
    )

    if source["grid_id"].eq("").any():
        raise ValueError(
            "Empty grid IDs were found."
        )

    if source["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs were found."
        )

    print(
        "Loaded source rows: "
        f"{len(source):,}"
    )

    print(
        "Loaded source columns: "
        f"{len(source.columns):,}"
    )

    return source


def numeric_feature_columns() -> tuple[str, ...]:
    """Return all numeric columns required by the pipeline."""

    return (
        *CONSTANT_METADATA_COLUMNS,
        *ROAD_FEATURE_COLUMNS,
        *PARKING_FEATURE_COLUMNS,
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
        TARGET_COLUMN,
    )


def convert_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert required model columns to finite numeric values."""

    result = dataframe.copy()

    for column in numeric_feature_columns():
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    feature_columns_without_target = [
        column
        for column in numeric_feature_columns()
        if column != TARGET_COLUMN
    ]

    missing_counts = (
        result[
            feature_columns_without_target
        ]
        .isna()
        .sum()
    )

    missing_counts = missing_counts.loc[
        missing_counts > 0
    ]

    if not missing_counts.empty:
        raise ValueError(
            "Numeric source columns contain missing values: "
            f"{missing_counts.to_dict()}"
        )

    for column in feature_columns_without_target:
        values = result[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

        if (values < 0).any():
            raise ValueError(
                f"Negative values found in {column}."
            )

    return result


def normalize_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize existing-station presence to binary 0/1."""

    result = dataframe.copy()

    target = pd.to_numeric(
        result[TARGET_COLUMN],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "Target contains missing or non-numeric values."
        )

    unique_values = set(
        target.astype(int).unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "Target must contain only 0 and 1. "
            f"Found: {sorted(unique_values)}"
        )

    if not np.allclose(
        target.to_numpy(dtype=float),
        target.astype(int).to_numpy(
            dtype=float
        ),
    ):
        raise ValueError(
            "Target contains non-integer values."
        )

    result[TARGET_COLUMN] = (
        target.astype(int)
    )

    return result


def prepare_training_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-safe existing-station training data.

    Charging-derived variables are intentionally excluded because
    they directly describe the current charging-station distribution.
    """

    training = source.loc[
        :,
        TRAINING_OUTPUT_COLUMNS,
    ].copy()

    forbidden_columns = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
    }

    leaked_columns = (
        forbidden_columns
        & set(training.columns)
    )

    if leaked_columns:
        raise RuntimeError(
            "Charging leakage columns entered training data: "
            f"{sorted(leaked_columns)}"
        )

    if training.isna().any().any():
        raise ValueError(
            "Training dataset contains missing values."
        )

    return training


def prepare_candidate_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Create candidate data from cells without an existing station."""

    candidates = source.loc[
        source[TARGET_COLUMN] == 0,
        CANDIDATE_OUTPUT_COLUMNS,
    ].copy()

    candidates.reset_index(
        drop=True,
        inplace=True,
    )

    if candidates.empty:
        raise ValueError(
            "The Ankara candidate dataset is empty."
        )

    if candidates.isna().any().any():
        missing = (
            candidates.isna().sum()
        )

        missing = missing.loc[
            missing > 0
        ]

        raise ValueError(
            "Candidate dataset contains missing values: "
            f"{missing.to_dict()}"
        )

    return candidates


def validate_output_relationships(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Validate target, candidate and leakage relationships."""

    if len(training) != len(source):
        raise ValueError(
            "Training dataset must preserve every grid cell."
        )

    positive_count = int(
        source[TARGET_COLUMN].sum()
    )

    expected_candidate_count = (
        len(source)
        - positive_count
    )

    if len(candidates) != expected_candidate_count:
        raise ValueError(
            "Candidate row count is incorrect. "
            f"Expected {expected_candidate_count:,}, "
            f"found {len(candidates):,}."
        )

    source_ids = set(
        source["grid_id"]
    )

    training_ids = set(
        training["grid_id"]
    )

    candidate_ids = set(
        candidates["grid_id"]
    )

    if source_ids != training_ids:
        raise ValueError(
            "Training grid IDs do not match source grid IDs."
        )

    positive_ids = set(
        source.loc[
            source[TARGET_COLUMN] == 1,
            "grid_id",
        ]
    )

    if positive_ids & candidate_ids:
        raise ValueError(
            "Existing-station cells leaked into candidate data."
        )

    if (
        candidate_ids | positive_ids
    ) != source_ids:
        raise ValueError(
            "Candidate and positive grid IDs do not partition "
            "the source dataset."
        )

    if training["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate training grid IDs were found."
        )

    if candidates["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate grid IDs were found."
        )

    forbidden_training_columns = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
    }

    if (
        forbidden_training_columns
        & set(training.columns)
    ):
        raise ValueError(
            "Charging-derived leakage remains "
            "in training output."
        )

    print(
        "Model dataset validation completed successfully."
    )


def save_outputs(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Save training and candidate CSV outputs."""

    for output_path in (
        TRAINING_OUTPUT_PATH,
        CANDIDATE_OUTPUT_PATH,
    ):
        if output_path.exists():
            output_path.unlink()

    training.to_csv(
        TRAINING_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Training dataset saved: {TRAINING_OUTPUT_PATH}"
    )

    print(
        f"Candidate dataset saved: {CANDIDATE_OUTPUT_PATH}"
    )


def create_summary(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Create a reproducibility summary."""

    positive_count = int(
        source[TARGET_COLUMN].sum()
    )

    negative_count = (
        len(source)
        - positive_count
    )

    positive_rate = (
        positive_count
        / len(source)
        * 100
    )

    training_feature_lines = "\n".join(
        f"- `{column}`"
        for column in TRAINING_FEATURE_COLUMNS
    )

    context_lines = "\n".join(
        f"- `{column}`"
        for column in CHARGING_CONTEXT_COLUMNS
    )

    leakage_lines = "\n".join(
        f"- `{column}`"
        for column in CHARGING_LEAKAGE_COLUMNS
    )

    summary = f"""# Ankara Model Dataset Summary

## Source Dataset

- Source rows: {len(source):,}
- Source columns: {len(source.columns):,}
- Existing-station grid cells: {positive_count:,}
- Grid cells without an existing station: {negative_count:,}
- Existing-station prevalence: {positive_rate:.4f}%
- Missing values in required model columns: 0

## Leakage-Safe Training Dataset

- Training rows: {len(training):,}
- Training columns: {len(training.columns):,}
- Positive target rows: {positive_count:,}
- Negative target rows: {negative_count:,}
- Target: `{TARGET_COLUMN}`
- Charging-derived predictor columns included: 0

### Predictor Columns

{training_feature_lines}

## Candidate Dataset

- Candidate rows: {len(candidates):,}
- Candidate columns: {len(candidates.columns):,}
- Existing-station cells excluded: {positive_count:,}

### Charging Context Retained for Suitability Analysis

{context_lines}

These variables describe the current infrastructure gap and are
retained for candidate suitability scoring. They are intentionally
excluded from the existing-station training predictors.

## Direct Charging Leakage Columns

{leakage_lines}

These columns directly describe charging infrastructure inside the
grid cell and are excluded from both model predictors and candidate
scoring inputs.

## Leakage Policy

The existing-station classification dataset uses only road and parking
variables as predictors.

Current charging-station distance and neighborhood counts are not used
as predictors because they are functions of the same existing station
distribution represented by the target.

Candidate-site suitability is a separate decision-support task.
Charging context is allowed there because infrastructure scarcity is
an explicit component of site need rather than a predictor used to
reproduce the existing-station target.

## Outputs

- `data/processed/ankara_existing_station_training_dataset.csv`
- `data/processed/ankara_candidate_site_dataset.csv`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Model dataset summary saved: {SUMMARY_OUTPUT_PATH}"
    )


def print_statistics(
    source: pd.DataFrame,
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Print key model-dataset statistics."""

    positive_count = int(
        source[TARGET_COLUMN].sum()
    )

    print("-" * 70)

    print(
        "Source rows: "
        f"{len(source):,}"
    )

    print(
        "Source columns: "
        f"{len(source.columns):,}"
    )

    print(
        "Training rows: "
        f"{len(training):,}"
    )

    print(
        "Training feature count: "
        f"{len(TRAINING_FEATURE_COLUMNS):,}"
    )

    print(
        "Positive target rows: "
        f"{positive_count:,}"
    )

    print(
        "Negative target rows: "
        f"{len(source) - positive_count:,}"
    )

    print(
        "Candidate rows: "
        f"{len(candidates):,}"
    )

    print(
        "Charging leakage predictors in training: 0"
    )


def main() -> None:
    """Create Ankara training and candidate datasets."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Model Dataset Pipeline"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_file()

    source = load_source_dataset()

    source = convert_numeric_columns(
        source
    )

    source = normalize_target(
        source
    )

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
        "Ankara model dataset pipeline "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
