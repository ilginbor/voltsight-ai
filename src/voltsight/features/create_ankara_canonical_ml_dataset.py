from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from voltsight.core.ankara_ml_features import (
    ACTIVITY_CONTEXT_FEATURE_COLUMNS,
    CANONICAL_ML_FEATURE_COLUMNS,
    CHARGING_CONTEXT_COLUMNS,
    CHARGING_LEAKAGE_COLUMNS,
    HISTORICAL_FULL_14_FEATURE_COLUMNS,
    NORMALIZED_12_FEATURE_COLUMNS,
    REDUNDANT_SCALE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_architecture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

HISTORICAL_TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

ACTIVITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_activity_features.csv"
)

TRAINING_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_training_dataset.csv"
)

CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_canonical_ml_candidate_dataset.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_canonical_ml_dataset_summary.md"
)


TRAINING_OUTPUT_COLUMNS = (
    "grid_id",
    *CANONICAL_ML_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

CANDIDATE_OUTPUT_COLUMNS = (
    "grid_id",
    *CANONICAL_ML_FEATURE_COLUMNS,
)


def validate_numeric_columns(
    dataframe: pd.DataFrame,
    columns: tuple[
        str,
        ...,
    ],
    *,
    label: str,
) -> pd.DataFrame:
    """Convert required columns to finite nonnegative numeric values."""

    result = dataframe.copy()

    for column in columns:
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
                f"{label} contains invalid values in {column}."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"{label} contains negative values in {column}."
            )

    return result


def validate_historical_training(
    training: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the existing historical road-and-parking training table."""

    required = {
        "grid_id",
        *HISTORICAL_FULL_14_FEATURE_COLUMNS,
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(
            training.columns
        )
    )

    if missing:
        raise ValueError(
            "Historical training columns are missing: "
            f"{sorted(missing)}"
        )

    result = training[
        [
            "grid_id",
            *HISTORICAL_FULL_14_FEATURE_COLUMNS,
            TARGET_COLUMN,
        ]
    ].copy()

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        "grid_id"
    ].eq("").any():
        raise ValueError(
            "Historical training contains empty grid IDs."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Historical training contains duplicate grid IDs."
        )

    result = validate_numeric_columns(
        result,
        HISTORICAL_FULL_14_FEATURE_COLUMNS,
        label="Historical training",
    )

    target = pd.to_numeric(
        result[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "Historical training target contains invalid values."
        )

    if not np.allclose(
        target.to_numpy(
            dtype=float
        ),
        target.astype(
            int
        ).to_numpy(
            dtype=float
        ),
    ):
        raise ValueError(
            "Historical training target contains non-integer values."
        )

    result[
        TARGET_COLUMN
    ] = target.astype(
        int
    )

    if set(
        result[
            TARGET_COLUMN
        ].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Historical training target must contain both classes."
        )

    return result


def validate_activity_features(
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the three total-activity context columns."""

    required = {
        "grid_id",
        *ACTIVITY_CONTEXT_FEATURE_COLUMNS,
    }

    missing = (
        required
        - set(
            activity.columns
        )
    )

    if missing:
        raise ValueError(
            "Activity columns are missing: "
            f"{sorted(missing)}"
        )

    result = activity[
        [
            "grid_id",
            *ACTIVITY_CONTEXT_FEATURE_COLUMNS,
        ]
    ].copy()

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        "grid_id"
    ].eq("").any():
        raise ValueError(
            "Activity features contain empty grid IDs."
        )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Activity features contain duplicate grid IDs."
        )

    result = validate_numeric_columns(
        result,
        ACTIVITY_CONTEXT_FEATURE_COLUMNS,
        label="Activity features",
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
            "1-km POI count cannot be below local POI count."
        )

    if (
        within_2km
        < within_1km
    ).any():
        raise ValueError(
            "2-km POI count cannot be below 1-km POI count."
        )

    return result


def build_canonical_datasets(
    historical_training: pd.DataFrame,
    activity: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the canonical 15-feature training and candidate datasets."""

    validate_feature_architecture()

    historical_training = (
        validate_historical_training(
            historical_training
        )
    )

    activity = (
        validate_activity_features(
            activity
        )
    )

    merged = historical_training.merge(
        activity,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != len(
        historical_training
    ):
        raise ValueError(
            "Activity merge changed the historical training row count."
        )

    if merged[
        list(
            ACTIVITY_CONTEXT_FEATURE_COLUMNS
        )
    ].isna().any().any():
        missing_count = int(
            merged[
                list(
                    ACTIVITY_CONTEXT_FEATURE_COLUMNS
                )
            ].isna().any(
                axis=1
            ).sum()
        )

        raise ValueError(
            "Not every historical training row matched activity features. "
            f"Missing rows: {missing_count}."
        )

    training = merged[
        list(
            TRAINING_OUTPUT_COLUMNS
        )
    ].copy()

    candidates = training.loc[
        training[
            TARGET_COLUMN
        ]
        == 0,
        list(
            CANDIDATE_OUTPUT_COLUMNS
        ),
    ].copy()

    candidates.reset_index(
        drop=True,
        inplace=True,
    )

    validate_output_relationships(
        training,
        candidates,
    )

    return (
        training,
        candidates,
    )


def validate_output_relationships(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Validate canonical schema, leakage policy, and row relationships."""

    if tuple(
        training.columns
    ) != TRAINING_OUTPUT_COLUMNS:
        raise ValueError(
            "Canonical training columns are not in the expected order."
        )

    if tuple(
        candidates.columns
    ) != CANDIDATE_OUTPUT_COLUMNS:
        raise ValueError(
            "Canonical candidate columns are not in the expected order."
        )

    if training[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical training contains duplicate grid IDs."
        )

    if candidates[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Canonical candidate data contains duplicate grid IDs."
        )

    expected_candidates = int(
        (
            training[
                TARGET_COLUMN
            ]
            == 0
        ).sum()
    )

    if len(
        candidates
    ) != expected_candidates:
        raise ValueError(
            "Canonical candidate row count does not match target negatives."
        )

    positive_ids = set(
        training.loc[
            training[
                TARGET_COLUMN
            ]
            == 1,
            "grid_id",
        ]
    )

    candidate_ids = set(
        candidates[
            "grid_id"
        ]
    )

    if positive_ids & candidate_ids:
        raise ValueError(
            "Positive station cells leaked into canonical candidate data."
        )

    forbidden = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
        *REDUNDANT_SCALE_FEATURE_COLUMNS,
    }

    leaked = forbidden & set(
        training.columns
    )

    if leaked:
        raise ValueError(
            "Excluded columns entered canonical training data: "
            f"{sorted(leaked)}"
        )

    for column in CANONICAL_ML_FEATURE_COLUMNS:
        values = training[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Canonical feature {column} contains non-finite values."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"Canonical feature {column} contains negative values."
            )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load source tables required for the canonical ML dataset."""

    if not HISTORICAL_TRAINING_PATH.exists():
        raise FileNotFoundError(
            "Historical Ankara training dataset was not found:\n"
            f"{HISTORICAL_TRAINING_PATH}"
        )

    if not ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            "Ankara activity feature dataset was not found:\n"
            f"{ACTIVITY_PATH}"
        )

    historical_training = pd.read_csv(
        HISTORICAL_TRAINING_PATH,
        dtype={
            "grid_id": str,
        },
    )

    activity = pd.read_csv(
        ACTIVITY_PATH,
        dtype={
            "grid_id": str,
        },
    )

    return (
        historical_training,
        activity,
    )


def save_outputs(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Save canonical ML training and candidate datasets."""

    TRAINING_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def create_summary(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Document the canonical Ankara ML feature architecture."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_lines = "\n".join(
        f"- `{feature}`"
        for feature in CANONICAL_ML_FEATURE_COLUMNS
    )

    removed_lines = "\n".join(
        f"- `{feature}`"
        for feature in REDUNDANT_SCALE_FEATURE_COLUMNS
    )

    activity_lines = "\n".join(
        f"- `{feature}`"
        for feature in ACTIVITY_CONTEXT_FEATURE_COLUMNS
    )

    summary = f"""# Ankara Canonical ML Dataset

## Purpose

This dataset defines the forward-looking canonical predictor architecture for
Ankara ML experiments after the road/parking redundancy and OSM activity
feature-family evaluations.

Historical baseline datasets and scripts remain valid historical references.

## Dataset

- Training rows: {len(training):,}
- Candidate rows: {len(candidates):,}
- Positive existing-station cells: {int(training[TARGET_COLUMN].sum()):,}
- Predictors: {len(CANONICAL_ML_FEATURE_COLUMNS)}
- Target: `{TARGET_COLUMN}`

## Canonical Feature Set

The canonical set contains the deduplicated normalized-12 road/parking
predictors plus three target-agnostic total OSM activity-context features.

{feature_lines}

## Removed Near-Deterministic Scale Duplicates

{removed_lines}

Their normalized counterparts remain in the canonical road/parking set.

## Added Activity Context

{activity_lines}

Category-specific OSM activity variables are not included in the canonical
feature set because the category-context sensitivity experiment did not show
the same model-general pooled-AP improvement as the total activity context.

Population features are not included in the canonical ML predictor set.

## Leakage Policy

Charging-derived context and direct charging-count variables are excluded from
the canonical ML predictors.

The target describes the existing charging-station distribution, so current
charging infrastructure cannot be used as a predictor in this classification
task.

Candidate-site suitability remains a separate decision-support layer where
charging scarcity is allowed as an explicit need component.

## Historical Compatibility

The existing `ankara_existing_station_training_dataset.csv` remains the
historical full-14 road/parking dataset.

This pipeline creates new canonical outputs instead of overwriting that
historical dataset.

## Outputs

- `data/processed/{TRAINING_OUTPUT_PATH.name}`
- `data/processed/{CANDIDATE_OUTPUT_PATH.name}`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_results(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Print canonical ML dataset statistics."""

    print(
        "-"
        * 70
    )

    print(
        "Training rows:",
        f"{len(training):,}",
    )

    print(
        "Positive rows:",
        f"{int(training[TARGET_COLUMN].sum()):,}",
    )

    print(
        "Candidate rows:",
        f"{len(candidates):,}",
    )

    print(
        "Canonical predictors:",
        len(
            CANONICAL_ML_FEATURE_COLUMNS
        ),
    )

    print(
        "Normalized road/parking predictors:",
        len(
            NORMALIZED_12_FEATURE_COLUMNS
        ),
    )

    print(
        "Activity context predictors:",
        len(
            ACTIVITY_CONTEXT_FEATURE_COLUMNS
        ),
    )


def main() -> None:
    """Create the Ankara canonical 15-feature ML datasets."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Canonical ML Dataset"
    )

    print(
        "="
        * 70
    )

    (
        historical_training,
        activity,
    ) = load_inputs()

    (
        training,
        candidates,
    ) = build_canonical_datasets(
        historical_training,
        activity,
    )

    save_outputs(
        training,
        candidates,
    )

    create_summary(
        training,
        candidates,
    )

    print_results(
        training,
        candidates,
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara canonical ML dataset completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
