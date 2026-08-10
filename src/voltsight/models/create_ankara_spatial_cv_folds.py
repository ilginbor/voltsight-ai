from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_existing_station_training_dataset.csv"
)

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_charging_features.gpkg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_cv_folds.csv"
)

BLOCK_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_spatial_cv_block_summary.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_spatial_cv_summary.md"
)

GRID_LAYER_NAME = "grid_charging_features"

TARGET_COLUMN = "has_existing_charging_station"

BLOCK_SIZE_METERS = 5_000
N_SPLITS = 5


def load_inputs() -> gpd.GeoDataFrame:
    """Load training labels and corresponding grid geometry."""

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_PATH}"
        )

    if not GRID_PATH.exists():
        raise FileNotFoundError(
            f"Spatial grid not found: {GRID_PATH}"
        )

    training = pd.read_csv(
        TRAINING_PATH,
        dtype={
            "grid_id": str,
        },
    )

    grid = gpd.read_file(
        GRID_PATH,
        layer=GRID_LAYER_NAME,
    )[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()

    if training.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if grid.empty:
        raise ValueError(
            "Spatial grid is empty."
        )

    if grid.crs is None:
        raise ValueError(
            "Grid CRS is missing."
        )

    if not grid.crs.is_projected:
        raise ValueError(
            "Spatial CV requires a projected CRS."
        )

    training["grid_id"] = (
        training["grid_id"]
        .astype(str)
    )

    grid["grid_id"] = (
        grid["grid_id"]
        .astype(str)
    )

    if training["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate training IDs were found."
        )

    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate spatial grid IDs were found."
        )

    required_training_columns = {
        "grid_id",
        TARGET_COLUMN,
    }

    missing_training_columns = (
        required_training_columns
        - set(training.columns)
    )

    if missing_training_columns:
        raise ValueError(
            "Training dataset is missing columns: "
            f"{sorted(missing_training_columns)}"
        )

    merged = grid.merge(
        training[
            [
                "grid_id",
                TARGET_COLUMN,
            ]
        ],
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(training):
        raise ValueError(
            "Training rows could not all be matched "
            "to grid geometry."
        )

    merged[TARGET_COLUMN] = (
        pd.to_numeric(
            merged[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    if set(
        merged[TARGET_COLUMN].unique()
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Target must contain both classes."
        )

    if merged.geometry.isna().any():
        raise ValueError(
            "Missing grid geometry was found."
        )

    if (
        ~merged.geometry.is_valid
    ).any():
        raise ValueError(
            "Invalid grid geometry was found."
        )

    return gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=grid.crs,
    )


def create_spatial_blocks(
    data: gpd.GeoDataFrame,
    *,
    block_size_m: int = BLOCK_SIZE_METERS,
) -> gpd.GeoDataFrame:
    """
    Assign each cell to a deterministic square spatial block.

    Block coordinates are anchored to the projected CRS origin rather
    than to the current dataset extent. This keeps IDs reproducible
    when the same area is processed again.
    """

    if block_size_m <= 0:
        raise ValueError(
            "Block size must be positive."
        )

    if data.crs is None:
        raise ValueError(
            "Input CRS is missing."
        )

    if not data.crs.is_projected:
        raise ValueError(
            "Spatial blocks require a projected CRS."
        )

    result = data.copy()

    points = (
        result.geometry
        .representative_point()
    )

    result[
        "spatial_block_x"
    ] = np.floor(
        points.x.to_numpy()
        / block_size_m
    ).astype(int)

    result[
        "spatial_block_y"
    ] = np.floor(
        points.y.to_numpy()
        / block_size_m
    ).astype(int)

    result[
        "spatial_block_id"
    ] = (
        "B"
        + result[
            "spatial_block_x"
        ].astype(str)
        + "_"
        + result[
            "spatial_block_y"
        ].astype(str)
    )

    return result


def summarize_blocks(
    data: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Summarize row and class counts for each spatial block."""

    required_columns = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Spatial data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        data.groupby(
            "spatial_block_id",
            as_index=False,
        )
        .agg(
            row_count=(
                "grid_id",
                "size",
            ),
            positive_count=(
                TARGET_COLUMN,
                "sum",
            ),
        )
    )

    summary[
        "negative_count"
    ] = (
        summary[
            "row_count"
        ]
        - summary[
            "positive_count"
        ]
    )

    summary = summary.sort_values(
        by=[
            "positive_count",
            "row_count",
            "spatial_block_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    return summary


def assign_blocks_to_folds(
    block_summary: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """
    Assign complete spatial blocks to folds.

    Stage 1:
        Blocks containing positive samples are assigned while
        prioritizing positive-count balance.

    Stage 2:
        Zero-positive blocks are assigned while prioritizing total
        row-count balance.

    This prevents the rare positive class from collapsing into only a
    few validation folds while also avoiding extremely unequal fold
    sizes.
    """

    if n_splits < 2:
        raise ValueError(
            "At least two folds are required."
        )

    if len(block_summary) < n_splits:
        raise ValueError(
            "There are fewer spatial blocks than folds."
        )

    required_columns = {
        "spatial_block_id",
        "row_count",
        "positive_count",
        "negative_count",
    }

    missing_columns = (
        required_columns
        - set(block_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Block summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    result = block_summary.copy()

    result["cv_fold"] = -1

    fold_positive_counts = np.zeros(
        n_splits,
        dtype=int,
    )

    fold_row_counts = np.zeros(
        n_splits,
        dtype=int,
    )

    positive_blocks = (
        result.loc[
            result["positive_count"] > 0
        ]
        .sort_values(
            by=[
                "positive_count",
                "row_count",
                "spatial_block_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="stable",
        )
    )

    for index, row in positive_blocks.iterrows():
        candidate_folds = sorted(
            range(n_splits),
            key=lambda fold: (
                fold_positive_counts[
                    fold
                ],
                fold_row_counts[
                    fold
                ],
                fold,
            ),
        )

        selected_fold = (
            candidate_folds[0]
        )

        result.loc[
            index,
            "cv_fold",
        ] = selected_fold

        fold_positive_counts[
            selected_fold
        ] += int(
            row["positive_count"]
        )

        fold_row_counts[
            selected_fold
        ] += int(
            row["row_count"]
        )

    zero_positive_blocks = (
        result.loc[
            result["positive_count"] == 0
        ]
        .sort_values(
            by=[
                "row_count",
                "spatial_block_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
    )

    for index, row in zero_positive_blocks.iterrows():
        selected_fold = min(
            range(n_splits),
            key=lambda fold: (
                fold_row_counts[
                    fold
                ],
                fold_positive_counts[
                    fold
                ],
                fold,
            ),
        )

        result.loc[
            index,
            "cv_fold",
        ] = selected_fold

        fold_row_counts[
            selected_fold
        ] += int(
            row["row_count"]
        )

    if (
        result["cv_fold"] < 0
    ).any():
        raise ValueError(
            "At least one spatial block was not assigned."
        )

    result[
        "cv_fold"
    ] = (
        result[
            "cv_fold"
        ].astype(int)
    )

    return result


def attach_fold_assignments(
    data: gpd.GeoDataFrame,
    block_assignments: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach block-level fold assignments to each grid cell."""

    result = data.merge(
        block_assignments[
            [
                "spatial_block_id",
                "cv_fold",
            ]
        ],
        on="spatial_block_id",
        how="left",
        validate="many_to_one",
    )

    if result[
        "cv_fold"
    ].isna().any():
        raise ValueError(
            "Some cells did not receive a CV fold."
        )

    result[
        "cv_fold"
    ] = (
        result[
            "cv_fold"
        ].astype(int)
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=data.crs,
    )


def validate_folds(
    data: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """Validate spatial block and class distribution."""

    required_columns = {
        "grid_id",
        TARGET_COLUMN,
        "spatial_block_id",
        "cv_fold",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if not data[
        "cv_fold"
    ].between(
        0,
        n_splits - 1,
    ).all():
        raise ValueError(
            "Invalid fold identifiers were found."
        )

    block_fold_counts = (
        data.groupby(
            "spatial_block_id"
        )[
            "cv_fold"
        ]
        .nunique()
    )

    if (
        block_fold_counts > 1
    ).any():
        raise ValueError(
            "A spatial block was split across folds."
        )

    fold_summary = (
        data.groupby(
            "cv_fold",
            as_index=False,
        )
        .agg(
            row_count=(
                "grid_id",
                "size",
            ),
            positive_count=(
                TARGET_COLUMN,
                "sum",
            ),
            block_count=(
                "spatial_block_id",
                "nunique",
            ),
        )
    )

    positive_block_counts = (
        data.loc[
            data[TARGET_COLUMN] == 1
        ]
        .groupby(
            "cv_fold"
        )[
            "spatial_block_id"
        ]
        .nunique()
    )

    fold_summary[
        "positive_block_count"
    ] = (
        fold_summary[
            "cv_fold"
        ]
        .map(
            positive_block_counts
        )
        .fillna(0)
        .astype(int)
    )

    fold_summary[
        "negative_count"
    ] = (
        fold_summary[
            "row_count"
        ]
        - fold_summary[
            "positive_count"
        ]
    )

    fold_summary[
        "positive_rate_percent"
    ] = (
        fold_summary[
            "positive_count"
        ]
        / fold_summary[
            "row_count"
        ]
        * 100
    )

    if len(fold_summary) != n_splits:
        raise ValueError(
            "Unexpected number of folds."
        )

    if (
        fold_summary[
            "positive_count"
        ] <= 0
    ).any():
        raise ValueError(
            "At least one spatial fold has no positive samples."
        )

    if int(
        fold_summary[
            "row_count"
        ].sum()
    ) != len(data):
        raise ValueError(
            "Fold row counts do not preserve the dataset."
        )

    if int(
        fold_summary[
            "positive_count"
        ].sum()
    ) != int(
        data[
            TARGET_COLUMN
        ].sum()
    ):
        raise ValueError(
            "Fold positive counts do not preserve the target."
        )

    return fold_summary


def save_outputs(
    data: gpd.GeoDataFrame,
    block_assignments: pd.DataFrame,
) -> None:
    """Save row-level and block-level assignments."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_output = pd.DataFrame(
        data[
            [
                "grid_id",
                TARGET_COLUMN,
                "spatial_block_id",
                "spatial_block_x",
                "spatial_block_y",
                "cv_fold",
            ]
        ]
    )

    row_output = row_output.sort_values(
        "grid_id",
        kind="stable",
    )

    block_output = (
        block_assignments
        .sort_values(
            "spatial_block_id",
            kind="stable",
        )
    )

    row_output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    block_output.to_csv(
        BLOCK_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary(
    data: gpd.GeoDataFrame,
    block_assignments: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    """Create reproducibility documentation."""

    fold_lines = "\n".join(
        (
            f"- Fold {int(row.cv_fold)}: "
            f"{int(row.row_count):,} rows, "
            f"{int(row.positive_count):,} positives, "
            f"{int(row.block_count):,} blocks, "
            f"{int(row.positive_block_count):,} positive blocks, "
            f"{row.positive_rate_percent:.4f}% positive"
        )
        for row in fold_summary.itertuples(
            index=False
        )
    )

    positive_blocks = int(
        (
            block_assignments[
                "positive_count"
            ] > 0
        ).sum()
    )

    minimum_rows = int(
        fold_summary[
            "row_count"
        ].min()
    )

    maximum_rows = int(
        fold_summary[
            "row_count"
        ].max()
    )

    minimum_positives = int(
        fold_summary[
            "positive_count"
        ].min()
    )

    maximum_positives = int(
        fold_summary[
            "positive_count"
        ].max()
    )

    summary = f"""# Ankara Spatial Cross-Validation Plan

## Configuration

- Grid rows: {len(data):,}
- Spatial block size: {BLOCK_SIZE_METERS / 1000:.0f} km
- Cross-validation folds: {N_SPLITS}
- Spatial blocks: {data["spatial_block_id"].nunique():,}
- Blocks containing at least one positive sample: {positive_blocks:,}
- Total positive samples: {int(data[TARGET_COLUMN].sum()):,}

## Fold Distribution

{fold_lines}

## Balance

- Minimum fold rows: {minimum_rows:,}
- Maximum fold rows: {maximum_rows:,}
- Minimum fold positives: {minimum_positives:,}
- Maximum fold positives: {maximum_positives:,}

## Assignment Strategy

Spatial blocks are 5 x 5 kilometre groups anchored to the projected
CRS coordinate system.

The assignment algorithm runs in two stages.

First, blocks containing existing charging-station cells are assigned
while prioritizing positive-count balance across folds.

Second, blocks containing no positive samples are assigned while
prioritizing total row-count balance.

A complete block always belongs to exactly one fold.

## Interpretation

This is spatial block cross-validation rather than ordinary random or
stratified row-level splitting.

Cells inside the same 5-kilometre block are kept together. This reduces
local train-validation dependence compared with random splitting.

However, cells located on opposite sides of neighboring block
boundaries can still belong to different folds. Therefore this method
should not be described as eliminating every possible form of spatial
dependence.

## Rare-Class Limitation

Only {int(data[TARGET_COLUMN].sum()):,} of {len(data):,} grid cells
contain an existing charging station.

Model evaluation should therefore emphasize:

- average precision / PR-AUC
- recall
- precision
- F1
- ROC-AUC as a secondary metric
- fold-level stability

Accuracy should not be used as the primary metric.

## Outputs

- `data/processed/ankara_spatial_cv_folds.csv`
- `data/processed/ankara_spatial_cv_block_summary.csv`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_fold_summary(
    data: gpd.GeoDataFrame,
    block_assignments: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    """Print key CV diagnostics."""

    print(
        "Spatial block size:",
        f"{BLOCK_SIZE_METERS / 1000:.1f} km",
    )

    print(
        "Spatial block count:",
        f"{data['spatial_block_id'].nunique():,}",
    )

    print(
        "Positive blocks:",
        f"{int((block_assignments['positive_count'] > 0).sum()):,}",
    )

    print()

    print(
        fold_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Fold row range:",
        f"{int(fold_summary['row_count'].min()):,}",
        "-",
        f"{int(fold_summary['row_count'].max()):,}",
    )

    print(
        "Fold positive range:",
        f"{int(fold_summary['positive_count'].min()):,}",
        "-",
        f"{int(fold_summary['positive_count'].max()):,}",
    )


def main() -> None:
    """Create Ankara spatial block cross-validation folds."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Spatial Cross-Validation Plan"
    )

    print("=" * 70)

    data = load_inputs()

    data = create_spatial_blocks(
        data
    )

    block_summary = summarize_blocks(
        data
    )

    block_assignments = (
        assign_blocks_to_folds(
            block_summary
        )
    )

    data = attach_fold_assignments(
        data,
        block_assignments,
    )

    fold_summary = validate_folds(
        data
    )

    save_outputs(
        data,
        block_assignments,
    )

    create_summary(
        data,
        block_assignments,
        fold_summary,
    )

    print_fold_summary(
        data,
        block_assignments,
        fold_summary,
    )

    print("=" * 70)

    print(
        "Ankara spatial CV plan completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
