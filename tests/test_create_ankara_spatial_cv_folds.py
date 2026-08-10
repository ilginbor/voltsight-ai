from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from voltsight.models.create_ankara_spatial_cv_folds import (
    BLOCK_SIZE_METERS,
    assign_blocks_to_folds,
    create_spatial_blocks,
    summarize_blocks,
    validate_folds,
)


def create_grid() -> gpd.GeoDataFrame:
    """Create synthetic cells across several spatial blocks."""

    rows = []

    for index in range(
        15
    ):
        x = (
            index * 6_000
        )

        rows.append(
            {
                "grid_id": (
                    f"ANK_{index + 1:06d}"
                ),
                "has_existing_charging_station": (
                    1
                    if index < 10
                    else 0
                ),
                "geometry": box(
                    x,
                    0,
                    x + 500,
                    500,
                ),
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def test_default_block_size_is_five_kilometres() -> None:
    """Ankara spatial CV should use 5-km blocks."""

    assert BLOCK_SIZE_METERS == 5_000


def test_spatial_blocks_are_deterministic() -> None:
    """Cells must receive stable block IDs."""

    data = create_grid()

    first = create_spatial_blocks(
        data,
        block_size_m=5_000,
    )

    second = create_spatial_blocks(
        data,
        block_size_m=5_000,
    )

    assert (
        first[
            "spatial_block_id"
        ].tolist()
        ==
        second[
            "spatial_block_id"
        ].tolist()
    )

    assert first[
        "spatial_block_id"
    ].notna().all()

    assert first[
        "spatial_block_id"
    ].nunique() > 1


def test_block_summary_preserves_rows_and_positives() -> None:
    """Block aggregation must preserve dataset totals."""

    data = create_spatial_blocks(
        create_grid(),
        block_size_m=5_000,
    )

    summary = summarize_blocks(
        data
    )

    assert int(
        summary[
            "row_count"
        ].sum()
    ) == len(data)

    assert int(
        summary[
            "positive_count"
        ].sum()
    ) == 10


def test_assignment_uses_all_requested_folds() -> None:
    """Positive blocks should be distributed across all folds."""

    data = create_spatial_blocks(
        create_grid(),
        block_size_m=5_000,
    )

    summary = summarize_blocks(
        data
    )

    assigned = assign_blocks_to_folds(
        summary,
        n_splits=5,
    )

    assert assigned[
        "cv_fold"
    ].between(
        0,
        4,
    ).all()

    assert set(
        assigned[
            "cv_fold"
        ].unique()
    ) == {
        0,
        1,
        2,
        3,
        4,
    }


def test_positive_blocks_are_balanced_before_zero_blocks() -> None:
    """Positive samples should remain distributed among folds."""

    block_summary = pd.DataFrame(
        {
            "spatial_block_id": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],
            "row_count": [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                100,
                500,
                500,
            ],
            "positive_count": [
                3,
                3,
                2,
                2,
                2,
                2,
                1,
                1,
                0,
                0,
            ],
        }
    )

    block_summary[
        "negative_count"
    ] = (
        block_summary[
            "row_count"
        ]
        - block_summary[
            "positive_count"
        ]
    )

    result = assign_blocks_to_folds(
        block_summary,
        n_splits=5,
    )

    positive_by_fold = (
        result.groupby(
            "cv_fold"
        )[
            "positive_count"
        ]
        .sum()
    )

    assert (
        positive_by_fold > 0
    ).all()

    assert (
        positive_by_fold.max()
        - positive_by_fold.min()
    ) <= 2


def test_zero_positive_blocks_help_balance_rows() -> None:
    """Zero-positive blocks should be sent toward smaller folds."""

    block_summary = pd.DataFrame(
        {
            "spatial_block_id": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],
            "row_count": [
                100,
                100,
                100,
                100,
                100,
                500,
                500,
                500,
                500,
                500,
            ],
            "positive_count": [
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )

    block_summary[
        "negative_count"
    ] = (
        block_summary[
            "row_count"
        ]
        - block_summary[
            "positive_count"
        ]
    )

    result = assign_blocks_to_folds(
        block_summary,
        n_splits=5,
    )

    row_counts = (
        result.groupby(
            "cv_fold"
        )[
            "row_count"
        ]
        .sum()
    )

    assert len(
        row_counts
    ) == 5

    assert (
        row_counts.max()
        - row_counts.min()
    ) <= 500


def test_validate_folds_accepts_complete_blocks() -> None:
    """A valid grouped fold layout should pass validation."""

    frame = pd.DataFrame(
        {
            "grid_id": [
                "A",
                "B",
                "C",
                "D",
            ],
            "spatial_block_id": [
                "B1",
                "B1",
                "B2",
                "B2",
            ],
            "cv_fold": [
                0,
                0,
                1,
                1,
            ],
            "has_existing_charging_station": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    summary = validate_folds(
        frame,
        n_splits=2,
    )

    assert len(
        summary
    ) == 2

    assert int(
        summary[
            "positive_count"
        ].sum()
    ) == 2


def test_validate_folds_rejects_split_blocks() -> None:
    """One spatial block may not appear in two folds."""

    frame = pd.DataFrame(
        {
            "grid_id": [
                "A",
                "B",
            ],
            "spatial_block_id": [
                "B0_0",
                "B0_0",
            ],
            "cv_fold": [
                0,
                1,
            ],
            "has_existing_charging_station": [
                1,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="split across folds",
    ):
        validate_folds(
            frame,
            n_splits=2,
        )


def test_validate_folds_rejects_fold_without_positive() -> None:
    """Every validation fold must contain a positive cell."""

    frame = pd.DataFrame(
        {
            "grid_id": [
                "A",
                "B",
                "C",
                "D",
            ],
            "spatial_block_id": [
                "B1",
                "B1",
                "B2",
                "B2",
            ],
            "cv_fold": [
                0,
                0,
                1,
                1,
            ],
            "has_existing_charging_station": [
                1,
                0,
                0,
                0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="no positive samples",
    ):
        validate_folds(
            frame,
            n_splits=2,
        )
