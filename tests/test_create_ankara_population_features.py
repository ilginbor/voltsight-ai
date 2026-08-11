from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box

from voltsight.features.create_ankara_population_features import (
    GRID_CELL_AREA_KM2,
    build_population_features,
    calculate_neighborhood_sum,
    create_circular_offsets,
    create_grid_cell_indices,
    create_study_grid_mask,
    derive_target_grid,
    validate_population_features,
)


def create_test_grid() -> gpd.GeoDataFrame:
    """Create a complete 3 x 3 synthetic 500-m grid."""

    rows: list[
        dict[str, object]
    ] = []

    grid_number = 1

    for row_index in range(3):
        for column_index in range(3):
            minx = (
                column_index
                * 500.0
            )

            maxx = (
                minx
                + 500.0
            )

            maxy = (
                1500.0
                - row_index
                * 500.0
            )

            miny = (
                maxy
                - 500.0
            )

            rows.append(
                {
                    "grid_id": (
                        f"ANK_{grid_number:06d}"
                    ),
                    "geometry": box(
                        minx,
                        miny,
                        maxx,
                        maxy,
                    ),
                }
            )

            grid_number += 1

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:32636",
    )


def test_circular_offsets_match_expected_lattice_sizes() -> None:
    one_km = create_circular_offsets(
        1_000,
        cell_size_m=500,
    )

    two_km = create_circular_offsets(
        2_000,
        cell_size_m=500,
    )

    assert len(one_km) == 13
    assert len(two_km) == 49
    assert (0, 0) in one_km
    assert (0, 0) in two_km


def test_neighborhood_sum_includes_center_and_expected_neighbors() -> None:
    surface = np.zeros(
        (
            7,
            7,
        ),
        dtype=float,
    )

    surface[
        3,
        3,
    ] = 10.0

    result = calculate_neighborhood_sum(
        surface,
        radius_m=1_000,
        cell_size_m=500,
    )

    assert result[
        3,
        3
    ] == 10.0

    assert result[
        3,
        5
    ] == 10.0

    assert result[
        1,
        1
    ] == 0.0


def test_target_grid_and_vector_cell_indices_are_aligned() -> None:
    grid = create_test_grid()

    transform, height, width = (
        derive_target_grid(
            grid
        )
    )

    rows, columns = (
        create_grid_cell_indices(
            grid,
            transform,
            height=height,
            width=width,
        )
    )

    assert height == 3
    assert width == 3

    assert np.array_equal(
        rows,
        np.array(
            [
                0, 0, 0,
                1, 1, 1,
                2, 2, 2,
            ]
        ),
    )

    assert np.array_equal(
        columns,
        np.array(
            [
                0, 1, 2,
                0, 1, 2,
                0, 1, 2,
            ]
        ),
    )


def test_build_population_features_uses_fixed_cell_area() -> None:
    grid = create_test_grid()

    transform = from_origin(
        0,
        1500,
        500,
        500,
    )

    rows, columns = (
        create_grid_cell_indices(
            grid,
            transform,
            height=3,
            width=3,
        )
    )

    mask = create_study_grid_mask(
        rows,
        columns,
        height=3,
        width=3,
    )

    population_surface = np.arange(
        1,
        10,
        dtype=float,
    ).reshape(
        3,
        3,
    )

    result = build_population_features(
        grid,
        population_surface,
        rows,
        columns,
        mask,
    )

    assert np.allclose(
        result[
            "population_density_per_km2"
        ].to_numpy(
            dtype=float
        ),
        result[
            "population_count"
        ].to_numpy(
            dtype=float
        )
        / GRID_CELL_AREA_KM2,
    )


def test_validate_population_features_accepts_complete_output() -> None:
    grid = create_test_grid()

    transform, height, width = (
        derive_target_grid(
            grid
        )
    )

    rows, columns = (
        create_grid_cell_indices(
            grid,
            transform,
            height=height,
            width=width,
        )
    )

    mask = create_study_grid_mask(
        rows,
        columns,
        height=height,
        width=width,
    )

    population_surface = np.full(
        (
            height,
            width,
        ),
        10.0,
    )

    result = build_population_features(
        grid,
        population_surface,
        rows,
        columns,
        mask,
    )

    validate_population_features(
        result,
        expected_rows=len(grid),
    )

    assert len(result) == 9
    assert pd.api.types.is_numeric_dtype(
        result[
            "population_within_2000m"
        ]
    )
