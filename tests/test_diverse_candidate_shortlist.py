from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.csv"
)

SHORTLIST_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_diverse_candidate_shortlist.csv"
)

SHORTLIST_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_diverse_candidate_shortlist.gpkg"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_diverse_candidate_shortlist_summary.md"
)

SHORTLIST_LAYER_NAME = "diverse_candidate_shortlist"

MINIMUM_SUITABILITY_SCORE = 60.0
MINIMUM_FEASIBILITY_SCORE = 60.0
MINIMUM_NEED_SCORE = 50.0
MINIMUM_SPACING_METERS = 1_000.0


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a required shortlist output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run create_diverse_candidate_shortlist.py first."
        )

    if path.stat().st_size == 0:
        pytest.fail(
            f"{description} is empty: {path}"
        )


@pytest.fixture(scope="session")
def source_scores() -> pd.DataFrame:
    """Load the complete candidate suitability table."""

    ensure_file_exists(
        SOURCE_SCORE_PATH,
        "Suitability score CSV",
    )

    return pd.read_csv(
        SOURCE_SCORE_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def shortlist() -> pd.DataFrame:
    """Load the spatially diverse shortlist CSV."""

    ensure_file_exists(
        SHORTLIST_CSV_PATH,
        "Diverse shortlist CSV",
    )

    return pd.read_csv(
        SHORTLIST_CSV_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def spatial_shortlist() -> gpd.GeoDataFrame:
    """Load the spatially diverse shortlist polygons."""

    ensure_file_exists(
        SHORTLIST_GPKG_PATH,
        "Diverse shortlist GeoPackage",
    )

    return gpd.read_file(
        SHORTLIST_GPKG_PATH,
        layer=SHORTLIST_LAYER_NAME,
    )


def test_shortlist_outputs_exist() -> None:
    """Every shortlist output must exist and contain data."""

    required_outputs = (
        SOURCE_SCORE_PATH,
        SHORTLIST_CSV_PATH,
        SHORTLIST_GPKG_PATH,
        SUMMARY_PATH,
    )

    for output_path in required_outputs:
        ensure_file_exists(
            output_path,
            output_path.name,
        )


def test_shortlist_count_ids_and_selection_ranks(
    shortlist: pd.DataFrame,
) -> None:
    """The shortlist must contain twenty unique sequential candidates."""

    assert len(shortlist) == 20

    assert shortlist["grid_id"].notna().all()
    assert shortlist["grid_id"].is_unique

    assert shortlist[
        "diverse_selection_rank"
    ].astype(int).tolist() == list(
        range(1, 21)
    )

    assert not shortlist.isna().any().any()


def test_shortlist_satisfies_score_thresholds(
    shortlist: pd.DataFrame,
) -> None:
    """Every selected candidate must pass all eligibility floors."""

    assert shortlist[
        "suitability_score"
    ].ge(
        MINIMUM_SUITABILITY_SCORE
    ).all()

    assert shortlist[
        "feasibility_score"
    ].ge(
        MINIMUM_FEASIBILITY_SCORE
    ).all()

    assert shortlist[
        "need_score"
    ].ge(
        MINIMUM_NEED_SCORE
    ).all()

    assert float(
        shortlist["suitability_score"].min()
    ) == pytest.approx(
        67.4030,
        abs=0.0001,
    )

    assert float(
        shortlist["feasibility_score"].min()
    ) == pytest.approx(
        70.5690,
        abs=0.0001,
    )

    assert float(
        shortlist["need_score"].min()
    ) == pytest.approx(
        53.8725,
        abs=0.0001,
    )


def test_source_eligibility_count(
    source_scores: pd.DataFrame,
) -> None:
    """The current score snapshot must produce 346 eligible cells."""

    eligible = source_scores.loc[
        source_scores["suitability_score"].ge(
            MINIMUM_SUITABILITY_SCORE
        )
        & source_scores["feasibility_score"].ge(
            MINIMUM_FEASIBILITY_SCORE
        )
        & source_scores["need_score"].ge(
            MINIMUM_NEED_SCORE
        )
    ]

    assert len(eligible) == 346


def test_shortlist_rows_match_source_scores(
    source_scores: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> None:
    """Selected attributes must match the complete score dataset."""

    columns = [
        "suitability_rank",
        "suitability_score",
        "suitability_percentile",
        "feasibility_score",
        "need_score",
        "accessibility_score",
        "parking_score",
        "infrastructure_gap_score",
        "technology_gap_score",
    ]

    merged = shortlist[
        [
            "grid_id",
            *columns,
        ]
    ].merge(
        source_scores[
            [
                "grid_id",
                *columns,
            ]
        ],
        on="grid_id",
        how="left",
        validate="one_to_one",
        suffixes=("_shortlist", "_source"),
    )

    assert len(merged) == len(shortlist)

    for column in columns:
        shortlist_values = pd.to_numeric(
            merged[f"{column}_shortlist"],
            errors="coerce",
        )

        source_values = pd.to_numeric(
            merged[f"{column}_source"],
            errors="coerce",
        )

        np.testing.assert_allclose(
            shortlist_values,
            source_values,
            atol=0.0001,
            rtol=0.0,
        )


def test_original_rank_order_and_snapshot(
    shortlist: pd.DataFrame,
) -> None:
    """Greedy selection must preserve ascending original rank order."""

    original_ranks = shortlist[
        "suitability_rank"
    ].astype(int)

    assert original_ranks.is_monotonic_increasing
    assert original_ranks.is_unique

    assert original_ranks.iloc[0] == 1
    assert original_ranks.iloc[-1] == 114

    assert shortlist.iloc[0][
        "grid_id"
    ] == "CKY_00162"

    assert shortlist.iloc[-1][
        "grid_id"
    ] == "CKY_01481"


def test_stored_spacing_and_nearest_ids(
    shortlist: pd.DataFrame,
) -> None:
    """Stored nearest-candidate information must be internally valid."""

    distances = pd.to_numeric(
        shortlist[
            "nearest_selected_candidate_m"
        ],
        errors="coerce",
    )

    assert distances.notna().all()

    assert distances.ge(
        MINIMUM_SPACING_METERS
    ).all()

    assert float(
        distances.min()
    ) == pytest.approx(
        1_000.0,
        abs=0.01,
    )

    selected_ids = set(
        shortlist["grid_id"].astype(str)
    )

    nearest_ids = set(
        shortlist[
            "nearest_selected_grid_id"
        ].astype(str)
    )

    assert nearest_ids.issubset(
        selected_ids
    )

    assert (
        shortlist["grid_id"].astype(str)
        != shortlist[
            "nearest_selected_grid_id"
        ].astype(str)
    ).all()


def test_pairwise_geometry_spacing(
    spatial_shortlist: gpd.GeoDataFrame,
) -> None:
    """Recomputed polygon-centroid distances must respect one kilometre."""

    assert spatial_shortlist.crs is not None

    projected = spatial_shortlist.to_crs(
        "EPSG:32636"
    ).copy()

    points = list(
        projected.geometry.centroid
    )

    actual_nearest_distances: dict[str, float] = {}

    for index, row in projected.iterrows():
        position = projected.index.get_loc(
            index
        )

        distances = [
            points[position].distance(
                other_point
            )
            for other_position, other_point
            in enumerate(points)
            if other_position != position
        ]

        actual_nearest_distances[
            str(row["grid_id"])
        ] = float(
            min(distances)
        )

    assert min(
        actual_nearest_distances.values()
    ) >= 999.99

    stored = (
        projected[
            [
                "grid_id",
                "nearest_selected_candidate_m",
            ]
        ]
        .copy()
    )

    for _, row in stored.iterrows():
        actual_distance = (
            actual_nearest_distances[
                str(row["grid_id"])
            ]
        )

        assert float(
            row[
                "nearest_selected_candidate_m"
            ]
        ) == pytest.approx(
            actual_distance,
            abs=0.02,
        )


def test_spatial_output_matches_csv(
    shortlist: pd.DataFrame,
    spatial_shortlist: gpd.GeoDataFrame,
) -> None:
    """CSV and GeoPackage shortlist outputs must contain the same IDs."""

    assert len(spatial_shortlist) == len(
        shortlist
    )

    assert spatial_shortlist.crs is not None

    assert spatial_shortlist[
        "grid_id"
    ].notna().all()

    assert spatial_shortlist[
        "grid_id"
    ].is_unique

    assert not spatial_shortlist.geometry.isna().any()
    assert not spatial_shortlist.geometry.is_empty.any()

    assert set(
        spatial_shortlist["grid_id"].astype(str)
    ) == set(
        shortlist["grid_id"].astype(str)
    )


def test_shortlist_has_geographic_diversity(
    shortlist: pd.DataFrame,
) -> None:
    """The shortlist must cover more than one local candidate cluster."""

    longitude_range = (
        shortlist["center_longitude"].max()
        - shortlist["center_longitude"].min()
    )

    latitude_range = (
        shortlist["center_latitude"].max()
        - shortlist["center_latitude"].min()
    )

    assert longitude_range > 0.15
    assert latitude_range > 0.10

    assert shortlist[
        "nearest_selected_candidate_m"
    ].max() > 10_000


def test_shortlist_summary_documents_rules() -> None:
    """The summary must document eligibility and spacing rules."""

    ensure_file_exists(
        SUMMARY_PATH,
        "Diverse shortlist summary",
    )

    summary = SUMMARY_PATH.read_text(
        encoding="utf-8"
    )

    normalized_summary = " ".join(
        summary.split()
    )

    required_text = (
        "Total scored candidates: 7,217",
        "Eligible candidates after score thresholds: 346",
        "Selected candidates: 20",
        "Suitability score: at least 60/100",
        "Feasibility score: at least 60/100",
        "Need score: at least 50/100",
        "at least 1,000 metres",
        "Worst original suitability rank selected: 114",
        "field review",
    )

    for text in required_text:
        assert text in normalized_summary
