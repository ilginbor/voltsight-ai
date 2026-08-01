from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_site_dataset.csv"
)

SCORE_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.csv"
)

SCORE_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.gpkg"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_candidate_suitability_summary.md"
)

SCORE_LAYER_NAME = "candidate_suitability_scores"

SCORE_COLUMNS = [
    "road_proximity_score",
    "main_road_presence_score",
    "road_density_score",
    "parking_proximity_score",
    "parking_coverage_score",
    "parking_area_score",
    "charging_distance_gap_score",
    "charging_density_gap_score",
    "ac_gap_score",
    "dc_gap_score",
    "accessibility_score",
    "parking_score",
    "infrastructure_gap_score",
    "technology_gap_score",
    "feasibility_score",
    "need_score",
    "suitability_score",
    "suitability_percentile",
]


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a required scoring output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run create_suitability_scores.py first."
        )

    if path.stat().st_size == 0:
        pytest.fail(
            f"{description} is empty: {path}"
        )


@pytest.fixture(scope="session")
def candidates() -> pd.DataFrame:
    """Load the candidate-site source dataset."""

    ensure_file_exists(
        CANDIDATE_PATH,
        "Candidate-site dataset",
    )

    return pd.read_csv(
        CANDIDATE_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def scores() -> pd.DataFrame:
    """Load suitability score CSV output."""

    ensure_file_exists(
        SCORE_CSV_PATH,
        "Suitability score CSV",
    )

    return pd.read_csv(
        SCORE_CSV_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def spatial_scores() -> gpd.GeoDataFrame:
    """Load suitability polygon output."""

    ensure_file_exists(
        SCORE_GPKG_PATH,
        "Suitability score GeoPackage",
    )

    return gpd.read_file(
        SCORE_GPKG_PATH,
        layer=SCORE_LAYER_NAME,
    )


def test_suitability_outputs_exist() -> None:
    """All scoring outputs must exist and contain data."""

    required_outputs = (
        CANDIDATE_PATH,
        SCORE_CSV_PATH,
        SCORE_GPKG_PATH,
        SUMMARY_PATH,
    )

    for output_path in required_outputs:
        ensure_file_exists(
            output_path,
            output_path.name,
        )


def test_score_row_count_and_grid_ids(
    scores: pd.DataFrame,
) -> None:
    """Every candidate grid must receive exactly one score."""

    assert len(scores) == 7_217

    assert scores["grid_id"].notna().all()
    assert scores["grid_id"].is_unique

    assert scores[
        "score_explanation"
    ].notna().all()

    assert scores[
        "score_explanation"
    ].astype(str).str.strip().ne("").all()


def test_scored_grid_ids_match_candidate_source(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Scoring must neither add nor remove candidate grid cells."""

    candidate_ids = set(
        candidates["grid_id"].astype(str)
    )

    score_ids = set(
        scores["grid_id"].astype(str)
    )

    assert score_ids == candidate_ids


def test_score_columns_are_finite_and_bounded(
    scores: pd.DataFrame,
) -> None:
    """Every component score must be finite and between zero and 100."""

    for column in SCORE_COLUMNS:
        assert column in scores.columns

        values = pd.to_numeric(
            scores[column],
            errors="coerce",
        )

        assert values.notna().all()

        assert np.isfinite(
            values.to_numpy(
                dtype=float
            )
        ).all()

        assert values.between(
            0.0,
            100.0,
            inclusive="both",
        ).all()


def test_weighted_score_formulas(
    scores: pd.DataFrame,
) -> None:
    """Saved component scores must reproduce the documented formulas."""

    expected_accessibility = (
        scores["road_proximity_score"] * 0.45
        + scores["main_road_presence_score"] * 0.35
        + scores["road_density_score"] * 0.20
    )

    expected_parking = (
        scores["parking_proximity_score"] * 0.45
        + scores["parking_coverage_score"] * 0.35
        + scores["parking_area_score"] * 0.20
    )

    expected_infrastructure_gap = (
        scores["charging_distance_gap_score"] * 0.75
        + scores["charging_density_gap_score"] * 0.25
    )

    expected_technology_gap = (
        scores["dc_gap_score"] * 0.60
        + scores["ac_gap_score"] * 0.40
    )

    expected_feasibility = (
        scores["accessibility_score"] * 0.60
        + scores["parking_score"] * 0.40
    )

    expected_need = (
        scores["infrastructure_gap_score"] * 0.85
        + scores["technology_gap_score"] * 0.15
    )

    expected_suitability = np.sqrt(
        scores["feasibility_score"]
        * scores["need_score"]
    )

    comparisons = (
        (
            scores["accessibility_score"],
            expected_accessibility,
        ),
        (
            scores["parking_score"],
            expected_parking,
        ),
        (
            scores["infrastructure_gap_score"],
            expected_infrastructure_gap,
        ),
        (
            scores["technology_gap_score"],
            expected_technology_gap,
        ),
        (
            scores["feasibility_score"],
            expected_feasibility,
        ),
        (
            scores["need_score"],
            expected_need,
        ),
        (
            scores["suitability_score"],
            expected_suitability,
        ),
    )

    for actual, expected in comparisons:
        np.testing.assert_allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            atol=0.02,
            rtol=0.0,
        )


def test_ranks_and_percentiles_are_complete(
    scores: pd.DataFrame,
) -> None:
    """Ranks must be unique and percentiles must follow rank order."""

    ranks = pd.to_numeric(
        scores["suitability_rank"],
        errors="coerce",
    )

    assert ranks.notna().all()
    assert ranks.is_unique

    assert set(
        ranks.astype(int)
    ) == set(
        range(
            1,
            len(scores) + 1,
        )
    )

    first = scores.loc[
        ranks.idxmin()
    ]

    last = scores.loc[
        ranks.idxmax()
    ]

    assert float(
        first["suitability_percentile"]
    ) == pytest.approx(
        100.0,
        abs=0.0001,
    )

    assert float(
        last["suitability_percentile"]
    ) == pytest.approx(
        0.0,
        abs=0.0001,
    )

    ordered = scores.sort_values(
        "suitability_rank"
    )

    assert ordered[
        "suitability_score"
    ].is_monotonic_decreasing


def test_priority_band_distribution(
    scores: pd.DataFrame,
) -> None:
    """Relative priority bands must reproduce the expected population."""

    counts = (
        scores["priority_band"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert counts == {
        "A - Highest priority": 73,
        "B - High priority": 288,
        "C - Medium priority": 1_083,
        "D - Lower priority": 2_165,
        "E - Lowest priority": 3_608,
    }


def test_highest_ranked_candidate_snapshot(
    scores: pd.DataFrame,
) -> None:
    """The current reproducible dataset must retain its top candidate."""

    top = (
        scores.sort_values(
            "suitability_rank"
        )
        .iloc[0]
    )

    assert top["grid_id"] == "CKY_00162"
    assert int(top["suitability_rank"]) == 1

    assert float(
        top["suitability_score"]
    ) == pytest.approx(
        77.3917,
        abs=0.0001,
    )

    assert float(
        top["feasibility_score"]
    ) == pytest.approx(
        96.3723,
        abs=0.0001,
    )

    assert float(
        top["need_score"]
    ) == pytest.approx(
        62.1493,
        abs=0.0001,
    )


def test_spatial_output_matches_csv(
    scores: pd.DataFrame,
    spatial_scores: gpd.GeoDataFrame,
) -> None:
    """CSV and polygon outputs must contain the same candidate IDs."""

    assert len(spatial_scores) == len(scores)

    assert spatial_scores.crs is not None

    assert spatial_scores[
        "grid_id"
    ].notna().all()

    assert spatial_scores[
        "grid_id"
    ].is_unique

    assert not spatial_scores.geometry.isna().any()
    assert not spatial_scores.geometry.is_empty.any()

    assert set(
        spatial_scores["grid_id"].astype(str)
    ) == set(
        scores["grid_id"].astype(str)
    )


def test_suitability_summary_documents_method() -> None:
    """The summary must describe the scoring method and limitations."""

    ensure_file_exists(
        SUMMARY_PATH,
        "Suitability summary",
    )

    summary = SUMMARY_PATH.read_text(
        encoding="utf-8"
    )

    required_text = (
        "Candidate rows: 7,217",
        "This is an explainable decision-support score",
        "45% proximity to a main road",
        "Feasibility = 60% accessibility + 40% parking",
        "Suitability = square root of feasibility multiplied by need",
        "A: top 1%",
        "Important Limitations",
    )

    for text in required_text:
        assert text in summary
