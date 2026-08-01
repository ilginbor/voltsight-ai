from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.csv"
)

TRAINING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_existing_station_training_dataset.csv"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_site_dataset.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_model_dataset_summary.md"
)

TARGET_COLUMN = "has_existing_charging_station"

LEAKAGE_COLUMNS = {
    "charging_station_count",
    "known_charging_capacity",
    "charging_capacity_record_count",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
}

EXPECTED_TRAINING_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "grid_size_m",
    "cell_area_m2",
    "center_longitude",
    "center_latitude",
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
    "parking_count",
    "known_parking_capacity",
    "parking_capacity_record_count",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "parking_area_ratio",
    TARGET_COLUMN,
}

EXPECTED_CANDIDATE_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "grid_size_m",
    "cell_area_m2",
    "center_longitude",
    "center_latitude",
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "nearest_main_road_type",
    "parking_count",
    "known_parking_capacity",
    "parking_capacity_record_count",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "parking_area_ratio",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
}


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a model-dataset output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run create_model_dataset.py first."
        )

    if path.stat().st_size == 0:
        pytest.fail(
            f"{description} is empty: {path}"
        )


@pytest.fixture(scope="session")
def source() -> pd.DataFrame:
    """Load the complete charging-feature source dataset."""

    ensure_file_exists(
        SOURCE_PATH,
        "Charging-feature source CSV",
    )

    return pd.read_csv(
        SOURCE_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def training() -> pd.DataFrame:
    """Load the leakage-safe supervised dataset."""

    ensure_file_exists(
        TRAINING_PATH,
        "Existing-station training CSV",
    )

    return pd.read_csv(
        TRAINING_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def candidates() -> pd.DataFrame:
    """Load the candidate-site ranking dataset."""

    ensure_file_exists(
        CANDIDATE_PATH,
        "Candidate-site CSV",
    )

    return pd.read_csv(
        CANDIDATE_PATH,
        encoding="utf-8-sig",
    )


def test_model_dataset_outputs_exist() -> None:
    """Every reproducible model-dataset output must exist."""

    required_outputs = (
        SOURCE_PATH,
        TRAINING_PATH,
        CANDIDATE_PATH,
        SUMMARY_PATH,
    )

    for output_path in required_outputs:
        ensure_file_exists(
            output_path,
            output_path.name,
        )


def test_training_row_count_matches_source(
    source: pd.DataFrame,
    training: pd.DataFrame,
) -> None:
    """Every source grid cell must appear in supervised training."""

    assert len(source) == 7_227
    assert len(training) == len(source)


def test_target_distribution(
    training: pd.DataFrame,
) -> None:
    """The existing-station target must contain ten positives."""

    target = pd.to_numeric(
        training[TARGET_COLUMN],
        errors="coerce",
    )

    assert target.notna().all()

    assert set(
        target.astype(int).unique()
    ) == {0, 1}

    assert int(target.sum()) == 10

    assert int(
        target.eq(0).sum()
    ) == 7_217


def test_training_schema(
    training: pd.DataFrame,
) -> None:
    """The supervised dataset must contain its expected columns."""

    assert set(training.columns) == (
        EXPECTED_TRAINING_COLUMNS
    )

    assert len(training.columns) == 23


def test_training_excludes_charging_leakage(
    training: pd.DataFrame,
) -> None:
    """Charging-derived columns must not reveal the target."""

    leaked_columns = (
        LEAKAGE_COLUMNS
        & set(training.columns)
    )

    assert leaked_columns == set()


def test_candidate_schema_and_count(
    candidates: pd.DataFrame,
) -> None:
    """The ranking dataset must contain only station-free cells."""

    assert len(candidates) == 7_217

    assert set(candidates.columns) == (
        EXPECTED_CANDIDATE_COLUMNS
    )

    assert len(candidates.columns) == 27

    assert TARGET_COLUMN not in candidates.columns


def test_candidate_ids_match_negative_training_rows(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Candidate IDs must exactly equal grids whose target is zero."""

    negative_ids = set(
        training.loc[
            training[TARGET_COLUMN].eq(0),
            "grid_id",
        ].astype(str)
    )

    candidate_ids = set(
        candidates["grid_id"].astype(str)
    )

    assert candidate_ids == negative_ids


def test_grid_ids_are_unique(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Training and candidate outputs must not duplicate grid cells."""

    assert training["grid_id"].notna().all()
    assert candidates["grid_id"].notna().all()

    assert training["grid_id"].is_unique
    assert candidates["grid_id"].is_unique


def test_outputs_have_no_missing_or_infinite_values(
    training: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Model-ready outputs must contain finite complete values."""

    assert not training.isna().any().any()
    assert not candidates.isna().any().any()

    for dataframe in (
        training,
        candidates,
    ):
        numeric = dataframe.select_dtypes(
            include="number"
        )

        assert np.isfinite(
            numeric.to_numpy(
                dtype=float
            )
        ).all()


def test_candidate_charging_context_is_preserved(
    candidates: pd.DataFrame,
) -> None:
    """Candidate ranking must retain infrastructure coverage fields."""

    charging_context = {
        "distance_to_nearest_charging_station_m",
        "charging_station_count_within_1000m",
        "charging_station_count_within_2000m",
        "ac_station_count_within_1000m",
        "dc_station_count_within_1000m",
    }

    assert charging_context.issubset(
        candidates.columns
    )

    distance = pd.to_numeric(
        candidates[
            "distance_to_nearest_charging_station_m"
        ],
        errors="coerce",
    )

    assert distance.notna().all()
    assert distance.gt(0).all()


def test_model_dataset_summary_documents_limitations() -> None:
    """The summary must explain class imbalance and leakage control."""

    ensure_file_exists(
        SUMMARY_PATH,
        "Model dataset summary",
    )

    summary = SUMMARY_PATH.read_text(
        encoding="utf-8"
    )

    required_text = (
        "Source grid rows: 7,227",
        "Positive rows: 10",
        "Negative rows: 7,217",
        "Candidate rows: 7,217",
        "extremely imbalanced target",
        "Leakage Controls",
        "explainable",
    )

    for text in required_text:
        assert text in summary
