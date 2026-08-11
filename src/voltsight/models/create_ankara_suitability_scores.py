from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_site_dataset.csv"
)

GRID_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_grid_charging_features.gpkg"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.csv"
)

GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.gpkg"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_candidate_suitability_summary.md"
)

GRID_LAYER_NAME = "grid_charging_features"
OUTPUT_LAYER_NAME = "candidate_suitability_scores"


ACCESSIBILITY_WEIGHTS = {
    "road_proximity_score": 0.45,
    "main_road_presence_score": 0.35,
    "road_density_score": 0.20,
}

PARKING_WEIGHTS = {
    "parking_proximity_score": 0.45,
    "parking_coverage_score": 0.35,
    "parking_area_score": 0.20,
}

INFRASTRUCTURE_GAP_WEIGHTS = {
    "charging_distance_gap_score": 0.75,
    "charging_density_gap_score": 0.25,
}

TECHNOLOGY_GAP_WEIGHTS = {
    "dc_gap_score": 0.60,
    "ac_gap_score": 0.40,
}

FEASIBILITY_WEIGHTS = {
    "accessibility_score": 0.60,
    "parking_score": 0.40,
}

NEED_WEIGHTS = {
    "infrastructure_gap_score": 0.85,
    "technology_gap_score": 0.15,
}


REQUIRED_COLUMNS = {
    "grid_id",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_1000m",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
}

NUMERIC_INPUT_COLUMNS = (
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_1000m",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
)

COMPONENT_COLUMNS = (
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
)


def create_output_directories() -> None:
    """Create output directories."""

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_input_files() -> None:
    """Ensure required candidate and spatial files exist."""

    missing = [
        path
        for path in (
            CANDIDATE_INPUT_PATH,
            GRID_INPUT_PATH,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required Ankara suitability inputs are missing:\n"
            + "\n".join(
                f"- {path}"
                for path in missing
            )
        )


def load_candidate_dataset() -> pd.DataFrame:
    """Load and validate Ankara candidate attributes."""

    candidates = pd.read_csv(
        CANDIDATE_INPUT_PATH,
        dtype={
            "grid_id": str,
        },
    )

    if candidates.empty:
        raise ValueError(
            "Ankara candidate dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if candidates["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate grid IDs were found."
        )

    for column in NUMERIC_INPUT_COLUMNS:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        )

        if candidates[column].isna().any():
            raise ValueError(
                f"Missing/non-numeric values found in {column}."
            )

        values = candidates[column].to_numpy(
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

    print(
        "Loaded Ankara candidates: "
        f"{len(candidates):,}"
    )

    return candidates


def positive_percentile_score(
    values: pd.Series,
) -> pd.Series:
    """Score positive values by percentile while keeping zeros at zero."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    ).astype(float)

    result = pd.Series(
        0.0,
        index=values.index,
        dtype=float,
    )

    positive_mask = numeric.gt(0)

    if positive_mask.any():
        result.loc[positive_mask] = (
            numeric.loc[positive_mask]
            .rank(
                method="average",
                pct=True,
                ascending=True,
            )
            .mul(100.0)
        )

    return result.clip(
        0.0,
        100.0,
    )


def lower_is_better_percentile_score(
    values: pd.Series,
) -> pd.Series:
    """Give higher percentile scores to smaller distance values."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    ).astype(float)

    return (
        numeric.rank(
            method="average",
            pct=True,
            ascending=False,
        )
        .mul(100.0)
        .clip(
            0.0,
            100.0,
        )
    )


def higher_is_better_percentile_score(
    values: pd.Series,
) -> pd.Series:
    """Give higher percentile scores to larger values."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    ).astype(float)

    return (
        numeric.rank(
            method="average",
            pct=True,
            ascending=True,
        )
        .mul(100.0)
        .clip(
            0.0,
            100.0,
        )
    )


def percentile_score(
    values: pd.Series,
    *,
    higher_is_better: bool,
) -> pd.Series:
    """Backward-compatible percentile wrapper for ordered numeric values."""

    if higher_is_better:
        return higher_is_better_percentile_score(
            values
        )

    return lower_is_better_percentile_score(
        values
    )


def distance_gap_score(
    values: pd.Series,
) -> pd.Series:
    """Score larger distances as larger infrastructure gaps."""

    return higher_is_better_percentile_score(
        values
    )


def low_count_gap_score(
    values: pd.Series,
) -> pd.Series:
    """Score infrastructure scarcity using a decreasing count curve."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    ).astype(float)

    numeric = numeric.clip(
        lower=0.0
    )

    return (
        100.0
        / (1.0 + numeric)
    ).clip(
        0.0,
        100.0,
    )


def absence_score(
    values: pd.Series,
) -> pd.Series:
    """Return 100 where infrastructure is absent, otherwise 0."""

    numeric = pd.to_numeric(
        values,
        errors="raise",
    )

    return (
        numeric.le(0)
        .astype(float)
        * 100.0
    )


def weighted_score(
    dataframe: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Calculate a weighted 0-100 score."""

    if not np.isclose(
        sum(weights.values()),
        1.0,
    ):
        raise ValueError(
            "Weights must sum to 1."
        )

    score = pd.Series(
        0.0,
        index=dataframe.index,
        dtype=float,
    )

    for column, weight in weights.items():
        score = (
            score
            + dataframe[column] * weight
        )

    return score.clip(
        0.0,
        100.0,
    )


def assign_priority_band(
    percentile: pd.Series,
) -> pd.Series:
    """Assign relative A-E priority bands."""

    result = pd.Series(
        "E",
        index=percentile.index,
        dtype=object,
    )

    result.loc[
        percentile >= 99
    ] = "A"

    result.loc[
        (percentile >= 95)
        & (percentile < 99)
    ] = "B"

    result.loc[
        (percentile >= 80)
        & (percentile < 95)
    ] = "C"

    result.loc[
        (percentile >= 50)
        & (percentile < 80)
    ] = "D"

    return result


def build_explanation(
    row: pd.Series,
) -> str:
    """Create a concise explanation for one candidate."""

    strengths: list[str] = []
    needs: list[str] = []

    if row["accessibility_score"] >= 70:
        strengths.append(
            "strong road accessibility"
        )

    if row["parking_score"] >= 70:
        strengths.append(
            "strong parking feasibility"
        )

    if row["infrastructure_gap_score"] >= 70:
        needs.append(
            "large charging infrastructure gap"
        )

    if row["technology_gap_score"] >= 70:
        needs.append(
            "AC/DC technology gap"
        )

    if not strengths:
        strengths.append(
            "moderate site feasibility"
        )

    if not needs:
        needs.append(
            "moderate infrastructure need"
        )

    return (
        "; ".join(strengths)
        + "; "
        + "; ".join(needs)
    )


def create_suitability_scores(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Create explainable Ankara candidate suitability scores."""

    result = candidates.copy()

    result[
        "road_proximity_score"
    ] = lower_is_better_percentile_score(
        result["distance_to_main_road_m"]
    )

    result[
        "main_road_presence_score"
    ] = (
        result["main_road_segment_count"]
        .gt(0)
        .astype(float)
        * 100.0
    )

    result[
        "road_density_score"
    ] = positive_percentile_score(
        result["road_density_km_per_km2"]
    )

    result[
        "parking_proximity_score"
    ] = lower_is_better_percentile_score(
        result[
            "distance_to_nearest_parking_m"
        ]
    )

    result[
        "parking_coverage_score"
    ] = positive_percentile_score(
        result[
            "parking_count_within_1000m"
        ]
    )

    result[
        "parking_area_score"
    ] = positive_percentile_score(
        result["parking_area_m2"]
    )

    result[
        "charging_distance_gap_score"
    ] = distance_gap_score(
        result[
            "distance_to_nearest_charging_station_m"
        ]
    )

    result[
        "charging_density_gap_score"
    ] = low_count_gap_score(
        result[
            "charging_station_count_within_2000m"
        ]
    )

    result[
        "ac_gap_score"
    ] = absence_score(
        result[
            "ac_station_count_within_1000m"
        ]
    )

    result[
        "dc_gap_score"
    ] = absence_score(
        result[
            "dc_station_count_within_1000m"
        ]
    )

    result[
        "accessibility_score"
    ] = weighted_score(
        result,
        ACCESSIBILITY_WEIGHTS,
    )

    result[
        "parking_score"
    ] = weighted_score(
        result,
        PARKING_WEIGHTS,
    )

    result[
        "infrastructure_gap_score"
    ] = weighted_score(
        result,
        INFRASTRUCTURE_GAP_WEIGHTS,
    )

    result[
        "technology_gap_score"
    ] = weighted_score(
        result,
        TECHNOLOGY_GAP_WEIGHTS,
    )

    result[
        "feasibility_score"
    ] = weighted_score(
        result,
        FEASIBILITY_WEIGHTS,
    )

    result[
        "need_score"
    ] = weighted_score(
        result,
        NEED_WEIGHTS,
    )

    result[
        "suitability_score"
    ] = np.sqrt(
        result["feasibility_score"]
        * result["need_score"]
    )

    result[
        "suitability_percentile"
    ] = (
        result[
            "suitability_score"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 100.0
    )

    result[
        "suitability_rank"
    ] = (
        result[
            "suitability_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result[
        "priority_band"
    ] = assign_priority_band(
        result[
            "suitability_percentile"
        ]
    )

    for column in (
        *COMPONENT_COLUMNS,
        "suitability_percentile",
    ):
        result[column] = (
            result[column]
            .clip(
                0.0,
                100.0,
            )
            .round(4)
        )

    result[
        "score_explanation"
    ] = result.apply(
        build_explanation,
        axis=1,
    )

    return result


def validate_scores(
    scores: pd.DataFrame,
) -> None:
    """Validate suitability outputs."""

    if scores.empty:
        raise ValueError(
            "Suitability score dataset is empty."
        )

    if scores["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate suitability grid IDs were found."
        )

    for column in (
        *COMPONENT_COLUMNS,
        "suitability_percentile",
    ):
        values = scores[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite score in {column}."
            )

        if (
            (values < 0)
            | (values > 100)
        ).any():
            raise ValueError(
                f"Out-of-range score in {column}."
            )

    expected_ranks = set(
        range(
            1,
            int(
                scores[
                    "suitability_rank"
                ].max()
            )
            + 1,
        )
    )

    actual_ranks = set(
        scores[
            "suitability_rank"
        ].astype(int)
    )

    if not actual_ranks.issubset(
        expected_ranks
    ):
        raise ValueError(
            "Invalid suitability ranks were found."
        )

    if not set(
        scores[
            "priority_band"
        ].unique()
    ).issubset(
        {
            "A",
            "B",
            "C",
            "D",
            "E",
        }
    ):
        raise ValueError(
            "Invalid priority bands were found."
        )

    print(
        "Ankara suitability validation "
        "completed successfully."
    )


def create_spatial_output(
    scores: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join candidate scores back to Ankara grid geometry."""

    grid = gpd.read_file(
        GRID_INPUT_PATH,
        layer=GRID_LAYER_NAME,
    )

    if grid.empty:
        raise ValueError(
            "Ankara spatial grid is empty."
        )

    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate grid IDs found in spatial data."
        )

    candidate_ids = set(
        scores["grid_id"]
    )

    spatial = grid.loc[
        grid["grid_id"].isin(
            candidate_ids
        ),
        [
            "grid_id",
            "geometry",
        ],
    ].copy()

    spatial = spatial.merge(
        scores,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(spatial) != len(scores):
        raise ValueError(
            "Spatial suitability row count does "
            "not match candidate score count."
        )

    return gpd.GeoDataFrame(
        spatial,
        geometry="geometry",
        crs=grid.crs,
    )


def save_outputs(
    scores: pd.DataFrame,
    spatial: gpd.GeoDataFrame,
) -> None:
    """Save CSV and GeoPackage outputs."""

    for path in (
        CSV_OUTPUT_PATH,
        GPKG_OUTPUT_PATH,
    ):
        if path.exists():
            path.unlink()

    scores.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    spatial.to_file(
        GPKG_OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    print(
        f"Suitability CSV saved: {CSV_OUTPUT_PATH}"
    )

    print(
        f"Suitability GeoPackage saved: {GPKG_OUTPUT_PATH}"
    )


def create_summary(
    scores: pd.DataFrame,
) -> None:
    """Create Ankara scoring summary."""

    band_counts = (
        scores[
            "priority_band"
        ]
        .value_counts()
        .reindex(
            [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
            fill_value=0,
        )
    )

    top = (
        scores
        .sort_values(
            [
                "suitability_rank",
                "grid_id",
            ]
        )
        .iloc[0]
    )

    summary = f"""# Ankara Candidate Suitability Summary

## Candidates

- Candidate grid cells: {len(scores):,}
- Median suitability score: {scores["suitability_score"].median():.2f}
- Maximum suitability score: {scores["suitability_score"].max():.2f}
- Minimum suitability score: {scores["suitability_score"].min():.2f}
- Top candidate: `{top["grid_id"]}`
- Top candidate score: {top["suitability_score"]:.4f}

## Priority Bands

- A: {int(band_counts["A"]):,}
- B: {int(band_counts["B"]):,}
- C: {int(band_counts["C"]):,}
- D: {int(band_counts["D"]):,}
- E: {int(band_counts["E"]):,}

## Scoring Model

### Accessibility

- Main-road proximity: 45%
- Main-road presence: 35%
- Road density: 20%

### Parking

- Nearest-parking proximity: 45%
- Parking within 1 km: 35%
- Local parking area: 20%

### Infrastructure Gap

- Distance to nearest charging station: 75%
- Charging-station scarcity within 2 km: 25%

### Technology Gap

- DC absence within 1 km: 60%
- AC absence within 1 km: 40%

### Composite Scores

- Feasibility = 60% accessibility + 40% parking
- Need = 85% infrastructure gap + 15% technology gap
- Suitability = geometric mean of feasibility and need

## Interpretation

The score is an explainable decision-support ranking rather than a
probability that a charging station should be constructed.

Percentile transformations are calculated over Ankara candidate cells,
so scores are relative to the province-wide candidate distribution.
Positive-only road-density and parking transformations keep true zero
values at zero instead of assigning artificial percentile credit.
Charging-station count scarcity uses a deterministic decreasing curve.

## Outputs

- `data/processed/ankara_candidate_suitability_scores.csv`
- `data/processed/ankara_candidate_suitability_scores.gpkg`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def print_statistics(
    scores: pd.DataFrame,
) -> None:
    """Print key suitability statistics."""

    band_counts = (
        scores[
            "priority_band"
        ]
        .value_counts()
    )

    top = scores.sort_values(
        [
            "suitability_rank",
            "grid_id",
        ]
    ).iloc[0]

    print("-" * 70)

    print(
        "Candidate count: "
        f"{len(scores):,}"
    )

    print(
        "Median suitability: "
        f"{scores['suitability_score'].median():.2f}"
    )

    print(
        "Maximum suitability: "
        f"{scores['suitability_score'].max():.2f}"
    )

    print(
        "Minimum suitability: "
        f"{scores['suitability_score'].min():.2f}"
    )

    for band in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        print(
            f"Priority band {band}: "
            f"{int(band_counts.get(band, 0)):,}"
        )

    print(
        "Top candidate: "
        f"{top['grid_id']}"
    )

    print(
        "Top candidate score: "
        f"{top['suitability_score']:.4f}"
    )


def main() -> None:
    """Run Ankara explainable suitability scoring."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Candidate Suitability Scoring"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_files()

    candidates = load_candidate_dataset()

    scores = create_suitability_scores(
        candidates
    )

    validate_scores(
        scores
    )

    spatial = create_spatial_output(
        scores
    )

    save_outputs(
        scores,
        spatial,
    )

    create_summary(
        scores
    )

    print_statistics(
        scores
    )

    print("=" * 70)

    print(
        "Ankara suitability scoring "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
