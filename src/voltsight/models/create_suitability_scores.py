from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_site_dataset.csv"
)

GRID_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_grid_charging_features.gpkg"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.csv"
)

GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.gpkg"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_candidate_suitability_summary.md"
)

GRID_LAYER_NAME = "grid_charging_features"
OUTPUT_LAYER_NAME = "candidate_suitability_scores"

REQUIRED_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "center_longitude",
    "center_latitude",
    "main_road_length_m",
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

NUMERIC_INPUT_COLUMNS = [
    "center_longitude",
    "center_latitude",
    "main_road_length_m",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
    "parking_area_m2",
    "distance_to_nearest_parking_m",
    "parking_count_within_1000m",
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
]

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

SCORE_COMPONENT_COLUMNS = [
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

COMPONENT_LABELS = {
    "accessibility_score": "road accessibility",
    "parking_score": "parking suitability",
    "infrastructure_gap_score": "charging infrastructure gap",
    "technology_gap_score": "AC/DC technology gap",
}


def create_output_directories() -> None:
    """Create directories required by the scoring pipeline."""

    directories = {
        CSV_OUTPUT_PATH.parent,
        GPKG_OUTPUT_PATH.parent,
        SUMMARY_OUTPUT_PATH.parent,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_input_files() -> None:
    """Ensure that required candidate and grid files exist."""

    required_paths = (
        CANDIDATE_INPUT_PATH,
        GRID_INPUT_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Required suitability input was not found:\n"
                f"{path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Required suitability input is empty: {path}"
            )


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
                f"Column {column!r} contains missing or "
                "non-numeric values."
            )

        values = result[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column {column!r} contains non-finite values."
            )

    return result


def load_candidate_dataset() -> pd.DataFrame:
    """Load and validate candidate grid-cell features."""

    candidates = pd.read_csv(
        CANDIDATE_INPUT_PATH,
        encoding="utf-8-sig",
    )

    if candidates.empty:
        raise ValueError(
            "The candidate-site dataset contains no rows."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "The candidate dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if candidates["grid_id"].isna().any():
        raise ValueError(
            "The candidate dataset contains missing grid IDs."
        )

    if candidates["grid_id"].duplicated().any():
        raise ValueError(
            "The candidate dataset contains duplicate grid IDs."
        )

    candidates = convert_numeric_columns(
        candidates,
        NUMERIC_INPUT_COLUMNS,
    )

    non_negative_columns = [
        column
        for column in NUMERIC_INPUT_COLUMNS
        if column not in {
            "center_longitude",
            "center_latitude",
        }
    ]

    for column in non_negative_columns:
        if candidates[column].lt(0).any():
            raise ValueError(
                f"Column {column!r} contains negative values."
            )

    return candidates


def positive_percentile_score(
    series: pd.Series,
) -> pd.Series:
    """Score positive values by percentile while keeping zeros at zero."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    result = pd.Series(
        0.0,
        index=series.index,
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
        lower=0.0,
        upper=100.0,
    )


def lower_is_better_percentile_score(
    series: pd.Series,
) -> pd.Series:
    """Give higher percentile scores to smaller distance values."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        numeric.rank(
            method="average",
            pct=True,
            ascending=False,
        )
        .mul(100.0)
        .clip(
            lower=0.0,
            upper=100.0,
        )
    )


def higher_is_better_percentile_score(
    series: pd.Series,
) -> pd.Series:
    """Give higher percentile scores to larger values."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        numeric.rank(
            method="average",
            pct=True,
            ascending=True,
        )
        .mul(100.0)
        .clip(
            lower=0.0,
            upper=100.0,
        )
    )


def low_count_gap_score(
    series: pd.Series,
) -> pd.Series:
    """Score infrastructure scarcity using a decreasing count curve."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    numeric = numeric.clip(
        lower=0.0
    )

    return (
        100.0
        / (1.0 + numeric)
    ).clip(
        lower=0.0,
        upper=100.0,
    )


def absence_score(
    series: pd.Series,
) -> pd.Series:
    """Return 100 when a nearby technology is absent and zero otherwise."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    return (
        numeric.le(0)
        .astype(float)
        .mul(100.0)
    )


def weighted_score(
    dataframe: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Calculate a weighted arithmetic score."""

    weight_sum = sum(
        weights.values()
    )

    if not np.isclose(
        weight_sum,
        1.0,
    ):
        raise ValueError(
            "Score weights must sum to one. "
            f"Found {weight_sum:.6f}."
        )

    result = pd.Series(
        0.0,
        index=dataframe.index,
        dtype=float,
    )

    for column, weight in weights.items():
        if column not in dataframe.columns:
            raise ValueError(
                f"Score component {column!r} is missing."
            )

        result = (
            result
            + dataframe[column].astype(float) * weight
        )

    return result.clip(
        lower=0.0,
        upper=100.0,
    )


def assign_priority_band(
    percentile: pd.Series,
) -> pd.Series:
    """Assign relative priority bands from suitability percentiles."""

    conditions = [
        percentile.ge(99.0),
        percentile.ge(95.0),
        percentile.ge(80.0),
        percentile.ge(50.0),
    ]

    choices = [
        "A - Highest priority",
        "B - High priority",
        "C - Medium priority",
        "D - Lower priority",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="E - Lowest priority",
        ),
        index=percentile.index,
        dtype="object",
    )


def build_explanation(
    row: pd.Series,
) -> str:
    """Create a compact explanation of strengths and bottlenecks."""

    component_values = {
        column: float(row[column])
        for column in COMPONENT_LABELS
    }

    ordered = sorted(
        component_values.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    strongest_column, strongest_value = ordered[0]
    second_column, second_value = ordered[1]
    weakest_column, weakest_value = ordered[-1]

    return (
        f"Feasibility {row['feasibility_score']:.1f}/100; "
        f"need {row['need_score']:.1f}/100. "
        f"Strengths: "
        f"{COMPONENT_LABELS[strongest_column]} "
        f"({strongest_value:.1f}), "
        f"{COMPONENT_LABELS[second_column]} "
        f"({second_value:.1f}). "
        f"Main constraint: "
        f"{COMPONENT_LABELS[weakest_column]} "
        f"({weakest_value:.1f})."
    )


def create_suitability_scores(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate explainable component and overall suitability scores."""

    scored = candidates.copy()

    scored["road_proximity_score"] = (
        lower_is_better_percentile_score(
            scored["distance_to_main_road_m"]
        )
    )

    scored["main_road_presence_score"] = (
        positive_percentile_score(
            scored["main_road_length_m"]
        )
    )

    scored["road_density_score"] = (
        positive_percentile_score(
            scored["road_density_km_per_km2"]
        )
    )

    scored["parking_proximity_score"] = (
        lower_is_better_percentile_score(
            scored["distance_to_nearest_parking_m"]
        )
    )

    scored["parking_coverage_score"] = (
        positive_percentile_score(
            scored["parking_count_within_1000m"]
        )
    )

    scored["parking_area_score"] = (
        positive_percentile_score(
            scored["parking_area_m2"]
        )
    )

    scored["charging_distance_gap_score"] = (
        higher_is_better_percentile_score(
            scored[
                "distance_to_nearest_charging_station_m"
            ]
        )
    )

    scored["charging_density_gap_score"] = (
        low_count_gap_score(
            scored[
                "charging_station_count_within_2000m"
            ]
        )
    )

    scored["ac_gap_score"] = absence_score(
        scored[
            "ac_station_count_within_1000m"
        ]
    )

    scored["dc_gap_score"] = absence_score(
        scored[
            "dc_station_count_within_1000m"
        ]
    )

    scored["accessibility_score"] = weighted_score(
        scored,
        ACCESSIBILITY_WEIGHTS,
    )

    scored["parking_score"] = weighted_score(
        scored,
        PARKING_WEIGHTS,
    )

    scored["infrastructure_gap_score"] = weighted_score(
        scored,
        INFRASTRUCTURE_GAP_WEIGHTS,
    )

    scored["technology_gap_score"] = weighted_score(
        scored,
        TECHNOLOGY_GAP_WEIGHTS,
    )

    scored["feasibility_score"] = weighted_score(
        scored,
        FEASIBILITY_WEIGHTS,
    )

    scored["need_score"] = weighted_score(
        scored,
        NEED_WEIGHTS,
    )

    scored["suitability_score"] = np.sqrt(
        scored["feasibility_score"]
        * scored["need_score"]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    scored["suitability_rank"] = (
        scored["suitability_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    row_count = len(scored)

    if row_count == 1:
        scored["suitability_percentile"] = 100.0
    else:
        scored["suitability_percentile"] = (
            (
                row_count
                - scored["suitability_rank"]
            )
            / (row_count - 1)
            * 100.0
        )

    scored["priority_band"] = assign_priority_band(
        scored["suitability_percentile"]
    )

    scored["score_explanation"] = scored.apply(
        build_explanation,
        axis=1,
    )

    numeric_score_columns = (
        SCORE_COMPONENT_COLUMNS
        + [
            "suitability_rank",
        ]
    )

    for column in numeric_score_columns:
        if column == "suitability_rank":
            continue

        scored[column] = (
            scored[column]
            .astype(float)
            .round(4)
        )

    return scored.sort_values(
        by=[
            "suitability_rank",
            "grid_id",
        ]
    ).reset_index(
        drop=True
    )


def validate_scores(
    source: pd.DataFrame,
    scored: pd.DataFrame,
) -> None:
    """Validate score ranges, ranking and candidate preservation."""

    if len(scored) != len(source):
        raise ValueError(
            "Scored row count does not match candidate input."
        )

    if set(scored["grid_id"]) != set(
        source["grid_id"]
    ):
        raise ValueError(
            "Scored grid IDs do not match candidate input."
        )

    if scored["grid_id"].duplicated().any():
        raise ValueError(
            "Scored output contains duplicate grid IDs."
        )

    if scored.isna().any().any():
        missing = scored.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            "Scored output contains missing values:\n"
            f"{missing.to_string()}"
        )

    for column in SCORE_COMPONENT_COLUMNS:
        values = pd.to_numeric(
            scored[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Score column {column!r} contains invalid values."
            )

        if not values.between(
            0.0,
            100.0,
            inclusive="both",
        ).all():
            raise ValueError(
                f"Score column {column!r} is outside 0-100."
            )

    expected_ranks = set(
        range(
            1,
            len(scored) + 1,
        )
    )

    actual_ranks = set(
        scored["suitability_rank"]
        .astype(int)
    )

    if actual_ranks != expected_ranks:
        raise ValueError(
            "Suitability ranks are not complete and unique."
        )


def create_spatial_output(
    scored: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach suitability scores to original grid polygons."""

    grid = gpd.read_file(
        GRID_INPUT_PATH,
        layer=GRID_LAYER_NAME,
    )

    if grid.empty:
        raise ValueError(
            "The grid GeoPackage contains no rows."
        )

    if grid.crs is None:
        raise ValueError(
            "The grid GeoPackage has no CRS."
        )

    if "grid_id" not in grid.columns:
        raise ValueError(
            "The grid GeoPackage does not contain grid_id."
        )

    score_columns = [
        "grid_id",
        *SCORE_COMPONENT_COLUMNS,
        "suitability_rank",
        "priority_band",
        "score_explanation",
    ]

    spatial = grid[
        [
            "grid_id",
            "geometry",
        ]
    ].merge(
        scored[score_columns],
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(spatial) != len(scored):
        raise ValueError(
            "Not every candidate score matched a grid polygon."
        )

    return gpd.GeoDataFrame(
        spatial,
        geometry="geometry",
        crs=grid.crs,
    )


def save_outputs(
    scored: pd.DataFrame,
    spatial: gpd.GeoDataFrame,
) -> None:
    """Save CSV and polygon GeoPackage outputs."""

    scored.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if GPKG_OUTPUT_PATH.exists():
        GPKG_OUTPUT_PATH.unlink()

    spatial.to_file(
        GPKG_OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    print(
        "Suitability CSV saved:"
    )
    print(CSV_OUTPUT_PATH)

    print()

    print(
        "Suitability GeoPackage saved:"
    )
    print(GPKG_OUTPUT_PATH)


def create_summary(
    scored: pd.DataFrame,
) -> None:
    """Create a reproducibility and score-method summary."""

    top_columns = [
        "suitability_rank",
        "grid_id",
        "district",
        "suitability_score",
        "feasibility_score",
        "need_score",
        "accessibility_score",
        "parking_score",
        "infrastructure_gap_score",
        "technology_gap_score",
        "priority_band",
    ]

    top_twenty = (
        scored[top_columns]
        .head(20)
        .to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    band_counts = (
        scored["priority_band"]
        .value_counts()
        .sort_index()
    )

    band_lines = "\n".join(
        f"- {band}: {int(count):,}"
        for band, count in band_counts.items()
    )

    score_description = (
        scored[
            [
                "accessibility_score",
                "parking_score",
                "infrastructure_gap_score",
                "technology_gap_score",
                "feasibility_score",
                "need_score",
                "suitability_score",
            ]
        ]
        .describe()
        .T[
            [
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .round(2)
        .to_markdown()
    )

    summary = f"""# Çankaya Candidate Suitability Score Summary

## Source

- Candidate source: `{CANDIDATE_INPUT_PATH.name}`
- Candidate rows: {len(scored):,}
- Generated at: {datetime.now(timezone.utc).isoformat()}
- Output CSV: `data/processed/{CSV_OUTPUT_PATH.name}`
- Output GeoPackage: `data/processed/{GPKG_OUTPUT_PATH.name}`

## Method

This is an explainable decision-support score, not a trained machine
learning prediction.

Feature percentiles are calculated relative to the current Çankaya
candidate grid population. Zero-inflated road and parking quantities
retain a score of zero when the underlying quantity is zero.

### Accessibility Score

- 45% proximity to a main road
- 35% main-road length inside the grid cell
- 20% road density

### Parking Score

- 45% proximity to the nearest mapped parking feature
- 35% parking count within 1,000 metres
- 20% mapped parking area inside the grid cell

### Infrastructure Gap Score

- 75% distance from the nearest existing charging station
- 25% scarcity of charging stations within 2,000 metres

### Technology Gap Score

- 60% absence of a mapped DC station within 1,000 metres
- 40% absence of a mapped AC station within 1,000 metres

### Combined Scores

- Feasibility = 60% accessibility + 40% parking
- Need = 85% infrastructure gap + 15% technology gap
- Suitability = square root of feasibility multiplied by need

The geometric combination prevents remote cells with a large
infrastructure gap but very poor road and parking feasibility from
automatically receiving the highest rankings.

## Priority Bands

{band_lines}

Bands are relative rankings:

- A: top 1%
- B: 95th to below 99th percentile
- C: 80th to below 95th percentile
- D: 50th to below 80th percentile
- E: below the 50th percentile

## Score Distribution

{score_description}

## Top 20 Candidate Grid Cells

{top_twenty}

## Important Limitations

The score reflects mapped OpenStreetMap, EPDK, road and parking
coverage. Missing map objects do not necessarily mean that real-world
infrastructure is absent.

The weights are explicit expert assumptions derived from the observed
feature distributions and correlations. They should later be tested
with stakeholder feedback, utilization data, population, traffic,
electric-grid capacity and verified installation outcomes.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()

    print(
        "Suitability summary saved:"
    )
    print(SUMMARY_OUTPUT_PATH)


def print_statistics(
    scored: pd.DataFrame,
) -> None:
    """Print final scoring statistics."""

    print("-" * 70)

    print(
        f"Scored candidate count: {len(scored):,}"
    )

    print(
        "Median suitability score: "
        f"{scored['suitability_score'].median():.2f}"
    )

    print(
        "Maximum suitability score: "
        f"{scored['suitability_score'].max():.2f}"
    )

    print(
        "Minimum suitability score: "
        f"{scored['suitability_score'].min():.2f}"
    )

    print()

    print("Priority band counts:")

    band_counts = (
        scored["priority_band"]
        .value_counts()
        .sort_index()
    )

    for band, count in band_counts.items():
        print(
            f"  {band}: {int(count):,}"
        )

    print()

    print("Top 10 candidates:")

    print(
        scored[
            [
                "suitability_rank",
                "grid_id",
                "suitability_score",
                "feasibility_score",
                "need_score",
                "priority_band",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


def main() -> None:
    """Create explainable suitability scores for candidate grid cells."""

    print("=" * 70)

    print(
        "VoltSight - Çankaya Candidate Suitability Scoring"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_files()

    candidates = load_candidate_dataset()

    scored = create_suitability_scores(
        candidates
    )

    validate_scores(
        candidates,
        scored,
    )

    spatial = create_spatial_output(
        scored
    )

    save_outputs(
        scored,
        spatial,
    )

    create_summary(
        scored
    )

    print_statistics(
        scored
    )

    print("=" * 70)

    print(
        "Candidate suitability scoring completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
