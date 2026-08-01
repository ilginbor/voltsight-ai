from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCORE_CSV_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.csv"
)

SCORE_GPKG_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_candidate_suitability_scores.gpkg"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_diverse_candidate_shortlist.csv"
)

GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cankaya_diverse_candidate_shortlist.gpkg"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_diverse_candidate_shortlist_summary.md"
)

INPUT_LAYER_NAME = "candidate_suitability_scores"
OUTPUT_LAYER_NAME = "diverse_candidate_shortlist"

PROJECTED_CRS = "EPSG:32636"

DESIRED_CANDIDATE_COUNT = 20
MINIMUM_SPACING_METERS = 1_000.0

MINIMUM_SUITABILITY_SCORE = 60.0
MINIMUM_FEASIBILITY_SCORE = 60.0
MINIMUM_NEED_SCORE = 50.0

REQUIRED_SCORE_COLUMNS = {
    "grid_id",
    "district",
    "city",
    "center_latitude",
    "center_longitude",
    "suitability_rank",
    "suitability_score",
    "suitability_percentile",
    "feasibility_score",
    "need_score",
    "accessibility_score",
    "parking_score",
    "infrastructure_gap_score",
    "technology_gap_score",
    "priority_band",
    "score_explanation",
}

NUMERIC_SCORE_COLUMNS = [
    "center_latitude",
    "center_longitude",
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

CSV_OUTPUT_COLUMNS = [
    "diverse_selection_rank",
    "suitability_rank",
    "grid_id",
    "district",
    "city",
    "center_latitude",
    "center_longitude",
    "suitability_score",
    "suitability_percentile",
    "feasibility_score",
    "need_score",
    "accessibility_score",
    "parking_score",
    "infrastructure_gap_score",
    "technology_gap_score",
    "nearest_selected_grid_id",
    "nearest_selected_candidate_m",
    "priority_band",
    "score_explanation",
]


def create_output_directories() -> None:
    """Create directories required by the shortlist pipeline."""

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
    """Ensure that suitability inputs exist and are not empty."""

    required_paths = (
        SCORE_CSV_INPUT_PATH,
        SCORE_GPKG_INPUT_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Required shortlist input was not found:\n"
                f"{path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Required shortlist input is empty: {path}"
            )


def load_score_attributes() -> pd.DataFrame:
    """Load and validate non-spatial suitability attributes."""

    scores = pd.read_csv(
        SCORE_CSV_INPUT_PATH,
        encoding="utf-8-sig",
    )

    if scores.empty:
        raise ValueError(
            "The suitability score CSV contains no rows."
        )

    missing_columns = (
        REQUIRED_SCORE_COLUMNS
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "The suitability CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if scores["grid_id"].isna().any():
        raise ValueError(
            "The suitability CSV contains missing grid IDs."
        )

    if scores["grid_id"].duplicated().any():
        raise ValueError(
            "The suitability CSV contains duplicate grid IDs."
        )

    for column in NUMERIC_SCORE_COLUMNS:
        scores[column] = pd.to_numeric(
            scores[column],
            errors="coerce",
        )

        if scores[column].isna().any():
            raise ValueError(
                f"Column {column!r} contains invalid values."
            )

        values = scores[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column {column!r} contains non-finite values."
            )

    if not scores[
        "suitability_score"
    ].between(
        0.0,
        100.0,
        inclusive="both",
    ).all():
        raise ValueError(
            "Suitability scores must be between zero and 100."
        )

    return scores


def load_spatial_scores() -> gpd.GeoDataFrame:
    """Load candidate polygons from the suitability GeoPackage."""

    spatial = gpd.read_file(
        SCORE_GPKG_INPUT_PATH,
        layer=INPUT_LAYER_NAME,
    )

    if spatial.empty:
        raise ValueError(
            "The suitability GeoPackage contains no rows."
        )

    if spatial.crs is None:
        raise ValueError(
            "The suitability GeoPackage has no CRS."
        )

    if "grid_id" not in spatial.columns:
        raise ValueError(
            "The suitability GeoPackage does not contain grid_id."
        )

    if spatial["grid_id"].isna().any():
        raise ValueError(
            "The suitability GeoPackage contains missing grid IDs."
        )

    if spatial["grid_id"].duplicated().any():
        raise ValueError(
            "The suitability GeoPackage contains duplicate grid IDs."
        )

    if spatial.geometry.isna().any():
        raise ValueError(
            "The suitability GeoPackage contains missing geometry."
        )

    if spatial.geometry.is_empty.any():
        raise ValueError(
            "The suitability GeoPackage contains empty geometry."
        )

    return spatial[
        [
            "grid_id",
            "geometry",
        ]
    ].copy()


def merge_attributes_and_geometry(
    attributes: pd.DataFrame,
    spatial: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Attach source polygons to suitability attributes."""

    merged = spatial.merge(
        attributes,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(attributes):
        raise ValueError(
            "Not every suitability row matched a grid polygon. "
            f"Expected {len(attributes):,}, found {len(merged):,}."
        )

    return gpd.GeoDataFrame(
        merged,
        geometry="geometry",
        crs=spatial.crs,
    )


def apply_eligibility_filters(
    candidates: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep candidates that satisfy minimum balanced-score rules."""

    eligible = candidates.loc[
        candidates["suitability_score"].ge(
            MINIMUM_SUITABILITY_SCORE
        )
        & candidates["feasibility_score"].ge(
            MINIMUM_FEASIBILITY_SCORE
        )
        & candidates["need_score"].ge(
            MINIMUM_NEED_SCORE
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No candidate satisfies the shortlist thresholds."
        )

    if len(eligible) < DESIRED_CANDIDATE_COUNT:
        raise ValueError(
            "There are not enough eligible candidates. "
            f"Required {DESIRED_CANDIDATE_COUNT}, "
            f"found {len(eligible)}."
        )

    return eligible.sort_values(
        by=[
            "suitability_rank",
            "grid_id",
        ]
    ).reset_index(
        drop=True
    )


def select_spatially_diverse_candidates(
    eligible: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Greedily select high-scoring cells with minimum spacing."""

    projected = eligible.to_crs(
        PROJECTED_CRS
    ).copy()

    projected["_selection_point"] = (
        projected.geometry.centroid
    )

    selected_indices: list[int] = []
    selected_points = []

    for index, row in projected.iterrows():
        point = row["_selection_point"]

        if selected_points:
            nearest_distance = min(
                point.distance(
                    selected_point
                )
                for selected_point in selected_points
            )

            if nearest_distance < MINIMUM_SPACING_METERS:
                continue

        selected_indices.append(
            index
        )

        selected_points.append(
            point
        )

        if (
            len(selected_indices)
            >= DESIRED_CANDIDATE_COUNT
        ):
            break

    if len(selected_indices) < DESIRED_CANDIDATE_COUNT:
        raise ValueError(
            "The spacing rule did not produce enough candidates. "
            f"Required {DESIRED_CANDIDATE_COUNT}, "
            f"selected {len(selected_indices)}."
        )

    selected_projected = projected.loc[
        selected_indices
    ].copy()

    selected_projected[
        "diverse_selection_rank"
    ] = range(
        1,
        len(selected_projected) + 1,
    )

    nearest_grid_ids: list[str] = []
    nearest_distances: list[float] = []

    selected_records = list(
        selected_projected.iterrows()
    )

    for current_index, current_row in selected_records:
        current_point = current_row[
            "_selection_point"
        ]

        nearest_grid_id = None
        nearest_distance = float("inf")

        for other_index, other_row in selected_records:
            if other_index == current_index:
                continue

            distance = current_point.distance(
                other_row["_selection_point"]
            )

            if distance < nearest_distance:
                nearest_distance = float(distance)
                nearest_grid_id = str(
                    other_row["grid_id"]
                )

        if nearest_grid_id is None:
            raise ValueError(
                "Could not calculate nearest selected candidate."
            )

        nearest_grid_ids.append(
            nearest_grid_id
        )

        nearest_distances.append(
            nearest_distance
        )

    selected_projected[
        "nearest_selected_grid_id"
    ] = nearest_grid_ids

    selected_projected[
        "nearest_selected_candidate_m"
    ] = np.round(
        nearest_distances,
        2,
    )

    selected_projected = selected_projected.drop(
        columns="_selection_point"
    )

    return selected_projected.to_crs(
        eligible.crs
    )


def validate_shortlist(
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Validate count, ranking, thresholds and pairwise spacing."""

    if len(shortlist) != DESIRED_CANDIDATE_COUNT:
        raise ValueError(
            "The final shortlist has an incorrect row count."
        )

    if shortlist["grid_id"].duplicated().any():
        raise ValueError(
            "The final shortlist contains duplicate grid IDs."
        )

    expected_selection_ranks = list(
        range(
            1,
            DESIRED_CANDIDATE_COUNT + 1,
        )
    )

    actual_selection_ranks = (
        shortlist[
            "diverse_selection_rank"
        ]
        .astype(int)
        .tolist()
    )

    if actual_selection_ranks != expected_selection_ranks:
        raise ValueError(
            "Diverse selection ranks are not sequential."
        )

    if shortlist[
        "suitability_score"
    ].lt(
        MINIMUM_SUITABILITY_SCORE
    ).any():
        raise ValueError(
            "A selected candidate is below the suitability floor."
        )

    if shortlist[
        "feasibility_score"
    ].lt(
        MINIMUM_FEASIBILITY_SCORE
    ).any():
        raise ValueError(
            "A selected candidate is below the feasibility floor."
        )

    if shortlist[
        "need_score"
    ].lt(
        MINIMUM_NEED_SCORE
    ).any():
        raise ValueError(
            "A selected candidate is below the need floor."
        )

    minimum_distance = float(
        shortlist[
            "nearest_selected_candidate_m"
        ].min()
    )

    if minimum_distance + 0.01 < MINIMUM_SPACING_METERS:
        raise ValueError(
            "The final shortlist violates the spacing rule. "
            f"Minimum distance: {minimum_distance:.2f} metres."
        )

    if shortlist.geometry.isna().any():
        raise ValueError(
            "The final shortlist contains missing geometry."
        )


def save_outputs(
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Save shortlist CSV and GeoPackage outputs."""

    shortlist_csv = pd.DataFrame(
        shortlist.drop(
            columns="geometry"
        )
    )

    shortlist_csv = shortlist_csv[
        CSV_OUTPUT_COLUMNS
    ].copy()

    shortlist_csv.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if GPKG_OUTPUT_PATH.exists():
        GPKG_OUTPUT_PATH.unlink()

    shortlist.to_file(
        GPKG_OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    print("Shortlist CSV saved:")
    print(CSV_OUTPUT_PATH)

    print()

    print("Shortlist GeoPackage saved:")
    print(GPKG_OUTPUT_PATH)


def create_summary(
    all_candidates: gpd.GeoDataFrame,
    eligible: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Create a reproducibility and decision-method summary."""

    failed_suitability = int(
        all_candidates[
            "suitability_score"
        ].lt(
            MINIMUM_SUITABILITY_SCORE
        ).sum()
    )

    failed_feasibility = int(
        all_candidates[
            "feasibility_score"
        ].lt(
            MINIMUM_FEASIBILITY_SCORE
        ).sum()
    )

    failed_need = int(
        all_candidates[
            "need_score"
        ].lt(
            MINIMUM_NEED_SCORE
        ).sum()
    )

    minimum_spacing = float(
        shortlist[
            "nearest_selected_candidate_m"
        ].min()
    )

    maximum_nearest_spacing = float(
        shortlist[
            "nearest_selected_candidate_m"
        ].max()
    )

    worst_original_rank = int(
        shortlist[
            "suitability_rank"
        ].max()
    )

    lowest_selected_score = float(
        shortlist[
            "suitability_score"
        ].min()
    )

    lowest_selected_feasibility = float(
        shortlist[
            "feasibility_score"
        ].min()
    )

    table_columns = [
        "diverse_selection_rank",
        "suitability_rank",
        "grid_id",
        "suitability_score",
        "feasibility_score",
        "need_score",
        "accessibility_score",
        "parking_score",
        "infrastructure_gap_score",
        "nearest_selected_candidate_m",
        "center_latitude",
        "center_longitude",
    ]

    shortlist_table = (
        shortlist[
            table_columns
        ]
        .to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    summary = f"""# Çankaya Diverse Candidate Shortlist

## Source

- Source candidate scores: `{SCORE_CSV_INPUT_PATH.name}`
- Total scored candidates: {len(all_candidates):,}
- Eligible candidates after score thresholds: {len(eligible):,}
- Selected candidates: {len(shortlist):,}
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Eligibility Rules

A grid cell must satisfy all of the following conditions before spatial
selection:

- Suitability score: at least {MINIMUM_SUITABILITY_SCORE:.0f}/100
- Feasibility score: at least {MINIMUM_FEASIBILITY_SCORE:.0f}/100
- Need score: at least {MINIMUM_NEED_SCORE:.0f}/100

Individual criterion failures in the complete score dataset:

- Below suitability threshold: {failed_suitability:,}
- Below feasibility threshold: {failed_feasibility:,}
- Below need threshold: {failed_need:,}

These counts overlap because a candidate can fail more than one rule.

## Spatial Selection

Eligible candidates are ordered by their original suitability rank.
The highest-ranked candidate is selected first. Each following
candidate is accepted only when its grid centroid is at least
{MINIMUM_SPACING_METERS:,.0f} metres from every previously selected
candidate.

This greedy procedure continues until
{DESIRED_CANDIDATE_COUNT} candidates are selected.

## Result Statistics

- Minimum selected-candidate spacing: {minimum_spacing:,.2f} metres
- Maximum nearest-selected spacing: {maximum_nearest_spacing:,.2f} metres
- Worst original suitability rank selected: {worst_original_rank:,}
- Lowest selected suitability score: {lowest_selected_score:.2f}
- Lowest selected feasibility score: {lowest_selected_feasibility:.2f}

## Selected Candidates

{shortlist_table}

## Interpretation

The complete suitability dataset should be used for continuous map
visualization and detailed analysis. This shortlist is intended for
field review, stakeholder evaluation and preliminary feasibility
assessment.

The shortlist does not represent final installation decisions.
Candidate polygons still require on-site validation, electrical-grid
capacity checks, ownership and permit review, traffic analysis and
verified demand data.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print()

    print("Shortlist summary saved:")
    print(SUMMARY_OUTPUT_PATH)


def print_statistics(
    all_candidates: gpd.GeoDataFrame,
    eligible: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Print final shortlist statistics."""

    print("-" * 70)

    print(
        f"Total scored candidates: "
        f"{len(all_candidates):,}"
    )

    print(
        f"Eligible candidate count: "
        f"{len(eligible):,}"
    )

    print(
        f"Selected candidate count: "
        f"{len(shortlist):,}"
    )

    print(
        "Minimum selected spacing: "
        f"{shortlist['nearest_selected_candidate_m'].min():,.2f} m"
    )

    print(
        "Worst original rank selected: "
        f"{int(shortlist['suitability_rank'].max()):,}"
    )

    print(
        "Lowest selected suitability score: "
        f"{shortlist['suitability_score'].min():.2f}"
    )

    print(
        "Lowest selected feasibility score: "
        f"{shortlist['feasibility_score'].min():.2f}"
    )

    print()

    display_columns = [
        "diverse_selection_rank",
        "suitability_rank",
        "grid_id",
        "suitability_score",
        "feasibility_score",
        "need_score",
        "parking_score",
        "nearest_selected_candidate_m",
    ]

    print(
        shortlist[
            display_columns
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Create a balanced and spatially diverse candidate shortlist."""

    print("=" * 70)

    print(
        "VoltSight - Çankaya Diverse Candidate Shortlist"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_files()

    attributes = load_score_attributes()

    spatial = load_spatial_scores()

    all_candidates = merge_attributes_and_geometry(
        attributes,
        spatial,
    )

    eligible = apply_eligibility_filters(
        all_candidates
    )

    shortlist = select_spatially_diverse_candidates(
        eligible
    )

    validate_shortlist(
        shortlist
    )

    save_outputs(
        shortlist
    )

    create_summary(
        all_candidates,
        eligible,
        shortlist,
    )

    print_statistics(
        all_candidates,
        eligible,
        shortlist,
    )

    print("=" * 70)

    print(
        "Diverse candidate shortlist completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
