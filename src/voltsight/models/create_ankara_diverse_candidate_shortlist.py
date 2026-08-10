from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCORE_GPKG_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.gpkg"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.csv"
)

GPKG_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.gpkg"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_diverse_candidate_shortlist_summary.md"
)

INPUT_LAYER_NAME = (
    "candidate_suitability_scores"
)

OUTPUT_LAYER_NAME = (
    "diverse_candidate_shortlist"
)

PROJECTED_CRS = "EPSG:32636"

DESIRED_CANDIDATE_COUNT = 20

MINIMUM_SPACING_METERS = 5_000.0

MINIMUM_SUITABILITY_SCORE = 60.0
MINIMUM_FEASIBILITY_SCORE = 60.0
MINIMUM_NEED_SCORE = 50.0


REQUIRED_COLUMNS = {
    "grid_id",
    "suitability_score",
    "suitability_rank",
    "suitability_percentile",
    "priority_band",
    "feasibility_score",
    "need_score",
    "accessibility_score",
    "parking_score",
    "infrastructure_gap_score",
    "technology_gap_score",
    "score_explanation",
    "geometry",
}

NUMERIC_SCORE_COLUMNS = (
    "suitability_score",
    "suitability_rank",
    "suitability_percentile",
    "feasibility_score",
    "need_score",
    "accessibility_score",
    "parking_score",
    "infrastructure_gap_score",
    "technology_gap_score",
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


def validate_input_file() -> None:
    """Ensure the Ankara suitability GeoPackage exists."""

    if not SCORE_GPKG_INPUT_PATH.exists():
        raise FileNotFoundError(
            "Ankara suitability GeoPackage was not found:\n"
            f"{SCORE_GPKG_INPUT_PATH}"
        )


def load_spatial_scores() -> gpd.GeoDataFrame:
    """Load and validate scored Ankara candidates."""

    scores = gpd.read_file(
        SCORE_GPKG_INPUT_PATH,
        layer=INPUT_LAYER_NAME,
    )

    if scores.empty:
        raise ValueError(
            "The Ankara suitability dataset is empty."
        )

    if scores.crs is None:
        raise ValueError(
            "The Ankara suitability dataset has no CRS."
        )

    if str(scores.crs) != PROJECTED_CRS:
        scores = scores.to_crs(
            PROJECTED_CRS
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "Suitability dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if scores["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate scored grid IDs were found."
        )

    if scores.geometry.isna().any():
        raise ValueError(
            "Missing candidate geometries were found."
        )

    if scores.geometry.is_empty.any():
        raise ValueError(
            "Empty candidate geometries were found."
        )

    if not scores.geometry.is_valid.all():
        raise ValueError(
            "Invalid candidate geometries were found."
        )

    for column in NUMERIC_SCORE_COLUMNS:
        scores[column] = pd.to_numeric(
            scores[column],
            errors="coerce",
        )

        if scores[column].isna().any():
            raise ValueError(
                f"Missing/non-numeric values found in {column}."
            )

        values = scores[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    print(
        "Loaded scored candidates: "
        f"{len(scores):,}"
    )

    return scores


def apply_eligibility_filters(
    scores: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Apply minimum suitability, feasibility and need thresholds."""

    eligible = scores.loc[
        (
            scores["suitability_score"]
            >= MINIMUM_SUITABILITY_SCORE
        )
        & (
            scores["feasibility_score"]
            >= MINIMUM_FEASIBILITY_SCORE
        )
        & (
            scores["need_score"]
            >= MINIMUM_NEED_SCORE
        )
    ].copy()

    eligible = eligible.sort_values(
        by=[
            "suitability_rank",
            "grid_id",
        ],
        ascending=[
            True,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if eligible.empty:
        raise ValueError(
            "No Ankara candidates satisfy "
            "the shortlist eligibility thresholds."
        )

    print(
        "Eligible candidates: "
        f"{len(eligible):,}"
    )

    return eligible


def create_candidate_points(
    candidates: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Create representative points for spacing calculations."""

    points = candidates.copy()

    points["geometry"] = (
        points.geometry
        .representative_point()
    )

    return gpd.GeoDataFrame(
        points,
        geometry="geometry",
        crs=candidates.crs,
    )


def select_spatially_diverse_candidates(
    eligible: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Greedily select the highest-ranked candidates while enforcing
    a minimum separation distance.
    """

    points = create_candidate_points(
        eligible
    )

    selected_indices: list[int] = []
    selected_points = []

    for index, row in points.iterrows():
        point = row.geometry

        if not selected_points:
            selected_indices.append(
                index
            )

            selected_points.append(
                point
            )

        else:
            nearest_distance = min(
                point.distance(
                    selected_point
                )
                for selected_point
                in selected_points
            )

            if (
                nearest_distance
                >= MINIMUM_SPACING_METERS
            ):
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
        raise RuntimeError(
            "The requested Ankara shortlist could not "
            "be filled with the configured spacing. "
            f"Requested {DESIRED_CANDIDATE_COUNT}, "
            f"selected {len(selected_indices)}."
        )

    shortlist = eligible.loc[
        selected_indices
    ].copy()

    shortlist = shortlist.reset_index(
        drop=True
    )

    shortlist.insert(
        0,
        "diverse_selection_rank",
        np.arange(
            1,
            len(shortlist) + 1,
            dtype=int,
        ),
    )

    return gpd.GeoDataFrame(
        shortlist,
        geometry="geometry",
        crs=eligible.crs,
    )


def add_spacing_diagnostics(
    shortlist: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add nearest selected-candidate distance diagnostics."""

    result = shortlist.copy()

    points = result.geometry.representative_point()

    nearest_ids: list[str | None] = []
    nearest_distances: list[float] = []

    for index, point in enumerate(points):
        other_indices = [
            other_index
            for other_index
            in range(len(points))
            if other_index != index
        ]

        if not other_indices:
            nearest_ids.append(
                None
            )

            nearest_distances.append(
                np.nan
            )

            continue

        distances = [
            (
                other_index,
                float(
                    point.distance(
                        points.iloc[
                            other_index
                        ]
                    )
                ),
            )
            for other_index
            in other_indices
        ]

        nearest_index, nearest_distance = min(
            distances,
            key=lambda item: item[1],
        )

        nearest_ids.append(
            str(
                result.iloc[
                    nearest_index
                ]["grid_id"]
            )
        )

        nearest_distances.append(
            nearest_distance
        )

    result[
        "nearest_selected_grid_id"
    ] = nearest_ids

    result[
        "nearest_selected_candidate_m"
    ] = np.round(
        nearest_distances,
        2,
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=shortlist.crs,
    )


def add_location_columns(
    shortlist: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add representative-point longitude and latitude."""

    result = shortlist.copy()

    points = gpd.GeoDataFrame(
        {
            "grid_id": result[
                "grid_id"
            ].astype(str),
        },
        geometry=(
            result.geometry
            .representative_point()
        ),
        crs=result.crs,
    ).to_crs(
        epsg=4326
    )

    result[
        "center_longitude"
    ] = (
        points.geometry.x
        .round(7)
        .to_numpy()
    )

    result[
        "center_latitude"
    ] = (
        points.geometry.y
        .round(7)
        .to_numpy()
    )

    return result


def validate_shortlist(
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Validate final Ankara shortlist."""

    if len(shortlist) != DESIRED_CANDIDATE_COUNT:
        raise ValueError(
            "Unexpected shortlist row count."
        )

    if shortlist["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate shortlist grid IDs were found."
        )

    if shortlist.isna().drop(
        columns=[
            "nearest_selected_grid_id",
        ],
        errors="ignore",
    ).any().any():
        raise ValueError(
            "Unexpected missing shortlist values were found."
        )

    if (
        shortlist["suitability_score"]
        < MINIMUM_SUITABILITY_SCORE
    ).any():
        raise ValueError(
            "A selected candidate violates "
            "the suitability threshold."
        )

    if (
        shortlist["feasibility_score"]
        < MINIMUM_FEASIBILITY_SCORE
    ).any():
        raise ValueError(
            "A selected candidate violates "
            "the feasibility threshold."
        )

    if (
        shortlist["need_score"]
        < MINIMUM_NEED_SCORE
    ).any():
        raise ValueError(
            "A selected candidate violates "
            "the need threshold."
        )

    minimum_spacing = float(
        shortlist[
            "nearest_selected_candidate_m"
        ].min()
    )

    if (
        minimum_spacing
        + 1e-6
        < MINIMUM_SPACING_METERS
    ):
        raise ValueError(
            "Selected candidates violate "
            "the minimum spacing rule."
        )

    if not shortlist.geometry.is_valid.all():
        raise ValueError(
            "Invalid shortlist geometry was found."
        )

    print(
        "Ankara diverse shortlist validation "
        "completed successfully."
    )


def save_outputs(
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Save shortlist CSV and GeoPackage."""

    for path in (
        CSV_OUTPUT_PATH,
        GPKG_OUTPUT_PATH,
    ):
        if path.exists():
            path.unlink()

    csv_frame = pd.DataFrame(
        shortlist.drop(
            columns="geometry"
        )
    )

    csv_frame.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    shortlist.to_file(
        GPKG_OUTPUT_PATH,
        layer=OUTPUT_LAYER_NAME,
        driver="GPKG",
    )

    print(
        f"Shortlist CSV saved: {CSV_OUTPUT_PATH}"
    )

    print(
        f"Shortlist GeoPackage saved: {GPKG_OUTPUT_PATH}"
    )


def create_summary(
    total_count: int,
    eligible_count: int,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Create Ankara shortlist summary."""

    minimum_spacing = float(
        shortlist[
            "nearest_selected_candidate_m"
        ].min()
    )

    worst_rank = int(
        shortlist[
            "suitability_rank"
        ].max()
    )

    top_rows = []

    for row in shortlist.itertuples(
        index=False
    ):
        top_rows.append(
            (
                f"- #{row.diverse_selection_rank}: "
                f"`{row.grid_id}` — "
                f"suitability {row.suitability_score:.2f}, "
                f"feasibility {row.feasibility_score:.2f}, "
                f"need {row.need_score:.2f}"
            )
        )

    selected_lines = "\n".join(
        top_rows
    )

    summary = f"""# Ankara Diverse Candidate Shortlist

## Selection Configuration

- Total scored candidates: {total_count:,}
- Eligible candidates: {eligible_count:,}
- Desired shortlist size: {DESIRED_CANDIDATE_COUNT:,}
- Minimum spatial separation: {MINIMUM_SPACING_METERS:,.0f} m
- Minimum suitability score: {MINIMUM_SUITABILITY_SCORE:.0f}
- Minimum feasibility score: {MINIMUM_FEASIBILITY_SCORE:.0f}
- Minimum need score: {MINIMUM_NEED_SCORE:.0f}

## Final Shortlist

- Selected candidates: {len(shortlist):,}
- Minimum observed spacing: {minimum_spacing:,.2f} m
- Best original suitability rank selected: {int(shortlist["suitability_rank"].min()):,}
- Worst original suitability rank selected: {worst_rank:,}
- Lowest selected suitability: {shortlist["suitability_score"].min():.4f}
- Lowest selected feasibility: {shortlist["feasibility_score"].min():.4f}
- Lowest selected need: {shortlist["need_score"].min():.4f}

## Selected Candidates

{selected_lines}

## Method

Candidates first pass the same suitability, feasibility and need
quality filters used by the Çankaya pilot.

The remaining Ankara candidates are ordered by their original
suitability rank. A greedy spatial-diversity rule then selects the
highest-ranked candidate whose representative point is at least
5 kilometres from every already-selected candidate.

The Ankara spacing threshold is intentionally larger than the
1-kilometre Çankaya pilot threshold because the province-wide study
area is substantially larger.

The spatial-diversity rule changes only the final shortlist. It does
not change any candidate's underlying suitability score.

## Outputs

- `data/processed/ankara_diverse_candidate_shortlist.csv`
- `data/processed/ankara_diverse_candidate_shortlist.gpkg`

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Shortlist summary saved: {SUMMARY_OUTPUT_PATH}"
    )


def print_statistics(
    total_count: int,
    eligible_count: int,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Print shortlist statistics."""

    print("-" * 70)

    print(
        "Total scored candidates: "
        f"{total_count:,}"
    )

    print(
        "Eligible candidate count: "
        f"{eligible_count:,}"
    )

    print(
        "Selected candidate count: "
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

    print(
        "Lowest selected need score: "
        f"{shortlist['need_score'].min():.2f}"
    )

    print()

    print(
        shortlist[
            [
                "diverse_selection_rank",
                "suitability_rank",
                "grid_id",
                "suitability_score",
                "feasibility_score",
                "need_score",
                "parking_score",
                "nearest_selected_candidate_m",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    """Create Ankara's spatially diverse final shortlist."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Diverse Candidate Shortlist"
    )

    print("=" * 70)

    create_output_directories()
    validate_input_file()

    scores = load_spatial_scores()

    eligible = apply_eligibility_filters(
        scores
    )

    shortlist = (
        select_spatially_diverse_candidates(
            eligible
        )
    )

    shortlist = add_spacing_diagnostics(
        shortlist
    )

    shortlist = add_location_columns(
        shortlist
    )

    validate_shortlist(
        shortlist
    )

    save_outputs(
        shortlist
    )

    create_summary(
        total_count=len(scores),
        eligible_count=len(eligible),
        shortlist=shortlist,
    )

    print_statistics(
        total_count=len(scores),
        eligible_count=len(eligible),
        shortlist=shortlist,
    )

    print("=" * 70)

    print(
        "Ankara diverse candidate shortlist "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
