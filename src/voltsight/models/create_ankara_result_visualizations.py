from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ankara_boundary_osm.geojson"
)

SCORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_candidate_suitability_scores.gpkg"
)

SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.gpkg"
)

SCORES_LAYER_NAME = "candidate_suitability_scores"
SHORTLIST_LAYER_NAME = "diverse_candidate_shortlist"

SUITABILITY_MAP_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_suitability_map.png"
)

SHORTLIST_MAP_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_final_shortlist_map.png"
)

DISTRIBUTION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_suitability_distribution.png"
)

FEASIBILITY_NEED_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_feasibility_need_plot.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_results_visualization_summary.md"
)

PROJECTED_CRS = "EPSG:32636"


def create_output_directory() -> None:
    """Create documentation output directory."""

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_inputs() -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Load Ankara boundary, scored candidates and shortlist."""

    required_paths = (
        BOUNDARY_PATH,
        SCORES_PATH,
        SHORTLIST_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input was not found: {path}"
            )

    boundary = gpd.read_file(
        BOUNDARY_PATH
    )

    scores = gpd.read_file(
        SCORES_PATH,
        layer=SCORES_LAYER_NAME,
    )

    shortlist = gpd.read_file(
        SHORTLIST_PATH,
        layer=SHORTLIST_LAYER_NAME,
    )

    if boundary.empty:
        raise ValueError(
            "Ankara boundary is empty."
        )

    if scores.empty:
        raise ValueError(
            "Ankara suitability dataset is empty."
        )

    if shortlist.empty:
        raise ValueError(
            "Ankara shortlist is empty."
        )

    if boundary.crs is None:
        raise ValueError(
            "Ankara boundary has no CRS."
        )

    if scores.crs is None:
        raise ValueError(
            "Suitability dataset has no CRS."
        )

    if shortlist.crs is None:
        raise ValueError(
            "Shortlist dataset has no CRS."
        )

    boundary = boundary.to_crs(
        PROJECTED_CRS
    )

    scores = scores.to_crs(
        PROJECTED_CRS
    )

    shortlist = shortlist.to_crs(
        PROJECTED_CRS
    )

    validate_inputs(
        scores,
        shortlist,
    )

    return (
        boundary,
        scores,
        shortlist,
    )


def validate_inputs(
    scores: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Validate result datasets before plotting."""

    required_score_columns = {
        "grid_id",
        "suitability_score",
        "feasibility_score",
        "need_score",
        "priority_band",
        "suitability_rank",
        "geometry",
    }

    missing_score_columns = (
        required_score_columns
        - set(scores.columns)
    )

    if missing_score_columns:
        raise ValueError(
            "Suitability data is missing columns: "
            f"{sorted(missing_score_columns)}"
        )

    required_shortlist_columns = {
        "grid_id",
        "diverse_selection_rank",
        "suitability_score",
        "geometry",
    }

    missing_shortlist_columns = (
        required_shortlist_columns
        - set(shortlist.columns)
    )

    if missing_shortlist_columns:
        raise ValueError(
            "Shortlist data is missing columns: "
            f"{sorted(missing_shortlist_columns)}"
        )

    if scores["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate scored candidate IDs were found."
        )

    if shortlist["grid_id"].duplicated().any():
        raise ValueError(
            "Duplicate shortlist IDs were found."
        )

    if len(shortlist) != 20:
        raise ValueError(
            "Expected exactly 20 shortlist candidates."
        )

    score_ids = set(
        scores["grid_id"].astype(str)
    )

    shortlist_ids = set(
        shortlist["grid_id"].astype(str)
    )

    if not shortlist_ids.issubset(
        score_ids
    ):
        raise ValueError(
            "Shortlist contains IDs that are not "
            "present in the suitability dataset."
        )

    numeric_columns = (
        "suitability_score",
        "feasibility_score",
        "need_score",
    )

    for column in numeric_columns:
        values = pd.to_numeric(
            scores[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid numeric values found in {column}."
            )

        if (
            (values < 0)
            | (values > 100)
        ).any():
            raise ValueError(
                f"Out-of-range values found in {column}."
            )

    for frame, name in (
        (
            scores,
            "suitability",
        ),
        (
            shortlist,
            "shortlist",
        ),
    ):
        if frame.geometry.isna().any():
            raise ValueError(
                f"Missing {name} geometry."
            )

        if frame.geometry.is_empty.any():
            raise ValueError(
                f"Empty {name} geometry."
            )

        if not frame.geometry.is_valid.all():
            raise ValueError(
                f"Invalid {name} geometry."
            )

    print(
        "Ankara result datasets validated successfully."
    )


def priority_band_counts(
    scores: pd.DataFrame,
) -> pd.Series:
    """Return deterministic A-E priority counts."""

    return (
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
        .astype(int)
    )


def create_suitability_map(
    boundary: gpd.GeoDataFrame,
    scores: gpd.GeoDataFrame,
) -> None:
    """Create province-wide suitability heatmap."""

    figure, axis = plt.subplots(
        figsize=(13, 11)
    )

    scores.plot(
        ax=axis,
        column="suitability_score",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidth=0,
        legend=True,
        rasterized=True,
        legend_kwds={
            "label": "Suitability score",
            "shrink": 0.65,
        },
    )

    boundary.boundary.plot(
        ax=axis,
        linewidth=0.8,
    )

    axis.set_title(
        "VoltSight - Ankara EV Charging Candidate Suitability"
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        SUITABILITY_MAP_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Suitability map saved: {SUITABILITY_MAP_PATH}"
    )


def create_shortlist_map(
    boundary: gpd.GeoDataFrame,
    scores: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Create map of 20 final spatially diverse candidates."""

    figure, axis = plt.subplots(
        figsize=(13, 11)
    )

    scores.plot(
        ax=axis,
        column="suitability_score",
        cmap="Greys",
        linewidth=0,
        alpha=0.45,
        rasterized=True,
    )

    boundary.boundary.plot(
        ax=axis,
        linewidth=0.9,
    )

    points = shortlist.copy()

    points["geometry"] = (
        points.geometry
        .representative_point()
    )

    points.plot(
        ax=axis,
        column="suitability_score",
        cmap="viridis",
        vmin=0,
        vmax=100,
        markersize=70,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )

    for row in points.itertuples(
        index=False
    ):
        axis.annotate(
            str(
                row.diverse_selection_rank
            ),
            xy=(
                row.geometry.x,
                row.geometry.y,
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            zorder=6,
        )

    axis.set_title(
        "VoltSight - Ankara Final 20 Candidates " + "(25 km Minimum Spacing)"
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_axis_off()

    figure.tight_layout()

    figure.savefig(
        SHORTLIST_MAP_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Shortlist map saved: {SHORTLIST_MAP_PATH}"
    )


def create_distribution_plot(
    scores: gpd.GeoDataFrame,
) -> None:
    """Create suitability-score distribution chart."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        scores[
            "suitability_score"
        ],
        bins=40,
        edgecolor="black",
        linewidth=0.4,
    )

    median_score = float(
        scores[
            "suitability_score"
        ].median()
    )

    axis.axvline(
        median_score,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Median = {median_score:.2f}"
        ),
    )

    axis.set_title(
        "Ankara Candidate Suitability Score Distribution"
    )

    axis.set_xlabel(
        "Suitability score"
    )

    axis.set_ylabel(
        "Candidate grid cells"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        DISTRIBUTION_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Distribution plot saved: {DISTRIBUTION_PATH}"
    )


def create_feasibility_need_plot(
    scores: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Visualize feasibility/need balance."""

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    scatter = axis.scatter(
        scores[
            "feasibility_score"
        ],
        scores[
            "need_score"
        ],
        c=scores[
            "suitability_score"
        ],
        cmap="viridis",
        s=7,
        alpha=0.25,
    )

    axis.scatter(
        shortlist[
            "feasibility_score"
        ],
        shortlist[
            "need_score"
        ],
        c=shortlist[
            "suitability_score"
        ],
        cmap="viridis",
        vmin=0,
        vmax=100,
        s=75,
        edgecolors="black",
        linewidths=0.7,
        label="Final shortlist",
    )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
    )

    colorbar.set_label(
        "Suitability score"
    )

    axis.set_xlim(
        0,
        100,
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.set_xlabel(
        "Feasibility score"
    )

    axis.set_ylabel(
        "Need score"
    )

    axis.set_title(
        "Ankara Candidate Feasibility vs Infrastructure Need"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FEASIBILITY_NEED_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Feasibility/need plot saved: "
        f"{FEASIBILITY_NEED_PATH}"
    )


def create_summary(
    scores: gpd.GeoDataFrame,
    shortlist: gpd.GeoDataFrame,
) -> None:
    """Create visualization-result summary."""

    bands = priority_band_counts(
        scores
    )

    top_candidate = (
        scores.sort_values(
            [
                "suitability_rank",
                "grid_id",
            ],
            kind="stable",
        )
        .iloc[0]
    )

    summary = f"""# Ankara Result Visualization Summary

## Suitability Dataset

- Candidate cells: {len(scores):,}
- Median suitability: {scores["suitability_score"].median():.2f}
- Maximum suitability: {scores["suitability_score"].max():.2f}
- Minimum suitability: {scores["suitability_score"].min():.2f}
- Top candidate: `{top_candidate["grid_id"]}`
- Top candidate score: {top_candidate["suitability_score"]:.4f}

## Priority Distribution

- A: {int(bands["A"]):,}
- B: {int(bands["B"]):,}
- C: {int(bands["C"]):,}
- D: {int(bands["D"]):,}
- E: {int(bands["E"]):,}

## Final Spatially Diverse Shortlist

- Final candidates: {len(shortlist):,}
- Minimum observed spacing: {shortlist["nearest_selected_candidate_m"].min():,.2f} m
- Lowest shortlist suitability: {shortlist["suitability_score"].min():.2f}
- Lowest shortlist feasibility: {shortlist["feasibility_score"].min():.2f}
- Lowest shortlist need: {shortlist["need_score"].min():.2f}
- Worst original rank selected: {int(shortlist["suitability_rank"].max()):,}

## Generated Figures

- `docs/ankara_suitability_map.png`
- `docs/ankara_final_shortlist_map.png`
- `docs/ankara_suitability_distribution.png`
- `docs/ankara_feasibility_need_plot.png`

## Interpretation

The province-wide suitability map shows relative suitability across
Ankara candidate cells.

The final shortlist map applies the five-kilometre spatial-diversity
constraint, so the final 20 sites are intended to represent distinct
high-quality investment areas rather than a cluster of neighboring
high-scoring grid cells.

The feasibility-versus-need plot highlights the core VoltSight design:
a strong candidate should combine practical installation conditions
with a meaningful charging-infrastructure gap.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"Visualization summary saved: {SUMMARY_PATH}"
    )


def main() -> None:
    """Create Ankara result visualizations."""

    print("=" * 70)

    print(
        "VoltSight - Ankara Result Visualizations"
    )

    print("=" * 70)

    create_output_directory()

    (
        boundary,
        scores,
        shortlist,
    ) = load_inputs()

    print(
        "Candidate count: "
        f"{len(scores):,}"
    )

    print(
        "Shortlist count: "
        f"{len(shortlist):,}"
    )

    create_suitability_map(
        boundary,
        scores,
    )

    create_shortlist_map(
        boundary,
        scores,
        shortlist,
    )

    create_distribution_plot(
        scores
    )

    create_feasibility_need_plot(
        scores,
        shortlist,
    )

    create_summary(
        scores,
        shortlist,
    )

    print("=" * 70)

    print(
        "Ankara result visualization "
        "pipeline completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
