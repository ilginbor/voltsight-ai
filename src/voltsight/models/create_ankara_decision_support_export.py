from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SHORTLIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_diverse_candidate_shortlist.csv"
)

ML_SUPPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_shortlist_ml_support.csv"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_decision_support_shortlist.csv"
)

JSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_decision_support_shortlist.json"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "ankara_decision_support_export_summary.md"
)

EXPECTED_SHORTLIST_COUNT = 20

SUITABILITY_COLUMNS = (
    "grid_id",
    "diverse_selection_rank",
    "center_longitude",
    "center_latitude",
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
    "nearest_selected_grid_id",
    "nearest_selected_candidate_m",
    "score_explanation",
)

ML_COLUMNS = (
    "grid_id",
    "shortlist_order",
    "logistic_regression_percentile",
    "random_forest_percentile",
    "hist_gradient_boosting_percentile",
    "ml_consensus_percentile",
    "ml_consensus_rank",
    "ml_min_percentile",
    "ml_max_percentile",
    "ml_model_spread",
    "models_top_20pct_count",
    "models_top_10pct_count",
    "at_least_two_models_top_20pct",
    "all_models_top_20pct",
)


def model_support_label(
    *,
    models_top_20pct_count: int,
) -> str:
    """Return a transparent cross-model support label."""

    count = int(
        models_top_20pct_count
    )

    if count == 3:
        return "all_three_top_20pct"

    if count == 2:
        return "two_of_three_top_20pct"

    if count == 1:
        return "one_of_three_top_20pct"

    if count == 0:
        return "no_model_top_20pct"

    raise ValueError(
        "models_top_20pct_count must be between 0 and 3."
    )


def validate_required_columns(
    dataframe: pd.DataFrame,
    *,
    required: tuple[
        str,
        ...,
    ],
    label: str,
) -> None:
    """Validate required input columns and unique grid IDs."""

    missing = (
        set(
            required
        )
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            f"{label} is missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{label} is empty."
        )

    if dataframe[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            f"{label} contains duplicate grid IDs."
        )


def build_export_table(
    shortlist: pd.DataFrame,
    ml_support: pd.DataFrame,
) -> pd.DataFrame:
    """Merge explainable shortlist results with fold-normalized ML support."""

    validate_required_columns(
        shortlist,
        required=SUITABILITY_COLUMNS,
        label="Shortlist",
    )

    validate_required_columns(
        ml_support,
        required=ML_COLUMNS,
        label="ML support",
    )

    shortlist = shortlist[
        list(
            SUITABILITY_COLUMNS
        )
    ].copy()

    ml_support = ml_support[
        list(
            ML_COLUMNS
        )
    ].copy()

    shortlist[
        "grid_id"
    ] = (
        shortlist[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    ml_support[
        "grid_id"
    ] = (
        ml_support[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    shortlist_ids = set(
        shortlist[
            "grid_id"
        ]
    )

    support_ids = set(
        ml_support[
            "grid_id"
        ]
    )

    if shortlist_ids != support_ids:
        raise ValueError(
            "Shortlist and ML-support grid IDs do not match exactly."
        )

    result = shortlist.merge(
        ml_support,
        on="grid_id",
        how="inner",
        validate="one_to_one",
    )

    result[
        "diverse_selection_rank"
    ] = pd.to_numeric(
        result[
            "diverse_selection_rank"
        ],
        errors="raise",
    ).astype(
        int
    )

    result[
        "shortlist_order"
    ] = pd.to_numeric(
        result[
            "shortlist_order"
        ],
        errors="raise",
    ).astype(
        int
    )

    if not np.array_equal(
        result[
            "diverse_selection_rank"
        ].to_numpy(
            dtype=int
        ),
        result[
            "shortlist_order"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise ValueError(
            "Shortlist order does not match diverse selection rank."
        )

    result[
        "model_support_label"
    ] = [
        model_support_label(
            models_top_20pct_count=count
        )
        for count in result[
            "models_top_20pct_count"
        ]
    ]

    result[
        "has_model_disagreement"
    ] = (
        result[
            "models_top_20pct_count"
        ].astype(
            int
        )
        < 3
    )

    result = result.sort_values(
        [
            "diverse_selection_rank",
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

    return result


def validate_export_table(
    dataframe: pd.DataFrame,
    *,
    expected_count: int = EXPECTED_SHORTLIST_COUNT,
) -> None:
    """Validate the frontend/API-ready decision-support export."""

    if len(
        dataframe
    ) != expected_count:
        raise ValueError(
            "Unexpected decision-support row count: "
            f"{len(dataframe)} != {expected_count}"
        )

    if dataframe[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Decision-support export contains duplicate grid IDs."
        )

    expected_ranks = np.arange(
        1,
        expected_count + 1,
        dtype=int,
    )

    actual_ranks = dataframe[
        "diverse_selection_rank"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        actual_ranks,
        expected_ranks,
    ):
        raise ValueError(
            "Decision-support export ranks are not sequential."
        )

    bounded_columns = (
        "suitability_score",
        "suitability_percentile",
        "feasibility_score",
        "need_score",
        "accessibility_score",
        "parking_score",
        "infrastructure_gap_score",
        "technology_gap_score",
        "logistic_regression_percentile",
        "random_forest_percentile",
        "hist_gradient_boosting_percentile",
        "ml_consensus_percentile",
        "ml_min_percentile",
        "ml_max_percentile",
    )

    for column in bounded_columns:
        numeric = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

        if numeric.isna().any():
            raise ValueError(
                f"Missing numeric values found in {column}."
            )

        if not numeric.between(
            0.0,
            100.0,
        ).all():
            raise ValueError(
                f"Values outside 0-100 found in {column}."
            )

    longitude = pd.to_numeric(
        dataframe[
            "center_longitude"
        ],
        errors="coerce",
    )

    latitude = pd.to_numeric(
        dataframe[
            "center_latitude"
        ],
        errors="coerce",
    )

    if (
        longitude.isna().any()
        or latitude.isna().any()
    ):
        raise ValueError(
            "Missing shortlist coordinates were found."
        )

    if not longitude.between(
        -180.0,
        180.0,
    ).all():
        raise ValueError(
            "Invalid longitude values were found."
        )

    if not latitude.between(
        -90.0,
        90.0,
    ).all():
        raise ValueError(
            "Invalid latitude values were found."
        )

    support_counts = pd.to_numeric(
        dataframe[
            "models_top_20pct_count"
        ],
        errors="raise",
    ).astype(
        int
    )

    if not support_counts.between(
        0,
        3,
    ).all():
        raise ValueError(
            "Invalid model-support counts were found."
        )


def clean_json_value(
    value: object,
) -> object:
    """Convert pandas / NumPy scalar values to JSON-safe Python values."""

    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(
            value
        )

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    return value


def create_candidate_payload(
    row: pd.Series,
) -> dict[
    str,
    object,
]:
    """Create one nested decision-support JSON candidate object."""

    return {
        "grid_id": str(
            row[
                "grid_id"
            ]
        ),
        "selection_rank": int(
            row[
                "diverse_selection_rank"
            ]
        ),
        "location": {
            "longitude": float(
                row[
                    "center_longitude"
                ]
            ),
            "latitude": float(
                row[
                    "center_latitude"
                ]
            ),
        },
        "suitability": {
            "score": float(
                row[
                    "suitability_score"
                ]
            ),
            "rank": int(
                row[
                    "suitability_rank"
                ]
            ),
            "percentile": float(
                row[
                    "suitability_percentile"
                ]
            ),
            "priority_band": str(
                row[
                    "priority_band"
                ]
            ),
            "feasibility": float(
                row[
                    "feasibility_score"
                ]
            ),
            "need": float(
                row[
                    "need_score"
                ]
            ),
            "accessibility": float(
                row[
                    "accessibility_score"
                ]
            ),
            "parking": float(
                row[
                    "parking_score"
                ]
            ),
            "infrastructure_gap": float(
                row[
                    "infrastructure_gap_score"
                ]
            ),
            "technology_gap": float(
                row[
                    "technology_gap_score"
                ]
            ),
            "explanation": str(
                row[
                    "score_explanation"
                ]
            ),
        },
        "spatial_diversity": {
            "nearest_selected_grid_id": (
                None
                if pd.isna(
                    row[
                        "nearest_selected_grid_id"
                    ]
                )
                else str(
                    row[
                        "nearest_selected_grid_id"
                    ]
                )
            ),
            "nearest_selected_candidate_m": float(
                row[
                    "nearest_selected_candidate_m"
                ]
            ),
        },
        "ml_support": {
            "method": "fold_normalized_spatial_oof_percentile",
            "logistic_regression_percentile": float(
                row[
                    "logistic_regression_percentile"
                ]
            ),
            "random_forest_percentile": float(
                row[
                    "random_forest_percentile"
                ]
            ),
            "hist_gradient_boosting_percentile": float(
                row[
                    "hist_gradient_boosting_percentile"
                ]
            ),
            "consensus_percentile": float(
                row[
                    "ml_consensus_percentile"
                ]
            ),
            "consensus_rank": int(
                row[
                    "ml_consensus_rank"
                ]
            ),
            "minimum_model_percentile": float(
                row[
                    "ml_min_percentile"
                ]
            ),
            "maximum_model_percentile": float(
                row[
                    "ml_max_percentile"
                ]
            ),
            "model_percentile_spread": float(
                row[
                    "ml_model_spread"
                ]
            ),
            "models_top_20pct_count": int(
                row[
                    "models_top_20pct_count"
                ]
            ),
            "models_top_10pct_count": int(
                row[
                    "models_top_10pct_count"
                ]
            ),
            "at_least_two_models_top_20pct": bool(
                row[
                    "at_least_two_models_top_20pct"
                ]
            ),
            "all_models_top_20pct": bool(
                row[
                    "all_models_top_20pct"
                ]
            ),
            "support_label": str(
                row[
                    "model_support_label"
                ]
            ),
            "has_model_disagreement": bool(
                row[
                    "has_model_disagreement"
                ]
            ),
        },
    }


def create_json_payload(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    object,
]:
    """Create the complete frontend/API-oriented JSON payload."""

    candidates = [
        create_candidate_payload(
            row
        )
        for _,
        row in dataframe.iterrows()
    ]

    return {
        "schema_version": "1.0",
        "study_area": "Ankara",
        "study_area_country": "TR",
        "coordinate_reference_system": "EPSG:4326",
        "candidate_count": len(
            candidates
        ),
        "decision_policy": {
            "primary_layer": "explainable_suitability",
            "supporting_layer": "fold_normalized_spatial_oof_ml",
            "ml_is_blended_into_suitability": False,
            "minimum_spacing_m": 25_000,
        },
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "candidates": candidates,
    }


def save_outputs(
    dataframe: pd.DataFrame,
    payload: dict[
        str,
        object,
    ],
) -> None:
    """Save flat CSV and nested JSON decision-support outputs."""

    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    JSON_OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def create_summary(
    dataframe: pd.DataFrame,
) -> None:
    """Document the decision-support export contract."""

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_three = int(
        (
            dataframe[
                "models_top_20pct_count"
            ]
            == 3
        ).sum()
    )

    two_of_three = int(
        (
            dataframe[
                "models_top_20pct_count"
            ]
            == 2
        ).sum()
    )

    median_consensus = float(
        dataframe[
            "ml_consensus_percentile"
        ].median()
    )

    summary = f"""# Ankara Decision-Support Export

## Purpose

This export packages the canonical 20-site Ankara shortlist into a stable,
frontend/API-oriented contract.

The export does not create a new score and does not blend machine-learning
predictions into suitability.

## Inputs

- `data/processed/{SHORTLIST_PATH.name}`
- `data/processed/{ML_SUPPORT_PATH.name}`

## Outputs

- `data/processed/{CSV_OUTPUT_PATH.name}`
- `data/processed/{JSON_OUTPUT_PATH.name}`

The CSV is a flat analysis-friendly table.

The JSON is a nested application-oriented representation with:

- location
- explainable suitability components
- spatial-diversity diagnostics
- fold-normalized spatial OOF ML support

## Decision Policy

Primary decision layer:

- explainable suitability
- eligibility thresholds
- 25-km spatial diversity

Supporting evidence layer:

- Logistic Regression fold-normalized OOF percentile
- Random Forest fold-normalized OOF percentile
- HistGradientBoosting fold-normalized OOF percentile
- median cross-model consensus
- explicit cross-model disagreement information

ML support is not a calibrated probability and is not blended into the
canonical suitability score.

## Current Shortlist Export

- Rows: {len(dataframe)}
- Median ML consensus percentile: {median_consensus:.2f}
- All three models in candidate top 20%: {all_three}/{len(dataframe)}
- Exactly two models in candidate top 20%: {two_of_three}/{len(dataframe)}

## JSON Contract

Top-level keys:

```text
schema_version
study_area
study_area_country
coordinate_reference_system
candidate_count
decision_policy
generated_at_utc
candidates
```

Each candidate contains:

```text
grid_id
selection_rank
location
suitability
spatial_diversity
ml_support
```

## Interpretation

The application layer should present suitability and ML support as separate
axes.

A candidate with high suitability and lower ML agreement is not automatically
invalid. It can represent a gap-oriented recommendation that differs from the
historical mapped station-placement pattern.

Cross-model disagreement should remain visible rather than being hidden behind
the median consensus value.

## Generated At

{datetime.now(timezone.utc).isoformat()}
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the canonical shortlist and its ML-support diagnostics."""

    for path in (
        SHORTLIST_PATH,
        ML_SUPPORT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    shortlist = pd.read_csv(
        SHORTLIST_PATH,
        dtype={
            "grid_id": str,
        },
    )

    ml_support = pd.read_csv(
        ML_SUPPORT_PATH,
        dtype={
            "grid_id": str,
        },
    )

    return (
        shortlist,
        ml_support,
    )


def print_results(
    dataframe: pd.DataFrame,
) -> None:
    """Print the compact decision-support export table."""

    columns = [
        "diverse_selection_rank",
        "grid_id",
        "center_longitude",
        "center_latitude",
        "suitability_score",
        "feasibility_score",
        "need_score",
        "ml_consensus_percentile",
        "models_top_20pct_count",
        "model_support_label",
    ]

    print(
        dataframe[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )


def main() -> None:
    """Create frontend/API-ready Ankara shortlist exports."""

    print(
        "="
        * 70
    )

    print(
        "VoltSight - Ankara Decision-Support Export"
    )

    print(
        "="
        * 70
    )

    (
        shortlist,
        ml_support,
    ) = load_inputs()

    dataframe = build_export_table(
        shortlist,
        ml_support,
    )

    validate_export_table(
        dataframe
    )

    payload = create_json_payload(
        dataframe
    )

    save_outputs(
        dataframe,
        payload,
    )

    create_summary(
        dataframe
    )

    print_results(
        dataframe
    )

    print(
        "="
        * 70
    )

    print(
        "Ankara decision-support export completed successfully."
    )

    print(
        "="
        * 70
    )


if __name__ == "__main__":
    main()
