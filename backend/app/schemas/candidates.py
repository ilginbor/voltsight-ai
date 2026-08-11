from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base schema that rejects undocumented response fields."""

    model_config = ConfigDict(
        extra="forbid",
    )


class CandidateLocation(StrictModel):
    """WGS84 representative-point coordinates."""

    longitude: float = Field(
        ge=-180.0,
        le=180.0,
    )
    latitude: float = Field(
        ge=-90.0,
        le=90.0,
    )


class CandidateSuitability(StrictModel):
    """Explainable suitability decision-support scores."""

    score: float = Field(
        ge=0.0,
        le=100.0,
    )
    rank: int = Field(
        ge=1,
    )
    percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    priority_band: str
    feasibility: float = Field(
        ge=0.0,
        le=100.0,
    )
    need: float = Field(
        ge=0.0,
        le=100.0,
    )
    accessibility: float = Field(
        ge=0.0,
        le=100.0,
    )
    parking: float = Field(
        ge=0.0,
        le=100.0,
    )
    infrastructure_gap: float = Field(
        ge=0.0,
        le=100.0,
    )
    technology_gap: float = Field(
        ge=0.0,
        le=100.0,
    )
    explanation: str


class CandidateSpatialDiversity(StrictModel):
    """Spatial-diversity diagnostics for the final shortlist."""

    nearest_selected_grid_id: str | None
    nearest_selected_candidate_m: float = Field(
        ge=0.0,
    )


class CandidateMLSupport(StrictModel):
    """Fold-normalized spatial-OOF supporting ML evidence."""

    method: Literal[
        "fold_normalized_spatial_oof_percentile"
    ]
    logistic_regression_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    random_forest_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    hist_gradient_boosting_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    consensus_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    consensus_rank: int = Field(
        ge=1,
    )
    minimum_model_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    maximum_model_percentile: float = Field(
        ge=0.0,
        le=100.0,
    )
    model_percentile_spread: float = Field(
        ge=0.0,
        le=100.0,
    )
    models_top_20pct_count: int = Field(
        ge=0,
        le=3,
    )
    models_top_10pct_count: int = Field(
        ge=0,
        le=3,
    )
    at_least_two_models_top_20pct: bool
    all_models_top_20pct: bool
    support_label: Literal[
        "all_three_top_20pct",
        "two_of_three_top_20pct",
        "one_of_three_top_20pct",
        "no_model_top_20pct",
    ]
    has_model_disagreement: bool


class DecisionSupportCandidate(StrictModel):
    """One final Ankara decision-support candidate."""

    grid_id: str
    selection_rank: int = Field(
        ge=1,
    )
    location: CandidateLocation
    suitability: CandidateSuitability
    spatial_diversity: CandidateSpatialDiversity
    ml_support: CandidateMLSupport


class DecisionPolicy(StrictModel):
    """Machine-readable separation between decision and ML layers."""

    primary_layer: Literal[
        "explainable_suitability"
    ]
    supporting_layer: Literal[
        "fold_normalized_spatial_oof_ml"
    ]
    ml_is_blended_into_suitability: Literal[
        False
    ]
    minimum_spacing_m: float = Field(
        gt=0.0,
    )


class DecisionSupportPayload(StrictModel):
    """Validated on-disk decision-support export contract."""

    schema_version: str
    study_area: str
    study_area_country: str
    coordinate_reference_system: str
    candidate_count: int = Field(
        ge=0,
    )
    decision_policy: DecisionPolicy
    generated_at_utc: datetime
    candidates: list[
        DecisionSupportCandidate
    ]


class CandidateListResponse(StrictModel):
    """API response containing the final candidate list."""

    count: int = Field(
        ge=0,
    )
    candidates: list[
        DecisionSupportCandidate
    ]


class DecisionSupportSummary(StrictModel):
    """Compact metadata response for the decision-support dataset."""

    schema_version: str
    study_area: str
    study_area_country: str
    coordinate_reference_system: str
    candidate_count: int = Field(
        ge=0,
    )
    generated_at_utc: datetime
    decision_policy: DecisionPolicy


class HealthResponse(StrictModel):
    """Backend health response."""

    status: Literal[
        "ok"
    ]
    service: Literal[
        "voltsight-api"
    ]
    data_available: bool
