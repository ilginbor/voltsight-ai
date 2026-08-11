from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.candidate_service import (
    CandidateNotFoundError,
    CandidateService,
    DecisionSupportDataInvalidError,
    get_candidate_service,
)


def create_candidate(
    *,
    grid_id: str,
    selection_rank: int,
    longitude: float,
    consensus: float,
) -> dict[
    str,
    object,
]:
    """Create one valid decision-support candidate payload."""

    return {
        "grid_id": grid_id,
        "selection_rank": selection_rank,
        "location": {
            "longitude": longitude,
            "latitude": 39.90,
        },
        "suitability": {
            "score": 80.0,
            "rank": selection_rank,
            "percentile": 95.0,
            "priority_band": "A",
            "feasibility": 75.0,
            "need": 85.0,
            "accessibility": 70.0,
            "parking": 80.0,
            "infrastructure_gap": 90.0,
            "technology_gap": 60.0,
            "explanation": "Test explanation.",
        },
        "spatial_diversity": {
            "nearest_selected_grid_id": (
                "ANK_000002"
                if selection_rank == 1
                else "ANK_000001"
            ),
            "nearest_selected_candidate_m": 25_100.0,
        },
        "ml_support": {
            "method": (
                "fold_normalized_spatial_oof_percentile"
            ),
            "logistic_regression_percentile": 92.0,
            "random_forest_percentile": 96.0,
            "hist_gradient_boosting_percentile": 94.0,
            "consensus_percentile": consensus,
            "consensus_rank": selection_rank,
            "minimum_model_percentile": 92.0,
            "maximum_model_percentile": 96.0,
            "model_percentile_spread": 4.0,
            "models_top_20pct_count": 3,
            "models_top_10pct_count": 3,
            "at_least_two_models_top_20pct": True,
            "all_models_top_20pct": True,
            "support_label": "all_three_top_20pct",
            "has_model_disagreement": False,
        },
    }


def create_payload() -> dict[
    str,
    object,
]:
    """Create a complete valid API fixture."""

    return {
        "schema_version": "1.0",
        "study_area": "Ankara",
        "study_area_country": "TR",
        "coordinate_reference_system": "EPSG:4326",
        "candidate_count": 2,
        "decision_policy": {
            "primary_layer": "explainable_suitability",
            "supporting_layer": (
                "fold_normalized_spatial_oof_ml"
            ),
            "ml_is_blended_into_suitability": False,
            "minimum_spacing_m": 25_000,
        },
        "generated_at_utc": (
            "2026-08-11T15:00:00+00:00"
        ),
        "candidates": [
            create_candidate(
                grid_id="ANK_000001",
                selection_rank=1,
                longitude=32.80,
                consensus=96.0,
            ),
            create_candidate(
                grid_id="ANK_000002",
                selection_rank=2,
                longitude=32.90,
                consensus=92.0,
            ),
        ],
    }


@pytest.fixture
def service(
    tmp_path: Path,
) -> CandidateService:
    """Create a service backed by a temporary valid JSON export."""

    data_path = (
        tmp_path
        / "decision_support.json"
    )

    data_path.write_text(
        json.dumps(
            create_payload()
        ),
        encoding="utf-8",
    )

    return CandidateService(
        data_path=data_path
    )


@pytest.fixture
def client(
    service: CandidateService,
):
    """Override the FastAPI service dependency with the temp fixture."""

    app.dependency_overrides[
        get_candidate_service
    ] = lambda: service

    with TestClient(
        app
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_service_loads_valid_payload(
    service: CandidateService,
) -> None:
    payload = service.load_payload()

    assert payload.candidate_count == 2

    assert [
        candidate.grid_id
        for candidate in payload.candidates
    ] == [
        "ANK_000001",
        "ANK_000002",
    ]


def test_service_rejects_count_mismatch(
    tmp_path: Path,
) -> None:
    payload = create_payload()
    payload[
        "candidate_count"
    ] = 3

    data_path = (
        tmp_path
        / "bad_count.json"
    )

    data_path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    service = CandidateService(
        data_path=data_path
    )

    with pytest.raises(
        DecisionSupportDataInvalidError
    ):
        service.load_payload()


def test_service_returns_candidate_case_insensitively(
    service: CandidateService,
) -> None:
    candidate = service.get_candidate(
        "ank_000001"
    )

    assert (
        candidate.grid_id
        == "ANK_000001"
    )


def test_service_missing_candidate_raises(
    service: CandidateService,
) -> None:
    with pytest.raises(
        CandidateNotFoundError
    ):
        service.get_candidate(
            "ANK_999999"
        )


def test_health_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "voltsight-api",
        "data_available": True,
    }


def test_summary_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/summary"
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "candidate_count"
    ] == 2

    assert (
        body[
            "decision_policy"
        ][
            "ml_is_blended_into_suitability"
        ]
        is False
    )


def test_candidate_list_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/candidates"
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "count"
    ] == 2

    assert [
        candidate[
            "grid_id"
        ]
        for candidate in body[
            "candidates"
        ]
    ] == [
        "ANK_000001",
        "ANK_000002",
    ]


def test_candidate_detail_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/candidates/ANK_000002"
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "grid_id"
    ] == "ANK_000002"

    assert (
        body[
            "ml_support"
        ][
            "consensus_percentile"
        ]
        == 92.0
    )


def test_candidate_detail_returns_404(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/candidates/ANK_999999"
    )

    assert response.status_code == 404
