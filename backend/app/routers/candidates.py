from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.app.schemas.candidates import (
    CandidateListResponse,
    DecisionSupportCandidate,
    DecisionSupportSummary,
)
from backend.app.services.candidate_service import (
    CandidateNotFoundError,
    CandidateService,
    DecisionSupportDataInvalidError,
    DecisionSupportDataNotFoundError,
    get_candidate_service,
)


router = APIRouter(
    tags=[
        "decision-support",
    ],
)


def _raise_api_error(
    error: Exception,
) -> None:
    """Translate service-layer failures into stable HTTP responses."""

    if isinstance(
        error,
        CandidateNotFoundError,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(
                error
            ),
        ) from error

    if isinstance(
        error,
        DecisionSupportDataNotFoundError,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Decision-support data is not available. "
                "Run the Ankara export pipeline first."
            ),
        ) from error

    if isinstance(
        error,
        DecisionSupportDataInvalidError,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Decision-support data failed validation."
            ),
        ) from error

    raise error


@router.get(
    "/summary",
    response_model=DecisionSupportSummary,
)
def read_summary(
    service: CandidateService = Depends(
        get_candidate_service
    ),
) -> DecisionSupportSummary:
    """Return decision-support metadata and policy."""

    try:
        return service.get_summary()
    except (
        DecisionSupportDataNotFoundError,
        DecisionSupportDataInvalidError,
    ) as error:
        _raise_api_error(
            error
        )

    raise AssertionError(
        "Unreachable error translation branch."
    )


@router.get(
    "/candidates",
    response_model=CandidateListResponse,
)
def read_candidates(
    service: CandidateService = Depends(
        get_candidate_service
    ),
) -> CandidateListResponse:
    """Return the canonical spatially diverse Ankara shortlist."""

    try:
        return service.list_candidates()
    except (
        DecisionSupportDataNotFoundError,
        DecisionSupportDataInvalidError,
    ) as error:
        _raise_api_error(
            error
        )

    raise AssertionError(
        "Unreachable error translation branch."
    )


@router.get(
    "/candidates/{grid_id}",
    response_model=DecisionSupportCandidate,
)
def read_candidate(
    grid_id: str,
    service: CandidateService = Depends(
        get_candidate_service
    ),
) -> DecisionSupportCandidate:
    """Return one final candidate by grid ID."""

    try:
        return service.get_candidate(
            grid_id
        )
    except (
        CandidateNotFoundError,
        DecisionSupportDataNotFoundError,
        DecisionSupportDataInvalidError,
    ) as error:
        _raise_api_error(
            error
        )

    raise AssertionError(
        "Unreachable error translation branch."
    )
