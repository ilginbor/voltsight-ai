from __future__ import annotations

from fastapi import (
    Depends,
    FastAPI,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.app.core.config import (
    API_PREFIX,
    CORS_ORIGINS,
)
from backend.app.routers.candidates import (
    router as candidates_router,
)
from backend.app.schemas.candidates import (
    HealthResponse,
)
from backend.app.services.candidate_service import (
    CandidateService,
    get_candidate_service,
)


app = FastAPI(
    title="VoltSight API",
    version="0.1.0",
    description=(
        "Read-only API for VoltSight Ankara EV-charging "
        "decision-support outputs."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        CORS_ORIGINS
    ),
    allow_credentials=False,
    allow_methods=[
        "GET",
    ],
    allow_headers=[
        "*",
    ],
)

app.include_router(
    candidates_router,
    prefix=API_PREFIX,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[
        "system",
    ],
)
def health(
    service: CandidateService = Depends(
        get_candidate_service
    ),
) -> HealthResponse:
    """Return process health without forcing the dataset to load."""

    return HealthResponse(
        status="ok",
        service="voltsight-api",
        data_available=(
            service.data_available
        ),
    )
