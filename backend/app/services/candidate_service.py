from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from backend.app.core.config import (
    DECISION_SUPPORT_JSON_PATH,
)
from backend.app.schemas.candidates import (
    CandidateListResponse,
    DecisionSupportCandidate,
    DecisionSupportPayload,
    DecisionSupportSummary,
)


class DecisionSupportDataNotFoundError(
    FileNotFoundError
):
    """Raised when the generated decision-support JSON is unavailable."""


class DecisionSupportDataInvalidError(
    ValueError
):
    """Raised when the generated decision-support JSON is invalid."""


class CandidateNotFoundError(
    LookupError
):
    """Raised when a requested candidate grid ID does not exist."""


class CandidateService:
    """Read and validate the generated Ankara decision-support export."""

    def __init__(
        self,
        data_path: Path = DECISION_SUPPORT_JSON_PATH,
    ) -> None:
        self.data_path = Path(
            data_path
        )
        self._payload: (
            DecisionSupportPayload
            | None
        ) = None
        self._candidate_index: dict[
            str,
            DecisionSupportCandidate,
        ] | None = None

    @property
    def data_available(
        self,
    ) -> bool:
        """Return whether the generated JSON exists."""

        return self.data_path.is_file()

    def _read_raw_payload(
        self,
    ) -> object:
        """Read the raw JSON document."""

        if not self.data_available:
            raise DecisionSupportDataNotFoundError(
                "Decision-support export was not found: "
                f"{self.data_path}"
            )

        try:
            return json.loads(
                self.data_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise DecisionSupportDataInvalidError(
                "Decision-support export could not be read "
                f"as valid JSON: {self.data_path}"
            ) from error

    @staticmethod
    def _validate_payload(
        raw_payload: object,
    ) -> DecisionSupportPayload:
        """Validate schema and cross-row invariants."""

        try:
            payload = (
                DecisionSupportPayload.model_validate(
                    raw_payload
                )
            )
        except ValidationError as error:
            raise DecisionSupportDataInvalidError(
                "Decision-support export failed schema validation."
            ) from error

        if (
            payload.candidate_count
            != len(
                payload.candidates
            )
        ):
            raise DecisionSupportDataInvalidError(
                "candidate_count does not match the candidate list."
            )

        grid_ids = [
            candidate.grid_id
            for candidate in payload.candidates
        ]

        if len(
            grid_ids
        ) != len(
            set(
                grid_ids
            )
        ):
            raise DecisionSupportDataInvalidError(
                "Decision-support export contains duplicate grid IDs."
            )

        selection_ranks = [
            candidate.selection_rank
            for candidate in payload.candidates
        ]

        expected_ranks = list(
            range(
                1,
                len(
                    payload.candidates
                )
                + 1,
            )
        )

        if selection_ranks != expected_ranks:
            raise DecisionSupportDataInvalidError(
                "Candidate selection ranks must be sequential and ordered."
            )

        return payload

    def load_payload(
        self,
        *,
        force_reload: bool = False,
    ) -> DecisionSupportPayload:
        """Load the export once and cache its validated representation."""

        if (
            self._payload is not None
            and not force_reload
        ):
            return self._payload

        raw_payload = self._read_raw_payload()

        payload = self._validate_payload(
            raw_payload
        )

        self._payload = payload
        self._candidate_index = {
            candidate.grid_id.upper(): candidate
            for candidate in payload.candidates
        }

        return payload

    def list_candidates(
        self,
    ) -> CandidateListResponse:
        """Return all final candidates in diverse-selection order."""

        payload = self.load_payload()

        return CandidateListResponse(
            count=payload.candidate_count,
            candidates=payload.candidates,
        )

    def get_candidate(
        self,
        grid_id: str,
    ) -> DecisionSupportCandidate:
        """Return one candidate by case-insensitive grid ID."""

        normalized = (
            str(
                grid_id
            )
            .strip()
            .upper()
        )

        if not normalized:
            raise CandidateNotFoundError(
                "Candidate grid ID is empty."
            )

        self.load_payload()

        assert (
            self._candidate_index
            is not None
        )

        candidate = self._candidate_index.get(
            normalized
        )

        if candidate is None:
            raise CandidateNotFoundError(
                f"Candidate not found: {normalized}"
            )

        return candidate

    def get_summary(
        self,
    ) -> DecisionSupportSummary:
        """Return compact dataset and decision-policy metadata."""

        payload = self.load_payload()

        return DecisionSupportSummary(
            schema_version=payload.schema_version,
            study_area=payload.study_area,
            study_area_country=(
                payload.study_area_country
            ),
            coordinate_reference_system=(
                payload.coordinate_reference_system
            ),
            candidate_count=(
                payload.candidate_count
            ),
            generated_at_utc=(
                payload.generated_at_utc
            ),
            decision_policy=(
                payload.decision_policy
            ),
        )


_candidate_service = CandidateService()


def get_candidate_service() -> CandidateService:
    """FastAPI dependency returning the shared read-only service."""

    return _candidate_service
