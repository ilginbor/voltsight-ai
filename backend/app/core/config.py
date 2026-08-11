from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DECISION_SUPPORT_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ankara_decision_support_shortlist.json"
)

API_PREFIX = "/api/v1"

CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
