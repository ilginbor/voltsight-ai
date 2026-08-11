import type {
  CandidateListResponse,
  DecisionSupportCandidate,
  DecisionSupportSummary,
} from "../types/api";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "/api/v1"
).replace(/\/$/, "");

async function requestJson<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      headers: {
        Accept: "application/json",
      },
      signal,
    },
  );

  if (!response.ok) {
    let detail = "";

    try {
      const body = await response.json() as {
        detail?: string;
      };
      detail = body.detail ?? "";
    } catch {
      detail = "";
    }

    throw new Error(
      detail ||
        `VoltSight API request failed with HTTP ${response.status}.`,
    );
  }

  return response.json() as Promise<T>;
}

export function getSummary(
  signal?: AbortSignal,
): Promise<DecisionSupportSummary> {
  return requestJson<DecisionSupportSummary>(
    "/summary",
    signal,
  );
}

export function getCandidates(
  signal?: AbortSignal,
): Promise<CandidateListResponse> {
  return requestJson<CandidateListResponse>(
    "/candidates",
    signal,
  );
}

export function getCandidate(
  gridId: string,
  signal?: AbortSignal,
): Promise<DecisionSupportCandidate> {
  return requestJson<DecisionSupportCandidate>(
    `/candidates/${encodeURIComponent(gridId)}`,
    signal,
  );
}
