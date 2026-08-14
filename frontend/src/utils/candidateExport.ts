import type {
  DecisionSupportCandidate,
} from "../types/api";

const CSV_HEADERS = [
  "selection_rank",
  "grid_id",
  "longitude",
  "latitude",
  "priority_band",
  "suitability_score",
  "suitability_rank",
  "feasibility",
  "need",
  "accessibility",
  "parking",
  "infrastructure_gap",
  "technology_gap",
  "ml_consensus_percentile",
  "ml_logistic_percentile",
  "ml_random_forest_percentile",
  "ml_hgb_percentile",
  "ml_model_spread",
  "models_top_20pct_count",
  "model_disagreement",
  "nearest_selected_grid_id",
  "nearest_selected_candidate_m",
] as const;

function escapeCsvValue(
  value: string | number | boolean | null,
): string {
  const text =
    value ===
    null
      ? ""
      : String(
          value,
        );

  return `"${text.replace(
    /"/g,
    '""',
  )}"`;
}

function candidateRow(
  candidate: DecisionSupportCandidate,
): Array<
  string | number | boolean | null
> {
  return [
    candidate.selection_rank,
    candidate.grid_id,
    candidate.location.longitude,
    candidate.location.latitude,
    candidate.suitability.priority_band,
    candidate.suitability.score,
    candidate.suitability.rank,
    candidate.suitability.feasibility,
    candidate.suitability.need,
    candidate.suitability.accessibility,
    candidate.suitability.parking,
    candidate.suitability.infrastructure_gap,
    candidate.suitability.technology_gap,
    candidate.ml_support.consensus_percentile,
    candidate.ml_support.logistic_regression_percentile,
    candidate.ml_support.random_forest_percentile,
    candidate.ml_support.hist_gradient_boosting_percentile,
    candidate.ml_support.model_percentile_spread,
    candidate.ml_support.models_top_20pct_count,
    candidate.ml_support.has_model_disagreement,
    candidate.spatial_diversity.nearest_selected_grid_id,
    candidate.spatial_diversity.nearest_selected_candidate_m,
  ];
}

export function buildCandidateCsv(
  candidates: DecisionSupportCandidate[],
): string {
  return [
    CSV_HEADERS.map(
      escapeCsvValue,
    ).join(
      ",",
    ),
    ...candidates.map(
      (
        candidate,
      ) =>
        candidateRow(
          candidate,
        )
          .map(
            escapeCsvValue,
          )
          .join(
            ",",
          ),
    ),
  ].join(
    "\n",
  );
}

export function downloadCandidateCsv(
  candidates: DecisionSupportCandidate[],
): boolean {
  if (
    candidates.length ===
    0
  ) {
    return false;
  }

  const blob =
    new Blob(
      [
        "\uFEFF",
        buildCandidateCsv(
          candidates,
        ),
      ],
      {
        type:
          "text/csv;charset=utf-8",
      },
    );

  const objectUrl =
    URL.createObjectURL(
      blob,
    );

  const anchor =
    document.createElement(
      "a",
    );

  anchor.href =
    objectUrl;

  anchor.download =
    "voltsight_ankara_filtered_candidates.csv";

  anchor.style.display =
    "none";

  document.body.appendChild(
    anchor,
  );

  anchor.click();
  anchor.remove();

  URL.revokeObjectURL(
    objectUrl,
  );

  return true;
}
