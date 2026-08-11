import type {
  CandidateLocation,
  CandidateMLSupport,
  CandidateSpatialDiversity,
  CandidateSuitability,
  DecisionSupportCandidate,
} from "../types/api";

type CandidateOverrides = Partial<
  Omit<
    DecisionSupportCandidate,
    | "location"
    | "suitability"
    | "spatial_diversity"
    | "ml_support"
  >
> & {
  location?: Partial<
    CandidateLocation
  >;
  suitability?: Partial<
    CandidateSuitability
  >;
  spatial_diversity?: Partial<
    CandidateSpatialDiversity
  >;
  ml_support?: Partial<
    CandidateMLSupport
  >;
};

export function createCandidate(
  overrides: CandidateOverrides = {},
): DecisionSupportCandidate {
  const base: DecisionSupportCandidate = {
    grid_id: "ANK_004300",
    selection_rank: 1,
    location: {
      longitude: 31.3405406,
      latitude: 40.1952609,
    },
    suitability: {
      score: 89.7487,
      rank: 1,
      percentile: 100.0,
      priority_band: "A",
      feasibility: 82.8032,
      need: 97.2768,
      accessibility: 94.069,
      parking: 65.9045,
      infrastructure_gap: 96.7962,
      technology_gap: 100.0,
      explanation:
        "strong road accessibility; large charging infrastructure gap; AC/DC technology gap",
    },
    spatial_diversity: {
      nearest_selected_grid_id: "ANK_007151",
      nearest_selected_candidate_m: 25079.87,
    },
    ml_support: {
      method: "fold_normalized_spatial_oof_percentile",
      logistic_regression_percentile: 72.1081,
      random_forest_percentile: 98.6855,
      hist_gradient_boosting_percentile: 92.6485,
      consensus_percentile: 92.6485,
      consensus_rank: 6687,
      minimum_model_percentile: 72.1081,
      maximum_model_percentile: 98.6855,
      model_percentile_spread: 26.5774,
      models_top_20pct_count: 2,
      models_top_10pct_count: 2,
      at_least_two_models_top_20pct: true,
      all_models_top_20pct: false,
      support_label: "two_of_three_top_20pct",
      has_model_disagreement: true,
    },
  };

  return {
    ...base,
    ...overrides,
    location: {
      ...base.location,
      ...overrides.location,
    },
    suitability: {
      ...base.suitability,
      ...overrides.suitability,
    },
    spatial_diversity: {
      ...base.spatial_diversity,
      ...overrides.spatial_diversity,
    },
    ml_support: {
      ...base.ml_support,
      ...overrides.ml_support,
    },
  };
}
