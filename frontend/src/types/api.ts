export type SupportLabel =
  | "all_three_top_20pct"
  | "two_of_three_top_20pct"
  | "one_of_three_top_20pct"
  | "no_model_top_20pct";

export interface CandidateLocation {
  longitude: number;
  latitude: number;
}

export interface CandidateSuitability {
  score: number;
  rank: number;
  percentile: number;
  priority_band: string;
  feasibility: number;
  need: number;
  accessibility: number;
  parking: number;
  infrastructure_gap: number;
  technology_gap: number;
  explanation: string;
}

export interface CandidateSpatialDiversity {
  nearest_selected_grid_id: string | null;
  nearest_selected_candidate_m: number;
}

export interface CandidateMLSupport {
  method: "fold_normalized_spatial_oof_percentile";
  logistic_regression_percentile: number;
  random_forest_percentile: number;
  hist_gradient_boosting_percentile: number;
  consensus_percentile: number;
  consensus_rank: number;
  minimum_model_percentile: number;
  maximum_model_percentile: number;
  model_percentile_spread: number;
  models_top_20pct_count: number;
  models_top_10pct_count: number;
  at_least_two_models_top_20pct: boolean;
  all_models_top_20pct: boolean;
  support_label: SupportLabel;
  has_model_disagreement: boolean;
}

export interface DecisionSupportCandidate {
  grid_id: string;
  selection_rank: number;
  location: CandidateLocation;
  suitability: CandidateSuitability;
  spatial_diversity: CandidateSpatialDiversity;
  ml_support: CandidateMLSupport;
}

export interface CandidateListResponse {
  count: number;
  candidates: DecisionSupportCandidate[];
}

export interface DecisionPolicy {
  primary_layer: "explainable_suitability";
  supporting_layer: "fold_normalized_spatial_oof_ml";
  ml_is_blended_into_suitability: false;
  minimum_spacing_m: number;
}

export interface DecisionSupportSummary {
  schema_version: string;
  study_area: string;
  study_area_country: string;
  coordinate_reference_system: string;
  candidate_count: number;
  generated_at_utc: string;
  decision_policy: DecisionPolicy;
}
