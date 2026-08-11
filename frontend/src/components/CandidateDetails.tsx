import type {
  DecisionSupportCandidate,
} from "../types/api";

import {
  MetricBar,
} from "./MetricBar";

interface CandidateDetailsProps {
  candidate: DecisionSupportCandidate | null;
}

function formatDistance(
  meters: number,
): string {
  return `${(meters / 1000).toFixed(1)} km`;
}

function supportLabel(
  candidate: DecisionSupportCandidate,
): string {
  switch (
    candidate
      .ml_support
      .support_label
  ) {
    case "all_three_top_20pct":
      return "All three models · top 20%";

    case "two_of_three_top_20pct":
      return "Two of three models · top 20%";

    case "one_of_three_top_20pct":
      return "One model · top 20%";

    default:
      return "No model · top 20%";
  }
}

export function CandidateDetails({
  candidate,
}: CandidateDetailsProps) {
  if (
    candidate ===
    null
  ) {
    return (
      <aside className="detail-panel detail-panel--empty">
        <p>
          Select a candidate on the map
          or shortlist.
        </p>
      </aside>
    );
  }

  const {
    suitability,
    ml_support: mlSupport,
    spatial_diversity:
      spatialDiversity,
  } = candidate;

  return (
    <aside className="detail-panel">
      <div className="detail-hero">
        <div>
          <p className="eyebrow">
            Candidate{" "}
            {candidate.selection_rank}
          </p>

          <h2>
            {candidate.grid_id}
          </h2>
        </div>

        <span
          className={`priority-chip priority-chip--${suitability.priority_band.toLowerCase()}`}
        >
          Priority{" "}
          {
            suitability.priority_band
          }
        </span>
      </div>

      <div className="primary-score">
        <div>
          <span>
            Explainable suitability
          </span>
          <strong>
            {suitability.score.toFixed(
              2,
            )}
          </strong>
        </div>

        <small>
          Province rank #
          {suitability.rank.toLocaleString()}
        </small>
      </div>

      <section className="detail-section">
        <div className="section-title-row">
          <h3>
            Decision factors
          </h3>
          <span>
            Primary layer
          </span>
        </div>

        <MetricBar
          label="Feasibility"
          value={
            suitability.feasibility
          }
        />

        <MetricBar
          label="Need"
          value={
            suitability.need
          }
        />

        <div className="metric-grid">
          <MetricBar
            label="Accessibility"
            value={
              suitability.accessibility
            }
            compact
          />

          <MetricBar
            label="Parking"
            value={
              suitability.parking
            }
            compact
          />

          <MetricBar
            label="Infrastructure gap"
            value={
              suitability.infrastructure_gap
            }
            compact
          />

          <MetricBar
            label="Technology gap"
            value={
              suitability.technology_gap
            }
            compact
          />
        </div>
      </section>

      <section className="detail-section">
        <div className="section-title-row">
          <h3>
            Historical-pattern ML support
          </h3>
          <span className="support-chip">
            {mlSupport.consensus_percentile.toFixed(
              2,
            )}
          </span>
        </div>

        <p className="support-explainer">
          Fold-normalized spatial OOF
          percentile. Supporting evidence,
          not a blended final score.
        </p>

        <MetricBar
          label="Logistic regression"
          value={
            mlSupport.logistic_regression_percentile
          }
          compact
        />

        <MetricBar
          label="Random forest"
          value={
            mlSupport.random_forest_percentile
          }
          compact
        />

        <MetricBar
          label="HistGradientBoosting"
          value={
            mlSupport.hist_gradient_boosting_percentile
          }
          compact
        />

        <div className="support-summary">
          <strong>
            {supportLabel(
              candidate,
            )}
          </strong>

          <span>
            Spread{" "}
            {mlSupport.model_percentile_spread.toFixed(
              1,
            )}{" "}
            pts
          </span>
        </div>

        {
          mlSupport.has_model_disagreement
            ? (
              <div className="warning-box">
                <strong>
                  Model disagreement
                </strong>

                <span>
                  The three models do not
                  provide uniformly strong
                  support. Keep the individual
                  percentiles visible.
                </span>
              </div>
            )
            : (
              <div className="agreement-box">
                <strong>
                  Cross-model agreement
                </strong>

                <span>
                  All three models place this
                  candidate in the candidate
                  top 20%.
                </span>
              </div>
            )
        }
      </section>

      <section className="detail-section detail-section--facts">
        <h3>
          Spatial context
        </h3>

        <dl className="facts-grid">
          <div>
            <dt>
              Nearest shortlist site
            </dt>
            <dd>
              {
                spatialDiversity.nearest_selected_grid_id ??
                "—"
              }
            </dd>
          </div>

          <div>
            <dt>
              Separation
            </dt>
            <dd>
              {formatDistance(
                spatialDiversity.nearest_selected_candidate_m,
              )}
            </dd>
          </div>

          <div>
            <dt>
              Longitude
            </dt>
            <dd>
              {candidate.location.longitude.toFixed(
                4,
              )}
            </dd>
          </div>

          <div>
            <dt>
              Latitude
            </dt>
            <dd>
              {candidate.location.latitude.toFixed(
                4,
              )}
            </dd>
          </div>
        </dl>
      </section>

      <section className="explanation-box">
        <p className="eyebrow">
          Why this site?
        </p>
        <p>
          {
            suitability.explanation
          }
        </p>
      </section>
    </aside>
  );
}
