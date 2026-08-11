import type {
  DecisionSupportCandidate,
} from "../types/api";

interface CandidateListProps {
  candidates: DecisionSupportCandidate[];
  selectedGridId: string | null;
  onSelect: (gridId: string) => void;
}

function supportText(
  candidate: DecisionSupportCandidate,
): string {
  const count =
    candidate.ml_support.models_top_20pct_count;

  return `${count}/3 models · top 20%`;
}

export function CandidateList({
  candidates,
  selectedGridId,
  onSelect,
}: CandidateListProps) {
  return (
    <aside className="candidate-list-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">
            Spatial shortlist
          </p>
          <h2>Final candidates</h2>
        </div>

        <span className="count-badge">
          {candidates.length}
        </span>
      </div>

      <div className="candidate-list">
        {candidates.map(
          (candidate) => {
            const selected =
              candidate.grid_id ===
              selectedGridId;

            return (
              <button
                type="button"
                className={
                  selected
                    ? "candidate-card candidate-card--selected"
                    : "candidate-card"
                }
                key={candidate.grid_id}
                onClick={() => {
                  onSelect(
                    candidate.grid_id,
                  );
                }}
              >
                <div className="candidate-card__rank">
                  {candidate.selection_rank}
                </div>

                <div className="candidate-card__content">
                  <div className="candidate-card__topline">
                    <strong>
                      {candidate.grid_id}
                    </strong>

                    <span
                      className={`priority-chip priority-chip--${candidate.suitability.priority_band.toLowerCase()}`}
                    >
                      Band{" "}
                      {
                        candidate
                          .suitability
                          .priority_band
                      }
                    </span>
                  </div>

                  <div className="candidate-card__metrics">
                    <span>
                      Suitability{" "}
                      <strong>
                        {candidate.suitability.score.toFixed(
                          1,
                        )}
                      </strong>
                    </span>

                    <span>
                      ML{" "}
                      <strong>
                        {candidate.ml_support.consensus_percentile.toFixed(
                          1,
                        )}
                      </strong>
                    </span>
                  </div>

                  <small>
                    {supportText(
                      candidate,
                    )}
                  </small>
                </div>
              </button>
            );
          },
        )}
      </div>
    </aside>
  );
}
