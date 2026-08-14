import type {
  DecisionSupportCandidate,
} from "../types/api";

interface CandidateCompareProps {
  candidates: DecisionSupportCandidate[];
  onRemove: (gridId: string) => void;
  onClear: () => void;
  onSelect: (gridId: string) => void;
}

interface ComparisonRow {
  label: string;
  render: (
    candidate: DecisionSupportCandidate,
  ) => string;
}

interface ComparisonHighlight {
  label: string;
  value: string;
  gridIds: string[];
}

function bestCandidates(
  candidates: DecisionSupportCandidate[],
  value: (
    candidate: DecisionSupportCandidate,
  ) => number,
  direction: "max" | "min",
): DecisionSupportCandidate[] {
  if (
    candidates.length ===
    0
  ) {
    return [];
  }

  const values =
    candidates.map(
      value,
    );

  const bestValue =
    direction ===
    "max"
      ? Math.max(
          ...values,
        )
      : Math.min(
          ...values,
        );

  return candidates.filter(
    (
      candidate,
    ) =>
      Math.abs(
        value(
          candidate,
        ) -
        bestValue,
      ) <
      1e-9,
  );
}

function comparisonHighlights(
  candidates: DecisionSupportCandidate[],
): ComparisonHighlight[] {
  const definitions = [
    {
      label:
        "En yüksek uygunluk",
      direction:
        "max" as const,
      value:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.score,
      format:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.score.toFixed(
            2,
          ),
    },
    {
      label:
        "En yüksek uygulanabilirlik",
      direction:
        "max" as const,
      value:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.feasibility,
      format:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.feasibility.toFixed(
            1,
          ),
    },
    {
      label:
        "En yüksek ihtiyaç",
      direction:
        "max" as const,
      value:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.need,
      format:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.suitability.need.toFixed(
            1,
          ),
    },
    {
      label:
        "En yüksek ML uzlaşısı",
      direction:
        "max" as const,
      value:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.ml_support.consensus_percentile,
      format:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          `${candidate.ml_support.consensus_percentile.toFixed(
            1,
          )}%`,
    },
    {
      label:
        "En düşük model farkı",
      direction:
        "min" as const,
      value:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          candidate.ml_support.model_percentile_spread,
      format:
        (
          candidate: DecisionSupportCandidate,
        ) =>
          `${candidate.ml_support.model_percentile_spread.toFixed(
            1,
          )} puan`,
    },
  ];

  return definitions.map(
    (
      definition,
    ) => {
      const winners =
        bestCandidates(
          candidates,
          definition.value,
          definition.direction,
        );

      return {
        label:
          definition.label,
        value:
          winners.length >
          0
            ? definition.format(
                winners[0],
              )
            : "—",
        gridIds:
          winners.map(
            (
              candidate,
            ) =>
              candidate.grid_id,
          ),
      };
    },
  );
}

const comparisonRows: ComparisonRow[] = [
  {
    label: "Uygunluk",
    render: (
      candidate,
    ) =>
      candidate.suitability.score.toFixed(
        2,
      ),
  },
  {
    label: "Uygulanabilirlik",
    render: (
      candidate,
    ) =>
      candidate.suitability.feasibility.toFixed(
        1,
      ),
  },
  {
    label: "İhtiyaç",
    render: (
      candidate,
    ) =>
      candidate.suitability.need.toFixed(
        1,
      ),
  },
  {
    label: "ML uzlaşısı",
    render: (
      candidate,
    ) =>
      candidate.ml_support.consensus_percentile.toFixed(
        1,
      ),
  },
  {
    label: "Lojistik",
    render: (
      candidate,
    ) =>
      candidate.ml_support.logistic_regression_percentile.toFixed(
        1,
      ),
  },
  {
    label: "Random Forest",
    render: (
      candidate,
    ) =>
      candidate.ml_support.random_forest_percentile.toFixed(
        1,
      ),
  },
  {
    label: "HGB",
    render: (
      candidate,
    ) =>
      candidate.ml_support.hist_gradient_boosting_percentile.toFixed(
        1,
      ),
  },
  {
    label: "Model farkı",
    render: (
      candidate,
    ) =>
      `${candidate.ml_support.model_percentile_spread.toFixed(1)} puan`,
  },
  {
    label: "İlk %20 model sayısı",
    render: (
      candidate,
    ) =>
      `${candidate.ml_support.models_top_20pct_count}/3`,
  },
  {
    label: "En yakın kısa liste adayına mesafe",
    render: (
      candidate,
    ) =>
      `${(
        candidate.spatial_diversity.nearest_selected_candidate_m /
        1000
      ).toFixed(1)} km`,
  },
];

export function CandidateCompare({
  candidates,
  onRemove,
  onClear,
  onSelect,
}: CandidateCompareProps) {
  const highlights =
    comparisonHighlights(
      candidates,
    );

  return (
    <section className="compare-panel">
      <div className="compare-panel__header">
        <div>
          <p className="eyebrow">
            Aday karşılaştırma
          </p>
          <h2>
            Kısa liste adaylarını karşılaştır
          </h2>
        </div>

        <div className="compare-panel__actions">
          <span>
            {candidates.length}/3 seçili
          </span>

          <button
            type="button"
            className="text-button"
            onClick={
              onClear
            }
          >
            Temizle
          </button>
        </div>
      </div>

      {
        candidates.length <
        2
          ? (
            <div className="compare-hint">
              Yan yana metrikleri görmek için
              bir aday daha ekleyin.
            </div>
          )
          : (
            <div className="compare-body">
              <section
                className="compare-summary"
                aria-label="Karşılaştırma özeti"
              >
                <div className="compare-summary__intro">
                  <div>
                    <p className="eyebrow">
                      Karşılaştırma özeti
                    </p>

                    <strong>
                      Metrik bazlı öne çıkanlar
                    </strong>
                  </div>

                  <span>
                    Yeni veya birleşik bir skor
                    üretilmez.
                  </span>
                </div>

                <div className="compare-summary__grid">
                  {
                    highlights.map(
                      (
                        highlight,
                      ) => (
                        <div
                          className="compare-highlight"
                          key={
                            highlight.label
                          }
                        >
                          <span>
                            {
                              highlight.label
                            }
                          </span>

                          <strong>
                            {
                              highlight.gridIds.join(
                                " / ",
                              )
                            }
                          </strong>

                          <small>
                            {
                              highlight.value
                            }
                          </small>
                        </div>
                      ),
                    )
                  }
                </div>

                <p className="compare-summary__note">
                  Özet yalnızca karşılaştırılan
                  adayların görünür metriklerinden
                  hesaplanır. Açıklanabilir uygunluk
                  ana karar katmanı olarak kalır; ML
                  desteği ayrı gösterilir.
                </p>
              </section>

              <div className="compare-table-wrap">
              <table className="compare-table">
                <thead>
                  <tr>
                    <th scope="col">
                      Metrik
                    </th>

                    {
                      candidates.map(
                        (
                          candidate,
                        ) => (
                          <th
                            scope="col"
                            key={
                              candidate.grid_id
                            }
                          >
                            <button
                              type="button"
                              className="compare-candidate-link"
                              onClick={
                                () => {
                                  onSelect(
                                    candidate.grid_id,
                                  );
                                }
                              }
                            >
                              <span>
                                #{candidate.selection_rank}
                              </span>
                              <strong>
                                {candidate.grid_id}
                              </strong>
                            </button>

                            <button
                              type="button"
                              className="compare-remove"
                              aria-label={`${candidate.grid_id} adayını karşılaştırmadan çıkar`}
                              onClick={
                                () => {
                                  onRemove(
                                    candidate.grid_id,
                                  );
                                }
                              }
                            >
                              ×
                            </button>
                          </th>
                        ),
                      )
                    }
                  </tr>
                </thead>

                <tbody>
                  {
                    comparisonRows.map(
                      (
                        row,
                      ) => (
                        <tr
                          key={
                            row.label
                          }
                        >
                          <th scope="row">
                            {row.label}
                          </th>

                          {
                            candidates.map(
                              (
                                candidate,
                              ) => (
                                <td
                                  key={
                                    `${row.label}-${candidate.grid_id}`
                                  }
                                >
                                  {row.render(
                                    candidate,
                                  )}
                                </td>
                              ),
                            )
                          }
                        </tr>
                      ),
                    )
                  }

                  <tr>
                    <th scope="row">
                      ML durumu
                    </th>

                    {
                      candidates.map(
                        (
                          candidate,
                        ) => (
                          <td
                            key={`status-${candidate.grid_id}`}
                          >
                            <span
                              className={
                                candidate.ml_support.has_model_disagreement
                                  ? "compare-status compare-status--warning"
                                  : "compare-status compare-status--agreement"
                              }
                            >
                              {
                                candidate.ml_support.has_model_disagreement
                                  ? "Uyuşmazlık"
                                  : "Uyum"
                              }
                            </span>
                          </td>
                        ),
                      )
                    }
                  </tr>
                </tbody>
                </table>
              </div>
            </div>
          )
      }
    </section>
  );
}
