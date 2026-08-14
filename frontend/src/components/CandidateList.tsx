import type {
  DecisionSupportCandidate,
} from "../types/api";

export type CandidateSortMode =
  | "selection_rank"
  | "suitability_desc"
  | "ml_desc"
  | "support_desc"
  | "disagreement_first";

export type CandidatePriorityFilter =
  | "all"
  | "A"
  | "B"
  | "C"
  | "D"
  | "E";

export type CandidateSupportFilter =
  | "all"
  | "all_three"
  | "two_plus"
  | "disagreement";

interface CandidateListProps {
  candidates: DecisionSupportCandidate[];
  totalCandidates?: number;
  selectedGridId: string | null;
  onSelect: (gridId: string) => void;
  searchQuery?: string;
  onSearchQueryChange?: (value: string) => void;
  sortMode?: CandidateSortMode;
  onSortModeChange?: (value: CandidateSortMode) => void;
  priorityFilter?: CandidatePriorityFilter;
  onPriorityFilterChange?: (
    value: CandidatePriorityFilter,
  ) => void;
  supportFilter?: CandidateSupportFilter;
  onSupportFilterChange?: (
    value: CandidateSupportFilter,
  ) => void;
  compareGridIds?: string[];
  onToggleCompare?: (gridId: string) => void;
  maxCompare?: number;
  filtersActive?: boolean;
  onResetFilters?: () => void;
}

function supportText(
  candidate: DecisionSupportCandidate,
): string {
  const count =
    candidate.ml_support.models_top_20pct_count;

  return `${count}/3 model · ilk %20`;
}

function disagreementText(
  candidate: DecisionSupportCandidate,
): string {
  return candidate.ml_support.has_model_disagreement
    ? `Model farkı ${candidate.ml_support.model_percentile_spread.toFixed(1)} puan`
    : "Modeller arası uyum";
}

export function CandidateList({
  candidates,
  totalCandidates = candidates.length,
  selectedGridId,
  onSelect,
  searchQuery = "",
  onSearchQueryChange,
  sortMode = "selection_rank",
  onSortModeChange,
  priorityFilter = "all",
  onPriorityFilterChange,
  supportFilter = "all",
  onSupportFilterChange,
  compareGridIds = [],
  onToggleCompare,
  maxCompare = 3,
  filtersActive = false,
  onResetFilters,
}: CandidateListProps) {
  const controlsEnabled =
    Boolean(
      onSearchQueryChange ||
      onSortModeChange ||
      onPriorityFilterChange ||
      onSupportFilterChange,
    );

  return (
    <aside className="candidate-list-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">
            Mekânsal kısa liste
          </p>
          <h2>Final adaylar</h2>
        </div>

        <span className="count-badge">
          {candidates.length}
        </span>
      </div>

      {
        controlsEnabled
          ? (
            <div className="candidate-controls">
              <label className="candidate-search">
                <span>
                  Aday ara
                </span>
                <input
                  type="search"
                  value={
                    searchQuery
                  }
                  placeholder="ANK_004300"
                  onChange={
                    (
                      event,
                    ) => {
                      onSearchQueryChange?.(
                        event.target.value,
                      );
                    }
                  }
                />
              </label>

              <div className="candidate-control-grid">
                <label>
                  <span>
                    Sıralama
                  </span>
                  <select
                    value={
                      sortMode
                    }
                    onChange={
                      (
                        event,
                      ) => {
                        onSortModeChange?.(
                          event.target.value as CandidateSortMode,
                        );
                      }
                    }
                  >
                    <option value="selection_rank">
                      Kısa liste sırası
                    </option>
                    <option value="suitability_desc">
                      Uygunluk
                    </option>
                    <option value="ml_desc">
                      ML uzlaşısı
                    </option>
                    <option value="support_desc">
                      Model desteği
                    </option>
                    <option value="disagreement_first">
                      Uyuşmazlık
                    </option>
                  </select>
                </label>

                <label>
                  <span>
                    Öncelik
                  </span>
                  <select
                    value={
                      priorityFilter
                    }
                    onChange={
                      (
                        event,
                      ) => {
                        onPriorityFilterChange?.(
                          event.target.value as CandidatePriorityFilter,
                        );
                      }
                    }
                  >
                    <option value="all">
                      Tüm bantlar
                    </option>
                    <option value="A">
                      Bant A
                    </option>
                    <option value="B">
                      Bant B
                    </option>
                    <option value="C">
                      Bant C
                    </option>
                    <option value="D">
                      Bant D
                    </option>
                    <option value="E">
                      Bant E
                    </option>
                  </select>
                </label>

                <label className="candidate-control-grid__wide">
                  <span>
                    ML desteği
                  </span>
                  <select
                    value={
                      supportFilter
                    }
                    onChange={
                      (
                        event,
                      ) => {
                        onSupportFilterChange?.(
                          event.target.value as CandidateSupportFilter,
                        );
                      }
                    }
                  >
                    <option value="all">
                      Tüm adaylar
                    </option>
                    <option value="all_three">
                      3 modelin tamamı ilk %20
                    </option>
                    <option value="two_plus">
                      En az 2 model ilk %20
                    </option>
                    <option value="disagreement">
                      Yalnızca model uyuşmazlığı
                    </option>
                  </select>
                </label>
              </div>

              <div className="candidate-filter-summary">
                <span
                  aria-label={`${candidates.length} / ${totalCandidates} aday gösteriliyor`}
                >
                  <strong>
                    {candidates.length}
                  </strong>
                  {" / "}
                  <strong>
                    {totalCandidates}
                  </strong>
                  {" aday gösteriliyor"}
                </span>

                {
                  filtersActive &&
                  onResetFilters
                    ? (
                      <button
                        type="button"
                        className="text-button"
                        onClick={
                          onResetFilters
                        }
                      >
                        Sıfırla
                      </button>
                    )
                    : null
                }
              </div>
            </div>
          )
          : null
      }

      <div className="candidate-list">
        {
          candidates.length ===
          0
            ? (
              <div className="candidate-empty">
                <strong>
                  Eşleşen aday yok
                </strong>
                <span>
                  Arama veya filtre ayarlarını
                  değiştirin.
                </span>
              </div>
            )
            : candidates.map(
                (
                  candidate,
                ) => {
                  const selected =
                    candidate.grid_id ===
                    selectedGridId;

                  const compared =
                    compareGridIds.includes(
                      candidate.grid_id,
                    );

                  const compareDisabled =
                    !compared &&
                    compareGridIds.length >=
                      maxCompare;

                  return (
                    <article
                      className={
                        selected
                          ? "candidate-card candidate-card--selected"
                          : "candidate-card"
                      }
                      key={
                        candidate.grid_id
                      }
                    >
                      <button
                        type="button"
                        className={
                          selected
                            ? "candidate-card__main candidate-card--selected"
                            : "candidate-card__main"
                        }
                        onClick={
                          () => {
                            onSelect(
                              candidate.grid_id,
                            );
                          }
                        }
                      >
                        <div className="candidate-card__rank">
                          {
                            candidate.selection_rank
                          }
                        </div>

                        <div className="candidate-card__content">
                          <div className="candidate-card__topline">
                            <strong>
                              {
                                candidate.grid_id
                              }
                            </strong>

                            <span
                              className={`priority-chip priority-chip--${candidate.suitability.priority_band.toLowerCase()}`}
                            >
                              Bant{" "}
                              {
                                candidate
                                  .suitability
                                  .priority_band
                              }
                            </span>
                          </div>

                          <div className="candidate-card__metrics">
                            <span>
                              Uygunluk{" "}
                              <strong>
                                {
                                  candidate.suitability.score.toFixed(
                                    1,
                                  )
                                }
                              </strong>
                            </span>

                            <span>
                              ML{" "}
                              <strong>
                                {
                                  candidate.ml_support.consensus_percentile.toFixed(
                                    1,
                                  )
                                }
                              </strong>
                            </span>
                          </div>

                          <small>
                            {supportText(
                              candidate,
                            )}
                          </small>

                          <small
                            className={
                              candidate.ml_support.has_model_disagreement
                                ? "candidate-card__status candidate-card__status--warning"
                                : "candidate-card__status candidate-card__status--agreement"
                            }
                          >
                            {disagreementText(
                              candidate,
                            )}
                          </small>
                        </div>
                      </button>

                      {
                        onToggleCompare
                          ? (
                            <button
                              type="button"
                              className={
                                compared
                                  ? "candidate-card__compare candidate-card__compare--active"
                                  : "candidate-card__compare"
                              }
                              aria-pressed={
                                compared
                              }
                              aria-label={
                                compared
                                  ? `${candidate.grid_id} adayını karşılaştırmadan çıkar`
                                  : `${candidate.grid_id} adayını karşılaştırmaya ekle`
                              }
                              title={
                                compareDisabled
                                  ? `En fazla ${maxCompare} aday karşılaştırılabilir`
                                  : compared
                                    ? "Karşılaştırmadan çıkar"
                                    : "Karşılaştırmaya ekle"
                              }
                              disabled={
                                compareDisabled
                              }
                              onClick={
                                () => {
                                  onToggleCompare(
                                    candidate.grid_id,
                                  );
                                }
                              }
                            >
                              {
                                compared
                                  ? "✓"
                                  : "+"
                              }
                            </button>
                          )
                          : null
                      }
                    </article>
                  );
                },
              )
        }
      </div>
    </aside>
  );
}
