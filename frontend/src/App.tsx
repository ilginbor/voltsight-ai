import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  CandidateCompare,
} from "./components/CandidateCompare";
import {
  CandidateDetails,
} from "./components/CandidateDetails";
import {
  CandidateList,
} from "./components/CandidateList";
import type {
  CandidatePriorityFilter,
  CandidateSortMode,
  CandidateSupportFilter,
} from "./components/CandidateList";
import {
  getCandidates,
  getSummary,
} from "./services/api";
import type {
  DecisionSupportCandidate,
  DecisionSupportSummary,
} from "./types/api";
import {
  downloadCandidateCsv,
} from "./utils/candidateExport";

const MapPanel =
  lazy(
    async () => {
      const module =
        await import(
          "./components/MapPanel"
        );

      return {
        default:
          module.MapPanel,
      };
    },
  );

function MapPanelFallback() {
  return (
    <section
      className="map-panel map-panel--loading"
      aria-busy="true"
      aria-label="Aday haritası yükleniyor"
    >
      <div className="map-panel__header">
        <div>
          <p className="eyebrow">
            Ankara · Türkiye
          </p>

          <h2>
            Aday haritası
          </h2>
        </div>
      </div>

      <div className="map-lazy-fallback">
        <div
          className="loader"
          aria-hidden="true"
        />

        <strong>
          Harita yükleniyor
        </strong>

        <span>
          Harita bileşeni ayrı bir paket
          olarak yükleniyor.
        </span>
      </div>
    </section>
  );
}

const MAX_COMPARE_CANDIDATES = 3;

function readCandidateFromUrl(): string | null {
  const params = new URLSearchParams(
    window.location.search,
  );

  return params.get("candidate");
}

function writeCandidateToUrl(
  gridId: string | null,
) {
  const url = new URL(
    window.location.href,
  );

  if (gridId) {
    url.searchParams.set(
      "candidate",
      gridId,
    );
  } else {
    url.searchParams.delete(
      "candidate",
    );
  }

  window.history.replaceState(
    {},
    "",
    url,
  );
}

function App() {
  const [
    summary,
    setSummary,
  ] =
    useState<DecisionSupportSummary | null>(
      null,
    );

  const [
    candidates,
    setCandidates,
  ] =
    useState<DecisionSupportCandidate[]>(
      [],
    );

  const [
    selectedGridId,
    setSelectedGridId,
  ] =
    useState<string | null>(
      null,
    );

  const [
    compareGridIds,
    setCompareGridIds,
  ] =
    useState<string[]>(
      [],
    );

  const [
    searchQuery,
    setSearchQuery,
  ] =
    useState("");

  const [
    sortMode,
    setSortMode,
  ] =
    useState<CandidateSortMode>(
      "selection_rank",
    );

  const [
    priorityFilter,
    setPriorityFilter,
  ] =
    useState<CandidatePriorityFilter>(
      "all",
    );

  const [
    supportFilter,
    setSupportFilter,
  ] =
    useState<CandidateSupportFilter>(
      "all",
    );

  const [
    loading,
    setLoading,
  ] =
    useState(
      true,
    );

  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null,
    );

  useEffect(
    () => {
      const controller =
        new AbortController();

      async function load() {
        try {
          setLoading(
            true,
          );

          setError(
            null,
          );

          const [
            summaryResponse,
            candidateResponse,
          ] =
            await Promise.all([
              getSummary(
                controller.signal,
              ),
              getCandidates(
                controller.signal,
              ),
            ]);

          setSummary(
            summaryResponse,
          );

          setCandidates(
            candidateResponse.candidates,
          );

          const requestedGridId =
            readCandidateFromUrl();

          const requestedCandidate =
            requestedGridId
              ? candidateResponse.candidates.find(
                  (candidate) =>
                    candidate.grid_id.toLowerCase() ===
                    requestedGridId.toLowerCase(),
                )
              : undefined;

          const initialGridId =
            requestedCandidate?.grid_id ??
            candidateResponse
              .candidates[0]
              ?.grid_id ??
            null;

          setSelectedGridId(
            initialGridId,
          );

          writeCandidateToUrl(
            initialGridId,
          );
        } catch (
          requestError
        ) {
          if (
            controller.signal.aborted
          ) {
            return;
          }

          setError(
            requestError instanceof Error
              ? requestError.message
              : "VoltSight verileri yüklenemedi.",
          );
        } finally {
          if (
            !controller.signal.aborted
          ) {
            setLoading(
              false,
            );
          }
        }
      }

      void load();

      return () => {
        controller.abort();
      };
    },
    [],
  );

  const handleSelect =
    useCallback(
      (
        gridId: string,
      ) => {
        setSelectedGridId(
          gridId,
        );

        writeCandidateToUrl(
          gridId,
        );
      },
      [],
    );

  const handleToggleCompare =
    useCallback(
      (
        gridId: string,
      ) => {
        setCompareGridIds(
          (
            current,
          ) => {
            if (
              current.includes(
                gridId,
              )
            ) {
              return current.filter(
                (
                  currentGridId,
                ) =>
                  currentGridId !==
                  gridId,
              );
            }

            if (
              current.length >=
              MAX_COMPARE_CANDIDATES
            ) {
              return current;
            }

            return [
              ...current,
              gridId,
            ];
          },
        );
      },
      [],
    );

  const handleClearCompare =
    useCallback(
      () => {
        setCompareGridIds(
          [],
        );
      },
      [],
    );

  const handleResetFilters =
    useCallback(
      () => {
        setSearchQuery(
          "",
        );

        setSortMode(
          "selection_rank",
        );

        setPriorityFilter(
          "all",
        );

        setSupportFilter(
          "all",
        );
      },
      [],
    );


  const handleCopyCandidateLink =
    useCallback(
      async (
        gridId: string,
      ): Promise<boolean> => {
        if (
          !navigator.clipboard
            ?.writeText
        ) {
          return false;
        }

        const url =
          new URL(
            window.location.href,
          );

        url.searchParams.set(
          "candidate",
          gridId,
        );

        try {
          await navigator.clipboard.writeText(
            url.toString(),
          );

          return true;
        } catch {
          return false;
        }
      },
      [],
    );

  const visibleCandidates =
    useMemo(
      () => {
        const normalizedQuery =
          searchQuery
            .trim()
            .toLowerCase();

        const filtered =
          candidates.filter(
            (
              candidate,
            ) => {
              if (
                normalizedQuery &&
                !candidate.grid_id
                  .toLowerCase()
                  .includes(
                    normalizedQuery,
                  )
              ) {
                return false;
              }

              if (
                priorityFilter !==
                  "all" &&
                candidate.suitability
                  .priority_band !==
                  priorityFilter
              ) {
                return false;
              }

              if (
                supportFilter ===
                  "all_three" &&
                !candidate.ml_support
                  .all_models_top_20pct
              ) {
                return false;
              }

              if (
                supportFilter ===
                  "two_plus" &&
                !candidate.ml_support
                  .at_least_two_models_top_20pct
              ) {
                return false;
              }

              if (
                supportFilter ===
                  "disagreement" &&
                !candidate.ml_support
                  .has_model_disagreement
              ) {
                return false;
              }

              return true;
            },
          );

        return [
          ...filtered,
        ].sort(
          (
            left,
            right,
          ) => {
            switch (
              sortMode
            ) {
              case "suitability_desc":
                return (
                  right.suitability
                    .score -
                    left.suitability
                      .score ||
                  left.selection_rank -
                    right.selection_rank
                );

              case "ml_desc":
                return (
                  right.ml_support
                    .consensus_percentile -
                    left.ml_support
                      .consensus_percentile ||
                  left.selection_rank -
                    right.selection_rank
                );

              case "support_desc":
                return (
                  right.ml_support
                    .models_top_20pct_count -
                    left.ml_support
                      .models_top_20pct_count ||
                  right.ml_support
                    .consensus_percentile -
                    left.ml_support
                      .consensus_percentile ||
                  left.selection_rank -
                    right.selection_rank
                );

              case "disagreement_first":
                return (
                  Number(
                    right.ml_support
                      .has_model_disagreement,
                  ) -
                    Number(
                      left.ml_support
                        .has_model_disagreement,
                    ) ||
                  right.ml_support
                    .model_percentile_spread -
                    left.ml_support
                      .model_percentile_spread ||
                  left.selection_rank -
                    right.selection_rank
                );

              case "selection_rank":
              default:
                return (
                  left.selection_rank -
                  right.selection_rank
                );
            }
          },
        );
      },
      [
        candidates,
        priorityFilter,
        searchQuery,
        sortMode,
        supportFilter,
      ],
    );

  useEffect(
    () => {
      if (
        loading
      ) {
        return;
      }

      if (
        visibleCandidates.length ===
        0
      ) {
        if (
          selectedGridId !==
          null
        ) {
          setSelectedGridId(
            null,
          );

          writeCandidateToUrl(
            null,
          );
        }

        return;
      }

      const selectedIsVisible =
        visibleCandidates.some(
          (
            candidate,
          ) =>
            candidate.grid_id ===
            selectedGridId,
        );

      if (
        !selectedIsVisible
      ) {
        handleSelect(
          visibleCandidates[0]
            .grid_id,
        );
      }
    },
    [
      handleSelect,
      loading,
      selectedGridId,
      visibleCandidates,
    ],
  );

  useEffect(
    () => {
      const handlePopState =
        () => {
          const requestedGridId =
            readCandidateFromUrl();

          if (
            requestedGridId ===
            null
          ) {
            return;
          }

          const matchingCandidate =
            candidates.find(
              (
                candidate,
              ) =>
                candidate.grid_id
                  .toLowerCase() ===
                requestedGridId.toLowerCase(),
            );

          if (
            matchingCandidate
          ) {
            setSelectedGridId(
              matchingCandidate.grid_id,
            );
          }
        };

      window.addEventListener(
        "popstate",
        handlePopState,
      );

      return () => {
        window.removeEventListener(
          "popstate",
          handlePopState,
        );
      };
    },
    [
      candidates,
    ],
  );

  const selectedCandidate =
    useMemo(
      () =>
        candidates.find(
          (
            candidate,
          ) =>
            candidate.grid_id ===
            selectedGridId,
        ) ??
        null,
      [
        candidates,
        selectedGridId,
      ],
    );

  const compareCandidates =
    useMemo(
      () =>
        compareGridIds
          .map(
            (
              gridId,
            ) =>
              candidates.find(
                (
                  candidate,
                ) =>
                  candidate.grid_id ===
                  gridId,
              ),
          )
          .filter(
            (
              candidate,
            ): candidate is DecisionSupportCandidate =>
              candidate !==
              undefined,
          ),
      [
        candidates,
        compareGridIds,
      ],
    );

  const filtersActive =
    Boolean(
      searchQuery.trim(),
    ) ||
    sortMode !==
      "selection_rank" ||
    priorityFilter !==
      "all" ||
    supportFilter !==
      "all";


  const handleDownloadCsv =
    useCallback(
      () => {
        downloadCandidateCsv(
          visibleCandidates,
        );
      },
      [
        visibleCandidates,
      ],
    );

  if (
    loading
  ) {
    return (
      <main className="state-page">
        <div className="state-card">
          <div className="loader" />
          <h1>
            VoltSight yükleniyor
          </h1>
          <p>
            Ankara karar destek kısa listesi
            yükleniyor.
          </p>
        </div>
      </main>
    );
  }

  if (
    error !==
    null
  ) {
    return (
      <main className="state-page">
        <div className="state-card state-card--error">
          <span className="state-icon">
            !
          </span>
          <h1>
            API kullanılamıyor
          </h1>
          <p>
            {error}
          </p>
          <small>
            FastAPI'yi 127.0.0.1:8000
            adresinde başlatın ve
            sayfayı yenileyin.
          </small>
        </div>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark">
            V
          </div>

          <div>
            <p className="brand-name">
              VoltSight
            </p>
            <p className="brand-subtitle">
              Ankara EV Şarj
              Karar Destek Sistemi
            </p>
          </div>
        </div>

        <div className="header-metrics">
          <div>
            <span>
              Final kısa liste
            </span>
            <strong>
              {
                summary?.candidate_count ??
                candidates.length
              }
            </strong>
          </div>

          <div>
            <span>
              Minimum aralık
            </span>
            <strong>
              {
                summary
                  ? `${summary.decision_policy.minimum_spacing_m / 1000} km`
                  : "—"
              }
            </strong>
          </div>

          <div>
            <span>
              Karşılaştırma
            </span>
            <strong>
              {compareCandidates.length}/
              {MAX_COMPARE_CANDIDATES}
            </strong>
          </div>

          <div>
            <span>
              ML politikası
            </span>
            <strong>
              Destekleyici kanıt
            </strong>
          </div>
        </div>
      </header>

      <section className="policy-banner">
        <strong>
          Karar mimarisi
        </strong>
        <span>
          Açıklanabilir uygunluk skoru ana
          sıralama katmanıdır. ML yüzdelikleri,
          tarihsel istasyon yerleşim örüntüleriyle
          uyumu gösterir ve uygunluk skoruna
          karıştırılmaz.
        </span>
      </section>

      {
        compareCandidates.length >
        0
          ? (
            <CandidateCompare
              candidates={
                compareCandidates
              }
              onClear={
                handleClearCompare
              }
              onRemove={
                handleToggleCompare
              }
              onSelect={
                handleSelect
              }
            />
          )
          : null
      }

      <main className="dashboard-grid">
        <CandidateList
          candidates={
            visibleCandidates
          }
          totalCandidates={
            candidates.length
          }
          selectedGridId={
            selectedGridId
          }
          onSelect={
            handleSelect
          }
          searchQuery={
            searchQuery
          }
          onSearchQueryChange={
            setSearchQuery
          }
          sortMode={
            sortMode
          }
          onSortModeChange={
            setSortMode
          }
          priorityFilter={
            priorityFilter
          }
          onPriorityFilterChange={
            setPriorityFilter
          }
          supportFilter={
            supportFilter
          }
          onSupportFilterChange={
            setSupportFilter
          }
          compareGridIds={
            compareGridIds
          }
          onToggleCompare={
            handleToggleCompare
          }
          maxCompare={
            MAX_COMPARE_CANDIDATES
          }
          filtersActive={
            filtersActive
          }
          onResetFilters={
            handleResetFilters
          }
          onDownloadCsv={
            handleDownloadCsv
          }
        />

        <Suspense
          fallback={
            <MapPanelFallback />
          }
        >
          <MapPanel
            candidates={
              visibleCandidates
            }
            selectedGridId={
              selectedGridId
            }
            compareGridIds={
              compareGridIds
            }
            onSelect={
              handleSelect
            }
          />
        </Suspense>

        <CandidateDetails
          candidate={
            selectedCandidate
          }
          isCompared={
            selectedCandidate
              ? compareGridIds.includes(
                  selectedCandidate.grid_id,
                )
              : false
          }
          compareDisabled={
            Boolean(
              selectedCandidate &&
              !compareGridIds.includes(
                selectedCandidate.grid_id,
              ) &&
              compareGridIds.length >=
                MAX_COMPARE_CANDIDATES,
            )
          }
          onToggleCompare={
            handleToggleCompare
          }
          onCopyLink={
            handleCopyCandidateLink
          }
        />
      </main>

      <footer className="app-footer">
        <span>
          Veri şeması{" "}
          {
            summary?.schema_version ??
            "—"
          }
        </span>

        <span>
          {
            summary?.coordinate_reference_system ??
            "EPSG:4326"
          }
        </span>

        <span>
          Seçili aday paylaşım için
          URL'ye yansıtılır
        </span>

        <span>
          İç mekânsal doğrulama ·
          kurulum olasılığı değildir
        </span>
      </footer>
    </div>
  );
}

export default App;
