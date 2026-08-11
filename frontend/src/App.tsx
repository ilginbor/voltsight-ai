import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  CandidateDetails,
} from "./components/CandidateDetails";
import {
  CandidateList,
} from "./components/CandidateList";
import {
  MapPanel,
} from "./components/MapPanel";
import {
  getCandidates,
  getSummary,
} from "./services/api";
import type {
  DecisionSupportCandidate,
  DecisionSupportSummary,
} from "./types/api";

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

          setSelectedGridId(
            candidateResponse
              .candidates[
                0
              ]
              ?.grid_id ??
              null,
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
              : "VoltSight data could not be loaded.",
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
      },
      [],
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

  if (
    loading
  ) {
    return (
      <main className="state-page">
        <div className="state-card">
          <div className="loader" />
          <h1>
            Loading VoltSight
          </h1>
          <p>
            Fetching the Ankara
            decision-support shortlist.
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
            API unavailable
          </h1>
          <p>
            {error}
          </p>
          <small>
            Start FastAPI on
            127.0.0.1:8000 and refresh
            this page.
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
              Ankara EV Charging
              Decision Support
            </p>
          </div>
        </div>

        <div className="header-metrics">
          <div>
            <span>
              Final shortlist
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
              Minimum spacing
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
              ML policy
            </span>
            <strong>
              Supporting evidence
            </strong>
          </div>
        </div>
      </header>

      <section className="policy-banner">
        <strong>
          Decision architecture
        </strong>
        <span>
          Explainable suitability remains
          the primary ranking. ML
          percentiles show agreement with
          historical station-placement
          patterns and are not blended into
          suitability.
        </span>
      </section>

      <main className="dashboard-grid">
        <CandidateList
          candidates={
            candidates
          }
          selectedGridId={
            selectedGridId
          }
          onSelect={
            handleSelect
          }
        />

        <MapPanel
          candidates={
            candidates
          }
          selectedGridId={
            selectedGridId
          }
          onSelect={
            handleSelect
          }
        />

        <CandidateDetails
          candidate={
            selectedCandidate
          }
        />
      </main>

      <footer className="app-footer">
        <span>
          Dataset schema{" "}
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
          Internal spatial validation ·
          not a construction probability
        </span>
      </footer>
    </div>
  );
}

export default App;
