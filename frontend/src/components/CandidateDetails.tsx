import {
  useEffect,
  useState,
} from "react";

import type {
  DecisionSupportCandidate,
} from "../types/api";

import {
  MetricBar,
} from "./MetricBar";

interface CandidateDetailsProps {
  candidate: DecisionSupportCandidate | null;
  isCompared?: boolean;
  compareDisabled?: boolean;
  onToggleCompare?: (gridId: string) => void;
  onCopyLink?: (
    gridId: string,
  ) =>
    | Promise<boolean>
    | boolean;
}

function formatDistance(
  meters: number,
): string {
  return `${(meters / 1000).toFixed(1)} km`;
}

function translateExplanation(
  explanation: string,
): string {
  const translations: Record<string, string> = {
    "strong road accessibility":
      "güçlü yol erişilebilirliği",
    "strong parking feasibility":
      "güçlü otopark uygulanabilirliği",
    "large charging infrastructure gap":
      "yüksek şarj altyapısı açığı",
    "AC/DC technology gap":
      "AC/DC teknoloji açığı",
    "moderate site feasibility":
      "orta düzey saha uygulanabilirliği",
    "moderate infrastructure need":
      "orta düzey altyapı ihtiyacı",
  };

  return explanation
    .split(";")
    .map(
      (part) => {
        const normalized =
          part.trim();

        return (
          translations[normalized] ??
          normalized
        );
      },
    )
    .filter(Boolean)
    .join("; ");
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
      return "Üç model de · ilk %20";

    case "two_of_three_top_20pct":
      return "Üç modelden ikisi · ilk %20";

    case "one_of_three_top_20pct":
      return "Bir model · ilk %20";

    default:
      return "Hiçbiri · ilk %20";
  }
}

export function CandidateDetails({
  candidate,
  isCompared = false,
  compareDisabled = false,
  onToggleCompare,
  onCopyLink,
}: CandidateDetailsProps) {
  const [
    copyStatus,
    setCopyStatus,
  ] =
    useState<
      "idle" |
      "copied" |
      "error"
    >(
      "idle",
    );

  useEffect(
    () => {
      setCopyStatus(
        "idle",
      );
    },
    [
      candidate?.grid_id,
    ],
  );

  if (
    candidate ===
    null
  ) {
    return (
      <aside className="detail-panel detail-panel--empty">
        <p>
          Haritadan veya kısa listeden
          bir aday seçin.
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

  const handleCopyLink =
    async () => {
      if (
        !onCopyLink
      ) {
        return;
      }

      try {
        const copied =
          await onCopyLink(
            candidate.grid_id,
          );

        setCopyStatus(
          copied
            ? "copied"
            : "error",
        );
      } catch {
        setCopyStatus(
          "error",
        );
      }
    };

  return (
    <aside className="detail-panel">
      <div className="detail-hero">
        <div>
          <p className="eyebrow">
            Aday{" "}
            {candidate.selection_rank}
          </p>

          <h2>
            {candidate.grid_id}
          </h2>
        </div>

        <div className="detail-hero__actions">
          <span
            className={`priority-chip priority-chip--${suitability.priority_band.toLowerCase()}`}
          >
            Öncelik{" "}
            {
              suitability.priority_band
            }
          </span>

          {
            onToggleCompare
              ? (
                <button
                  type="button"
                  className={
                    isCompared
                      ? "compare-action compare-action--active"
                      : "compare-action"
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
                    isCompared
                      ? "Karşılaştırmadan çıkar"
                      : compareDisabled
                        ? "Karşılaştırma dolu"
                        : "Karşılaştır"
                  }
                </button>
              )
              : null
          }
          {
            onCopyLink
              ? (
                <button
                  type="button"
                  className="share-action"
                  onClick={
                    () => {
                      void handleCopyLink();
                    }
                  }
                >
                  {
                    copyStatus ===
                    "copied"
                      ? "Kopyalandı"
                      : "Bağlantıyı kopyala"
                  }
                </button>
              )
              : null
          }

          {
            copyStatus ===
            "error"
              ? (
                <span
                  className="share-status share-status--error"
                  role="status"
                  aria-live="polite"
                >
                  Bağlantı kopyalanamadı
                </span>
              )
              : copyStatus ===
                "copied"
                ? (
                  <span
                    className="share-status"
                    role="status"
                    aria-live="polite"
                  >
                    Bağlantı kopyalandı
                  </span>
                )
                : null
          }
        </div>
      </div>

      <div className="primary-score">
        <div>
          <span>
            Açıklanabilir uygunluk
          </span>
          <strong>
            {suitability.score.toFixed(
              2,
            )}
          </strong>
        </div>

        <small>
          Ankara geneli sıra #
          {suitability.rank.toLocaleString()}
        </small>
      </div>

      <section className="detail-section">
        <div className="section-title-row">
          <h3>
            Karar bileşenleri
          </h3>
          <span>
            Ana katman
          </span>
        </div>

        <MetricBar
          label="Uygulanabilirlik"
          value={
            suitability.feasibility
          }
        />

        <MetricBar
          label="İhtiyaç"
          value={
            suitability.need
          }
        />

        <div className="metric-grid">
          <MetricBar
            label="Erişilebilirlik"
            value={
              suitability.accessibility
            }
            compact
          />

          <MetricBar
            label="Otopark"
            value={
              suitability.parking
            }
            compact
          />

          <MetricBar
            label="Altyapı açığı"
            value={
              suitability.infrastructure_gap
            }
            compact
          />

          <MetricBar
            label="Teknoloji açığı"
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
            Tarihsel örüntü ML desteği
          </h3>
          <span className="support-chip">
            {mlSupport.consensus_percentile.toFixed(
              2,
            )}
          </span>
        </div>

        <p className="support-explainer">
          Fold içi normalize edilmiş mekânsal OOF
          yüzdeliği. Birleştirilmiş final skor
          değil, destekleyici kanıttır.
        </p>

        <MetricBar
          label="Lojistik regresyon"
          value={
            mlSupport.logistic_regression_percentile
          }
          compact
        />

        <MetricBar
          label="Random Forest"
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
            Model farkı{" "}
            {mlSupport.model_percentile_spread.toFixed(
              1,
            )}{" "}
            puan
          </span>
        </div>

        {
          mlSupport.has_model_disagreement
            ? (
              <div className="warning-box">
                <strong>
                  Model uyuşmazlığı
                </strong>

                <span>
                  Üç model aynı düzeyde güçlü destek
                  vermiyor. Model yüzdeliklerini
                  ayrı ayrı değerlendirin.
                </span>
              </div>
            )
            : (
              <div className="agreement-box">
                <strong>
                  Modeller arası uyum
                </strong>

                <span>
                  Üç model de bu adayı, adaylar
                  arasında ilk %20'lik dilime
                  yerleştiriyor.
                </span>
              </div>
            )
        }
      </section>

      <section className="detail-section detail-section--facts">
        <h3>
          Mekânsal bağlam
        </h3>

        <dl className="facts-grid">
          <div>
            <dt>
              En yakın kısa liste adayı
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
              Mesafe
            </dt>
            <dd>
              {formatDistance(
                spatialDiversity.nearest_selected_candidate_m,
              )}
            </dd>
          </div>

          <div>
            <dt>
              Boylam
            </dt>
            <dd>
              {candidate.location.longitude.toFixed(
                4,
              )}
            </dd>
          </div>

          <div>
            <dt>
              Enlem
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
          Bu bölge neden öneriliyor?
        </p>
        <p>
          {
            translateExplanation(
              suitability.explanation,
            )
          }
        </p>
      </section>
    </aside>
  );
}
