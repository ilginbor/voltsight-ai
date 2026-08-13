import {
  render,
  screen,
} from "@testing-library/react";
import {
  describe,
  expect,
  it,
} from "vitest";

import {
  createCandidate,
} from "../test/fixtures";
import {
  CandidateDetails,
} from "./CandidateDetails";

describe(
  "CandidateDetails",
  () => {
    it(
      "uygunluk ve ML destek katmanlarını ayrı bölümlerde gösterir",
      () => {
        render(
          <CandidateDetails
            candidate={
              createCandidate()
            }
          />,
        );

        expect(
          screen.getByText(
            "Açıklanabilir uygunluk",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Tarihsel örüntü ML desteği",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "89.75",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getAllByText(
            "92.65",
          ),
        ).toHaveLength(
          2,
        );
      },
    );

    it(
      "modeller arası uyuşmazlığı görünür tutar",
      () => {
        render(
          <CandidateDetails
            candidate={
              createCandidate()
            }
          />,
        );

        expect(
          screen.getByText(
            "Model uyuşmazlığı",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Üç modelden ikisi · ilk %20",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "72.11",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "98.69",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "üç modelin de ilk yüzde 20 içinde olduğu durumda uyumu gösterir",
      () => {
        const candidate =
          createCandidate({
            ml_support: {
              all_models_top_20pct: true,
              has_model_disagreement: false,
              models_top_20pct_count: 3,
              support_label:
                "all_three_top_20pct",
              logistic_regression_percentile: 95.0,
              random_forest_percentile: 96.0,
              hist_gradient_boosting_percentile: 97.0,
              consensus_percentile: 96.0,
              minimum_model_percentile: 95.0,
              maximum_model_percentile: 97.0,
              model_percentile_spread: 2.0,
            },
          });

        render(
          <CandidateDetails
            candidate={
              candidate
            }
          />,
        );

        expect(
          screen.getByText(
            "Modeller arası uyum",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Üç model de · ilk %20",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "aday seçili değilken Türkçe boş durum mesajını gösterir",
      () => {
        render(
          <CandidateDetails
            candidate={
              null
            }
          />,
        );

        expect(
          screen.getByText(
            "Haritadan veya kısa listeden bir aday seçin.",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "aday açıklamasını Türkçeleştirir",
      () => {
        render(
          <CandidateDetails
            candidate={
              createCandidate()
            }
          />,
        );

        expect(
          screen.getByText(
            "güçlü yol erişilebilirliği; yüksek şarj altyapısı açığı; AC/DC teknoloji açığı",
          ),
        ).toBeInTheDocument();
      },
    );
  },
);
