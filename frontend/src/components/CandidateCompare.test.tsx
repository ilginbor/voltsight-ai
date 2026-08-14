import {
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import {
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  createCandidate,
} from "../test/fixtures";
import {
  CandidateCompare,
} from "./CandidateCompare";

function buildComparisonCandidates() {
  return [
    createCandidate(),
    createCandidate({
      grid_id:
        "ANK_055975",
      selection_rank: 2,
      suitability: {
        score: 84.8524,
        feasibility: 81.4,
        need: 88.6,
      },
      ml_support: {
        consensus_percentile: 98.8315,
        logistic_regression_percentile: 97.8,
        random_forest_percentile: 99.1,
        hist_gradient_boosting_percentile: 98.6,
        model_percentile_spread: 1.3,
        models_top_20pct_count: 3,
        support_label:
          "all_three_top_20pct",
        all_models_top_20pct: true,
        has_model_disagreement: false,
      },
    }),
  ];
}

describe(
  "CandidateCompare",
  () => {
    it(
      "tek aday seçiliyken ikinci aday ipucunu gösterir",
      () => {
        render(
          <CandidateCompare
            candidates={[
              createCandidate(),
            ]}
            onRemove={
              vi.fn()
            }
            onClear={
              vi.fn()
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        expect(
          screen.getByText(
            "1/3 seçili",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Yan yana metrikleri görmek için bir aday daha ekleyin.",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "iki adayı yan yana karşılaştırır",
      () => {
        render(
          <CandidateCompare
            candidates={
              buildComparisonCandidates()
            }
            onRemove={
              vi.fn()
            }
            onClear={
              vi.fn()
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        expect(
          screen.getByRole(
            "table",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Uygunluk",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "ML uzlaşısı",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Uyuşmazlık",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Uyum",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "aday başlığına tıklanınca adayı seçer",
      () => {
        const onSelect =
          vi.fn();

        render(
          <CandidateCompare
            candidates={
              buildComparisonCandidates()
            }
            onRemove={
              vi.fn()
            }
            onClear={
              vi.fn()
            }
            onSelect={
              onSelect
            }
          />,
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                /^#\s*2\s*ANK_055975$/i,
            },
          ),
        );

        expect(
          onSelect,
        ).toHaveBeenCalledWith(
          "ANK_055975",
        );
      },
    );

    it(
      "aday kaldırma ve temizleme işlemlerini bildirir",
      () => {
        const onRemove =
          vi.fn();

        const onClear =
          vi.fn();

        render(
          <CandidateCompare
            candidates={
              buildComparisonCandidates()
            }
            onRemove={
              onRemove
            }
            onClear={
              onClear
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "ANK_004300 adayını karşılaştırmadan çıkar",
            },
          ),
        );

        expect(
          onRemove,
        ).toHaveBeenCalledWith(
          "ANK_004300",
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Temizle",
            },
          ),
        );

        expect(
          onClear,
        ).toHaveBeenCalledOnce();
      },
    );

    it(
      "karşılaştırma için deterministik metrik özetini gösterir",
      () => {
        render(
          <CandidateCompare
            candidates={
              buildComparisonCandidates()
            }
            onRemove={
              vi.fn()
            }
            onClear={
              vi.fn()
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        const summary =
          screen.getByLabelText(
            "Karşılaştırma özeti",
          );

        expect(
          within(
            summary,
          ).getByText(
            "En yüksek uygunluk",
          ),
        ).toBeInTheDocument();

        expect(
          within(
            summary,
          ).getByText(
            "En yüksek ML uzlaşısı",
          ),
        ).toBeInTheDocument();

        expect(
          within(
            summary,
          ).getByText(
            "En düşük model farkı",
          ),
        ).toBeInTheDocument();

        expect(
          within(
            summary,
          ).getAllByText(
            "ANK_004300",
          ).length,
        ).toBeGreaterThan(
          0,
        );

        expect(
          within(
            summary,
          ).getAllByText(
            "ANK_055975",
          ).length,
        ).toBeGreaterThan(
          0,
        );

        expect(
          within(
            summary,
          ).getByText(
            /Yeni veya birleşik bir skor üretilmez/i,
          ),
        ).toBeInTheDocument();
      },
    );
  },
);
