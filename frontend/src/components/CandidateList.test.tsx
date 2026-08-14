import {
  fireEvent,
  render,
  screen,
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
  CandidateList,
} from "./CandidateList";

function buildCandidates() {
  return [
    createCandidate(),
    createCandidate({
      grid_id:
        "ANK_055975",
      selection_rank: 2,
      suitability: {
        score: 84.8524,
        rank: 17,
      },
      ml_support: {
        consensus_percentile: 98.8315,
        models_top_20pct_count: 3,
        support_label:
          "all_three_top_20pct",
        all_models_top_20pct: true,
        has_model_disagreement: false,
      },
    }),
    createCandidate({
      grid_id:
        "ANK_007151",
      selection_rank: 3,
      suitability: {
        score: 84.1993,
        rank: 23,
        priority_band: "B",
      },
      ml_support: {
        consensus_percentile: 81.5,
        models_top_20pct_count: 2,
        has_model_disagreement: true,
      },
    }),
    createCandidate({
      grid_id:
        "ANK_013005",
      selection_rank: 4,
      suitability: {
        score: 83.3316,
        rank: 31,
        priority_band: "B",
      },
      ml_support: {
        consensus_percentile: 99.4,
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
  "CandidateList",
  () => {
    it(
      "final adayları ve seçili aday durumunu gösterir",
      () => {
        const candidates =
          buildCandidates().slice(
            0,
            2,
          );

        render(
          <CandidateList
            candidates={
              candidates
            }
            selectedGridId="ANK_055975"
            onSelect={
              vi.fn()
            }
          />,
        );

        const selected =
          screen.getByRole(
            "button",
            {
              name:
                /ANK_055975/i,
            },
          );

        expect(
          selected,
        ).toHaveClass(
          "candidate-card--selected",
        );

        expect(
          screen.getByText(
            "Final adaylar",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "aday kartına tıklanınca grid kimliğini bildirir",
      () => {
        const onSelect =
          vi.fn();

        render(
          <CandidateList
            candidates={[
              createCandidate(),
            ]}
            selectedGridId={
              null
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
                /ANK_004300/i,
            },
          ),
        );

        expect(
          onSelect,
        ).toHaveBeenCalledOnce();

        expect(
          onSelect,
        ).toHaveBeenCalledWith(
          "ANK_004300",
        );
      },
    );

    it(
      "arama alanındaki değişikliği üst bileşene bildirir",
      () => {
        const onSearchQueryChange =
          vi.fn();

        render(
          <CandidateList
            candidates={
              buildCandidates()
            }
            totalCandidates={
              20
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            searchQuery=""
            onSearchQueryChange={
              onSearchQueryChange
            }
          />,
        );

        fireEvent.change(
          screen.getByRole(
            "searchbox",
            {
              name:
                "Aday ara",
            },
          ),
          {
            target: {
              value:
                "ANK_055975",
            },
          },
        );

        expect(
          onSearchQueryChange,
        ).toHaveBeenCalledWith(
          "ANK_055975",
        );

        expect(
          screen.getByLabelText(
            "4 / 20 aday gösteriliyor",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "sıralama, öncelik ve ML desteği kontrollerini bildirir",
      () => {
        const onSortModeChange =
          vi.fn();

        const onPriorityFilterChange =
          vi.fn();

        const onSupportFilterChange =
          vi.fn();

        render(
          <CandidateList
            candidates={
              buildCandidates()
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            sortMode="selection_rank"
            onSortModeChange={
              onSortModeChange
            }
            priorityFilter="all"
            onPriorityFilterChange={
              onPriorityFilterChange
            }
            supportFilter="all"
            onSupportFilterChange={
              onSupportFilterChange
            }
          />,
        );

        fireEvent.change(
          screen.getByRole(
            "combobox",
            {
              name:
                "Sıralama",
            },
          ),
          {
            target: {
              value:
                "ml_desc",
            },
          },
        );

        fireEvent.change(
          screen.getByRole(
            "combobox",
            {
              name:
                "Öncelik",
            },
          ),
          {
            target: {
              value:
                "B",
            },
          },
        );

        fireEvent.change(
          screen.getByRole(
            "combobox",
            {
              name:
                "ML desteği",
            },
          ),
          {
            target: {
              value:
                "disagreement",
            },
          },
        );

        expect(
          onSortModeChange,
        ).toHaveBeenCalledWith(
          "ml_desc",
        );

        expect(
          onPriorityFilterChange,
        ).toHaveBeenCalledWith(
          "B",
        );

        expect(
          onSupportFilterChange,
        ).toHaveBeenCalledWith(
          "disagreement",
        );
      },
    );

    it(
      "aktif filtreleri sıfırlar",
      () => {
        const onResetFilters =
          vi.fn();

        render(
          <CandidateList
            candidates={
              buildCandidates()
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            searchQuery=""
            onSearchQueryChange={
              vi.fn()
            }
            filtersActive
            onResetFilters={
              onResetFilters
            }
          />,
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Sıfırla",
            },
          ),
        );

        expect(
          onResetFilters,
        ).toHaveBeenCalledOnce();
      },
    );

    it(
      "karşılaştırma düğmesi grid kimliğini bildirir",
      () => {
        const onToggleCompare =
          vi.fn();

        render(
          <CandidateList
            candidates={[
              createCandidate(),
            ]}
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            onToggleCompare={
              onToggleCompare
            }
          />,
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "ANK_004300 adayını karşılaştırmaya ekle",
            },
          ),
        );

        expect(
          onToggleCompare,
        ).toHaveBeenCalledWith(
          "ANK_004300",
        );
      },
    );

    it(
      "karşılaştırma limiti doluyken yeni aday eklemeyi engeller",
      () => {
        render(
          <CandidateList
            candidates={
              buildCandidates()
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            compareGridIds={[
              "ANK_004300",
              "ANK_055975",
              "ANK_007151",
            ]}
            onToggleCompare={
              vi.fn()
            }
            maxCompare={
              3
            }
          />,
        );

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "ANK_013005 adayını karşılaştırmaya ekle",
            },
          ),
        ).toBeDisabled();

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "ANK_004300 adayını karşılaştırmadan çıkar",
            },
          ),
        ).not.toBeDisabled();
      },
    );

    it(
      "boş sonuç durumunu anlaşılır biçimde gösterir",
      () => {
        render(
          <CandidateList
            candidates={
              []
            }
            totalCandidates={
              20
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        expect(
          screen.getByText(
            "Eşleşen aday yok",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Arama veya filtre ayarlarını değiştirin.",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "görünür adayları CSV olarak dışa aktarma işlemini bildirir",
      () => {
        const onDownloadCsv =
          vi.fn();

        render(
          <CandidateList
            candidates={
              buildCandidates()
            }
            totalCandidates={
              20
            }
            selectedGridId={
              null
            }
            onSelect={
              vi.fn()
            }
            searchQuery=""
            onSearchQueryChange={
              vi.fn()
            }
            onDownloadCsv={
              onDownloadCsv
            }
          />,
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "CSV indir",
            },
          ),
        );

        expect(
          onDownloadCsv,
        ).toHaveBeenCalledOnce();
      },
    );
  },
);
