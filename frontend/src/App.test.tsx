import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import type {
  DecisionSupportCandidate,
  DecisionSupportSummary,
} from "./types/api";
import {
  createCandidate,
} from "./test/fixtures";

vi.mock(
  "./components/MapPanel",
  () => ({
    MapPanel:
      ({
        candidates,
        selectedGridId,
        compareGridIds = [],
        onSelect,
      }: {
        candidates:
          DecisionSupportCandidate[];
        selectedGridId:
          string | null;
        compareGridIds?:
          string[];
        onSelect:
          (
            gridId: string,
          ) => void;
      }) => (
        <section
          data-testid="map-panel-mock"
        >
          <span
            data-testid="map-candidate-count"
          >
            {
              candidates.length
            }
          </span>

          <span
            data-testid="map-selected-grid"
          >
            {
              selectedGridId ??
              "none"
            }
          </span>

          <span
            data-testid="map-compare-count"
          >
            {
              compareGridIds.length
            }
          </span>

          {
            candidates.map(
              (
                candidate,
              ) => (
                <button
                  type="button"
                  key={
                    candidate.grid_id
                  }
                  onClick={
                    () => {
                      onSelect(
                        candidate.grid_id,
                      );
                    }
                  }
                >
                  Harita{" "}
                  {
                    candidate.grid_id
                  }
                </button>
              ),
            )
          }
        </section>
      ),
  }),
);

vi.mock(
  "./services/api",
  () => ({
    getSummary:
      vi.fn(),
    getCandidates:
      vi.fn(),
  }),
);

vi.mock(
  "./utils/candidateExport",
  () => ({
    downloadCandidateCsv:
      vi.fn(),
  }),
);

import {
  getCandidates,
  getSummary,
} from "./services/api";
import {
  downloadCandidateCsv,
} from "./utils/candidateExport";
import App from "./App";

const summary: DecisionSupportSummary = {
  schema_version:
    "1.0",
  study_area:
    "Ankara",
  study_area_country:
    "TR",
  coordinate_reference_system:
    "EPSG:4326",
  candidate_count:
    4,
  generated_at_utc:
    "2026-08-13T00:00:00Z",
  decision_policy: {
    primary_layer:
      "explainable_suitability",
    supporting_layer:
      "fold_normalized_spatial_oof_ml",
    ml_is_blended_into_suitability:
      false,
    minimum_spacing_m:
      25000,
  },
};

function buildCandidates(): DecisionSupportCandidate[] {
  return [
    createCandidate(),
    createCandidate({
      grid_id:
        "ANK_055975",
      selection_rank:
        2,
      suitability: {
        score:
          84.8524,
        rank:
          17,
        priority_band:
          "A",
      },
      ml_support: {
        consensus_percentile:
          98.8315,
        models_top_20pct_count:
          3,
        support_label:
          "all_three_top_20pct",
        all_models_top_20pct:
          true,
        has_model_disagreement:
          false,
        model_percentile_spread:
          2.0,
      },
    }),
    createCandidate({
      grid_id:
        "ANK_007151",
      selection_rank:
        3,
      suitability: {
        score:
          84.1993,
        rank:
          23,
        priority_band:
          "B",
      },
      ml_support: {
        consensus_percentile:
          81.5,
        models_top_20pct_count:
          2,
        all_models_top_20pct:
          false,
        has_model_disagreement:
          true,
        model_percentile_spread:
          31.0,
      },
    }),
    createCandidate({
      grid_id:
        "ANK_013005",
      selection_rank:
        4,
      suitability: {
        score:
          83.3316,
        rank:
          31,
        priority_band:
          "B",
      },
      ml_support: {
        consensus_percentile:
          99.4,
        models_top_20pct_count:
          3,
        support_label:
          "all_three_top_20pct",
        all_models_top_20pct:
          true,
        has_model_disagreement:
          false,
        model_percentile_spread:
          1.0,
      },
    }),
  ];
}

describe(
  "App",
  () => {
    beforeEach(
      () => {
        vi.clearAllMocks();

        Object.defineProperty(
          navigator,
          "clipboard",
          {
            configurable:
              true,
            value: {
              writeText:
                vi
                  .fn()
                  .mockResolvedValue(
                    undefined,
                  ),
            },
          },
        );

        window.history.replaceState(
          {},
          "",
          "/",
        );

        vi.mocked(
          getSummary,
        ).mockResolvedValue(
          summary,
        );

        vi.mocked(
          getCandidates,
        ).mockResolvedValue({
          count:
            4,
          candidates:
            buildCandidates(),
        });
      },
    );

    it(
      "API verisini yükler ve ilk adayı URL ile seçer",
      async () => {
        render(
          <App />,
        );

        expect(
          screen.getByText(
            "VoltSight yükleniyor",
          ),
        ).toBeInTheDocument();

        await screen.findByText(
          "Final adaylar",
        );

        expect(
          screen.getByTestId(
            "map-selected-grid",
          ),
        ).toHaveTextContent(
          "ANK_004300",
        );

        expect(
          window.location.search,
        ).toBe(
          "?candidate=ANK_004300",
        );
      },
    );

    it(
      "URL içindeki geçerli adayı başlangıçta seçer",
      async () => {
        window.history.replaceState(
          {},
          "",
          "/?candidate=ANK_055975",
        );

        render(
          <App />,
        );

        await waitFor(
          () => {
            expect(
              screen.getByTestId(
                "map-selected-grid",
              ),
            ).toHaveTextContent(
              "ANK_055975",
            );
          },
        );

        expect(
          window.location.search,
        ).toBe(
          "?candidate=ANK_055975",
        );
      },
    );

    it(
      "arama ve ML uyuşmazlık filtresini haritaya uygular",
      async () => {
        render(
          <App />,
        );

        await screen.findByText(
          "Final adaylar",
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

        await waitFor(
          () => {
            expect(
              screen.getByTestId(
                "map-candidate-count",
              ),
            ).toHaveTextContent(
              "1",
            );
          },
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
                "",
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

        await waitFor(
          () => {
            expect(
              screen.getByTestId(
                "map-candidate-count",
              ),
            ).toHaveTextContent(
              "2",
            );
          },
        );
      },
    );

    it(
      "ML uzlaşısına göre azalan sıralama yapar",
      async () => {
        const {
          container,
        } =
          render(
            <App />,
          );

        await screen.findByText(
          "Final adaylar",
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

        const orderedGridIds =
          Array.from(
            container.querySelectorAll(
              ".candidate-card__topline > strong",
            ),
          ).map(
            (
              element,
            ) =>
              element.textContent,
          );

        expect(
          orderedGridIds[0],
        ).toBe(
          "ANK_013005",
        );

        expect(
          orderedGridIds[1],
        ).toBe(
          "ANK_055975",
        );
      },
    );

    it(
      "en fazla üç adayı karşılaştırmaya alır",
      async () => {
        render(
          <App />,
        );

        await screen.findByText(
          "Final adaylar",
        );

        for (
          const gridId of [
            "ANK_004300",
            "ANK_055975",
            "ANK_007151",
          ]
        ) {
          fireEvent.click(
            screen.getByRole(
              "button",
              {
                name:
                  `${gridId} adayını karşılaştırmaya ekle`,
              },
            ),
          );
        }

        expect(
          screen.getByTestId(
            "map-compare-count",
          ),
        ).toHaveTextContent(
          "3",
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
          screen.getAllByText(
            "3/3",
          ).length,
        ).toBeGreaterThanOrEqual(
          1,
        );
      },
    );

    it(
      "haritadan aday seçildiğinde URL seçimini günceller",
      async () => {
        render(
          <App />,
        );

        await screen.findByText(
          "Final adaylar",
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Harita ANK_055975",
            },
          ),
        );

        expect(
          screen.getByTestId(
            "map-selected-grid",
          ),
        ).toHaveTextContent(
          "ANK_055975",
        );

        expect(
          window.location.search,
        ).toBe(
          "?candidate=ANK_055975",
        );
      },
    );

    it(
      "seçili adayın paylaşım bağlantısını panoya kopyalar",
      async () => {
        render(
          <App />,
        );

        await screen.findByText(
          "Final adaylar",
        );

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Bağlantıyı kopyala",
            },
          ),
        );

        await waitFor(
          () => {
            expect(
              navigator.clipboard.writeText,
            ).toHaveBeenCalledWith(
              expect.stringContaining(
                "?candidate=ANK_004300",
              ),
            );
          },
        );

        expect(
          await screen.findByText(
            "Bağlantı kopyalandı",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "CSV indirmede yalnızca görünür filtrelenmiş adayları kullanır",
      async () => {
        render(
          <App />,
        );

        await screen.findByText(
          "Final adaylar",
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
          vi.mocked(
            downloadCandidateCsv,
          ),
        ).toHaveBeenCalledOnce();

        const exported =
          vi.mocked(
            downloadCandidateCsv,
          ).mock.calls[0][0];

        expect(
          exported,
        ).toHaveLength(
          1,
        );

        expect(
          exported[0].grid_id,
        ).toBe(
          "ANK_055975",
        );
      },
    );
  },
);
