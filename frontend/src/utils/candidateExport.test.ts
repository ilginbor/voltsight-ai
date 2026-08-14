import {
  describe,
  expect,
  it,
} from "vitest";

import {
  createCandidate,
} from "../test/fixtures";
import {
  buildCandidateCsv,
  downloadCandidateCsv,
} from "./candidateExport";

describe(
  "candidateExport",
  () => {
    it(
      "uygunluk ve ML katmanlarını ayrı CSV sütunlarında tutar",
      () => {
        const csv =
          buildCandidateCsv([
            createCandidate(),
          ]);

        expect(
          csv,
        ).toContain(
          '"suitability_score"',
        );

        expect(
          csv,
        ).toContain(
          '"ml_consensus_percentile"',
        );

        expect(
          csv,
        ).toContain(
          '"ml_model_spread"',
        );

        expect(
          csv,
        ).toContain(
          '"ANK_004300"',
        );

        expect(
          csv,
        ).toContain(
          '"89.7487"',
        );

        expect(
          csv,
        ).toContain(
          '"92.6485"',
        );
      },
    );

    it(
      "iki aday için başlık dahil üç CSV satırı üretir",
      () => {
        const csv =
          buildCandidateCsv([
            createCandidate(),
            createCandidate({
              grid_id:
                "ANK_055975",
              selection_rank:
                2,
            }),
          ]);

        expect(
          csv.split(
            "\n",
          ),
        ).toHaveLength(
          3,
        );
      },
    );

    it(
      "boş aday listesinde indirme başlatmaz",
      () => {
        expect(
          downloadCandidateCsv(
            [],
          ),
        ).toBe(
          false,
        );
      },
    );
  },
);
