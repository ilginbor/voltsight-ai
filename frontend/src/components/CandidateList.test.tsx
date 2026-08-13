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

describe(
  "CandidateList",
  () => {
    it(
      "renders the final candidate cards and selected state",
      () => {
        const candidates = [
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
        ];

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
      "reports the selected grid ID when a candidate is clicked",
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
  },
);

