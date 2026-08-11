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
      "shows the explainable suitability and ML support as separate sections",
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
            "Explainable suitability",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Historical-pattern ML support",
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
      "keeps cross-model disagreement visible",
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
            "Model disagreement",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "Two of three models · top 20%",
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
      "shows agreement when all three models are in the top 20 percent",
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
            "Cross-model agreement",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "All three models · top 20%",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "renders an empty-state instruction without a selected candidate",
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
            "Select a candidate on the map or shortlist.",
          ),
        ).toBeInTheDocument();
      },
    );
  },
);
