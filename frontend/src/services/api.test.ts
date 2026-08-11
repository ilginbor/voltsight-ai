import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  getCandidate,
  getCandidates,
  getSummary,
} from "./api";

function mockFetchResponse(
  body: unknown,
  options: {
    ok?: boolean;
    status?: number;
  } = {},
): void {
  const {
    ok = true,
    status = 200,
  } = options;

  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      status,
      json:
        vi.fn().mockResolvedValue(
          body,
        ),
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe(
  "VoltSight API client",
  () => {
    it(
      "requests the candidate list through the versioned API path",
      async () => {
        mockFetchResponse({
          count: 0,
          candidates: [],
        });

        await getCandidates();

        expect(
          fetch,
        ).toHaveBeenCalledWith(
          "/api/v1/candidates",
          expect.objectContaining({
            headers: {
              Accept:
                "application/json",
            },
          }),
        );
      },
    );

    it(
      "requests summary metadata",
      async () => {
        mockFetchResponse({
          schema_version: "1.0",
        });

        await getSummary();

        expect(
          fetch,
        ).toHaveBeenCalledWith(
          "/api/v1/summary",
          expect.any(
            Object,
          ),
        );
      },
    );

    it(
      "encodes candidate IDs in detail requests",
      async () => {
        mockFetchResponse({
          grid_id:
            "ANK TEST",
        });

        await getCandidate(
          "ANK TEST",
        );

        expect(
          fetch,
        ).toHaveBeenCalledWith(
          "/api/v1/candidates/ANK%20TEST",
          expect.any(
            Object,
          ),
        );
      },
    );

    it(
      "surfaces the backend detail message for failed requests",
      async () => {
        mockFetchResponse(
          {
            detail:
              "Candidate not found: ANK_999999",
          },
          {
            ok: false,
            status: 404,
          },
        );

        await expect(
          getCandidate(
            "ANK_999999",
          ),
        ).rejects.toThrow(
          "Candidate not found: ANK_999999",
        );
      },
    );
  },
);
