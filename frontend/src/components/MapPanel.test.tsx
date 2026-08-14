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

import {
  createCandidate,
} from "../test/fixtures";

const olMocks =
  vi.hoisted(
    () => ({
      fit:
        vi.fn(),
      animate:
        vi.fn(),
      changed:
        vi.fn(),
      setTarget:
        vi.fn(),
    }),
  );

vi.mock(
  "ol/Feature",
  () => ({
    default:
      class MockFeature {
        private values =
          new Map<
            string,
            unknown
          >();

        set(
          key: string,
          value: unknown,
        ) {
          this.values.set(
            key,
            value,
          );
        }

        get(
          key: string,
        ) {
          return this.values.get(
            key,
          );
        }
      },
  }),
);

vi.mock(
  "ol/geom/Point",
  () => ({
    default:
      class MockPoint {},
  }),
);

vi.mock(
  "ol/proj",
  () => ({
    fromLonLat:
      (
        coordinates: number[],
      ) =>
        coordinates,
  }),
);

vi.mock(
  "ol/source/OSM",
  () => ({
    default:
      class MockOSM {},
  }),
);

vi.mock(
  "ol/source/Vector",
  () => ({
    default:
      class MockVectorSource {
        constructor(
          _options?: unknown,
        ) {}

        getExtent() {
          return [
            0,
            0,
            1,
            1,
          ];
        }
      },
  }),
);

vi.mock(
  "ol/layer/Tile",
  () => ({
    default:
      class MockTileLayer {
        constructor(
          _options?: unknown,
        ) {}
      },
  }),
);

vi.mock(
  "ol/layer/Vector",
  () => ({
    default:
      class MockVectorLayer {
        constructor(
          _options?: unknown,
        ) {}

        changed() {
          olMocks.changed();
        }
      },
  }),
);

vi.mock(
  "ol/View",
  () => ({
    default:
      class MockView {
        constructor(
          _options?: unknown,
        ) {}

        fit(
          ...args: unknown[]
        ) {
          olMocks.fit(
            ...args,
          );
        }

        animate(
          ...args: unknown[]
        ) {
          olMocks.animate(
            ...args,
          );
        }
      },
  }),
);

vi.mock(
  "ol/Map",
  () => ({
    default:
      class MockMap {
        private view: {
          fit: (
            ...args: unknown[]
          ) => void;
          animate: (
            ...args: unknown[]
          ) => void;
        };

        constructor(
          options: {
            view: {
              fit: (
                ...args: unknown[]
              ) => void;
              animate: (
                ...args: unknown[]
              ) => void;
            };
          },
        ) {
          this.view =
            options.view;
        }

        on(
          _eventName: string,
          _handler: unknown,
        ) {}

        forEachFeatureAtPixel() {
          return undefined;
        }

        hasFeatureAtPixel() {
          return false;
        }

        getView() {
          return this.view;
        }

        setTarget(
          target: unknown,
        ) {
          olMocks.setTarget(
            target,
          );
        }
      },
  }),
);

vi.mock(
  "ol/style",
  () => {
    class EmptyStyleClass {
      constructor(
        _options?: unknown,
      ) {}
    }

    return {
      Circle:
        EmptyStyleClass,
      Fill:
        EmptyStyleClass,
      Stroke:
        EmptyStyleClass,
      Style:
        EmptyStyleClass,
      Text:
        EmptyStyleClass,
    };
  },
);

import {
  MapPanel,
} from "./MapPanel";

describe(
  "MapPanel",
  () => {
    beforeEach(
      () => {
        olMocks.fit.mockClear();
        olMocks.animate.mockClear();
        olMocks.changed.mockClear();
        olMocks.setTarget.mockClear();
      },
    );

    it(
      "Türkçe harita başlığını ve seçim lejandını gösterir",
      () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
            compareGridIds={
              []
            }
            onSelect={
              vi.fn()
            }
          />,
        );

        expect(
          screen.getByRole(
            "heading",
            {
              name:
                "Aday haritası",
            },
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "aday",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "karşılaştırma",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByText(
            "seçili",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByLabelText(
            "Etkileşimli Ankara aday haritası",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "uygunluk görünümünü varsayılan olarak aktif eder",
      () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
            onSelect={
              vi.fn()
            }
          />,
        );

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "Uygunluk",
            },
          ),
        ).toHaveAttribute(
          "aria-pressed",
          "true",
        );

        expect(
          screen.getByLabelText(
            "Uygunluk ölçeği",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "ML ve model uyuşmazlığı görünüm modları arasında geçiş yapar",
      () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
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
                "ML uzlaşısı",
            },
          ),
        );

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "ML uzlaşısı",
            },
          ),
        ).toHaveAttribute(
          "aria-pressed",
          "true",
        );

        expect(
          screen.getByLabelText(
            "ML uzlaşısı ölçeği",
          ),
        ).toBeInTheDocument();

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Model uyuşmazlığı",
            },
          ),
        );

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "Model uyuşmazlığı",
            },
          ),
        ).toHaveAttribute(
          "aria-pressed",
          "true",
        );

        expect(
          screen.getByLabelText(
            "Model uyuşmazlığı ölçeği",
          ),
        ).toBeInTheDocument();
      },
    );

    it(
      "adaylar bulunduğunda görünümü aday kapsamına oturtur",
      async () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
            onSelect={
              vi.fn()
            }
          />,
        );

        await waitFor(
          () => {
            expect(
              olMocks.fit,
            ).toHaveBeenCalled();
          },
        );
      },
    );

    it(
      "seçili adaya odaklanır",
      () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
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
                "Seçili adaya odaklan",
            },
          ),
        );

        expect(
          olMocks.animate,
        ).toHaveBeenCalledWith(
          expect.objectContaining({
            center: [
              31.3405406,
              40.1952609,
            ],
            zoom:
              12,
          }),
        );
      },
    );

    it(
      "tüm adayları göster düğmesi harita kapsamını yeniden ayarlar",
      () => {
        render(
          <MapPanel
            candidates={[
              createCandidate(),
            ]}
            selectedGridId="ANK_004300"
            onSelect={
              vi.fn()
            }
          />,
        );

        const initialFitCalls =
          olMocks.fit.mock.calls.length;

        fireEvent.click(
          screen.getByRole(
            "button",
            {
              name:
                "Tüm adayları göster",
            },
          ),
        );

        expect(
          olMocks.fit.mock.calls.length,
        ).toBeGreaterThan(
          initialFitCalls,
        );
      },
    );

    it(
      "filtre sonucu boşsa harita boş durumunu ve devre dışı kontrolleri gösterir",
      () => {
        render(
          <MapPanel
            candidates={
              []
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
            "Mevcut filtrelerle eşleşen aday yok.",
          ),
        ).toBeInTheDocument();

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "Seçili adaya odaklan",
            },
          ),
        ).toBeDisabled();

        expect(
          screen.getByRole(
            "button",
            {
              name:
                "Tüm adayları göster",
            },
          ),
        ).toBeDisabled();

        expect(
          olMocks.fit,
        ).not.toHaveBeenCalled();
      },
    );
  },
);
