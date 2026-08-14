import {
  useEffect,
  useRef,
  useState,
} from "react";

import Feature from "ol/Feature";
import Map from "ol/Map";
import View from "ol/View";
import Point from "ol/geom/Point";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import {
  fromLonLat,
} from "ol/proj";
import OSM from "ol/source/OSM";
import VectorSource from "ol/source/Vector";
import {
  Circle as CircleStyle,
  Fill,
  Stroke,
  Style,
  Text,
} from "ol/style";

import type {
  DecisionSupportCandidate,
} from "../types/api";

export type MapViewMode =
  | "suitability"
  | "ml_consensus"
  | "model_disagreement";

interface MapPanelProps {
  candidates: DecisionSupportCandidate[];
  selectedGridId: string | null;
  onSelect: (gridId: string) => void;
  compareGridIds?: string[];
}

interface HoveredCandidate {
  candidate: DecisionSupportCandidate;
  x: number;
  y: number;
}

interface LegendItem {
  label: string;
  className: string;
}

const VIEW_MODE_LABELS: Record<
  MapViewMode,
  string
> = {
  suitability:
    "Uygunluk",
  ml_consensus:
    "ML uzlaşısı",
  model_disagreement:
    "Model uyuşmazlığı",
};

const VIEW_MODE_DESCRIPTIONS: Record<
  MapViewMode,
  string
> = {
  suitability:
    "Açıklanabilir uygunluk skoru",
  ml_consensus:
    "Fold-normalize mekânsal OOF ML uzlaşı yüzdeliği",
  model_disagreement:
    "Model yüzdelikleri arasındaki fark; düşük değer daha yüksek uyum anlamına gelir",
};

const VIEW_MODE_LEGENDS: Record<
  MapViewMode,
  LegendItem[]
> = {
  suitability: [
    {
      label:
        "< 65",
      className:
        "map-scale-dot--low",
    },
    {
      label:
        "65–74",
      className:
        "map-scale-dot--medium",
    },
    {
      label:
        "75–84",
      className:
        "map-scale-dot--high",
    },
    {
      label:
        "85+",
      className:
        "map-scale-dot--very-high",
    },
  ],
  ml_consensus: [
    {
      label:
        "< 80",
      className:
        "map-scale-dot--low",
    },
    {
      label:
        "80–89",
      className:
        "map-scale-dot--medium",
    },
    {
      label:
        "90–94",
      className:
        "map-scale-dot--high",
    },
    {
      label:
        "95+",
      className:
        "map-scale-dot--very-high",
    },
  ],
  model_disagreement: [
    {
      label:
        "> 50",
      className:
        "map-scale-dot--low",
    },
    {
      label:
        "26–50",
      className:
        "map-scale-dot--medium",
    },
    {
      label:
        "11–25",
      className:
        "map-scale-dot--high",
    },
    {
      label:
        "0–10",
      className:
        "map-scale-dot--very-high",
    },
  ],
};

function markerFillColor(
  mode: MapViewMode,
  suitabilityScore: number,
  mlConsensus: number,
  modelSpread: number,
): string {
  if (
    mode ===
    "suitability"
  ) {
    if (
      suitabilityScore >=
      85
    ) {
      return "#2f855a";
    }

    if (
      suitabilityScore >=
      75
    ) {
      return "#2f6f95";
    }

    if (
      suitabilityScore >=
      65
    ) {
      return "#f6b73c";
    }

    return "#b54747";
  }

  if (
    mode ===
    "ml_consensus"
  ) {
    if (
      mlConsensus >=
      95
    ) {
      return "#2f855a";
    }

    if (
      mlConsensus >=
      90
    ) {
      return "#2f6f95";
    }

    if (
      mlConsensus >=
      80
    ) {
      return "#f6b73c";
    }

    return "#b54747";
  }

  if (
    modelSpread <=
    10
  ) {
    return "#2f855a";
  }

  if (
    modelSpread <=
    25
  ) {
    return "#2f6f95";
  }

  if (
    modelSpread <=
    50
  ) {
    return "#f6b73c";
  }

  return "#b54747";
}

function metricValue(
  candidate: DecisionSupportCandidate,
  mode: MapViewMode,
): string {
  if (
    mode ===
    "suitability"
  ) {
    return candidate.suitability.score.toFixed(
      1,
    );
  }

  if (
    mode ===
    "ml_consensus"
  ) {
    return `${candidate.ml_support.consensus_percentile.toFixed(
      1,
    )}%`;
  }

  return `${candidate.ml_support.model_percentile_spread.toFixed(
    1,
  )} puan`;
}

export function MapPanel({
  candidates,
  selectedGridId,
  onSelect,
  compareGridIds = [],
}: MapPanelProps) {
  const mapElementRef =
    useRef<HTMLDivElement | null>(
      null,
    );

  const mapRef =
    useRef<Map | null>(
      null,
    );

  const vectorLayerRef =
    useRef<VectorLayer<VectorSource> | null>(
      null,
    );

  const vectorSourceRef =
    useRef<VectorSource | null>(
      null,
    );

  const selectedGridIdRef =
    useRef<string | null>(
      selectedGridId,
    );

  const compareGridIdsRef =
    useRef<string[]>(
      compareGridIds,
    );

  const [
    viewMode,
    setViewMode,
  ] =
    useState<MapViewMode>(
      "suitability",
    );

  const viewModeRef =
    useRef<MapViewMode>(
      viewMode,
    );

  const [
    hoveredCandidate,
    setHoveredCandidate,
  ] =
    useState<HoveredCandidate | null>(
      null,
    );

  useEffect(
    () => {
      selectedGridIdRef.current =
        selectedGridId;

      vectorLayerRef.current?.changed();
    },
    [
      selectedGridId,
    ],
  );

  useEffect(
    () => {
      compareGridIdsRef.current =
        compareGridIds;

      vectorLayerRef.current?.changed();
    },
    [
      compareGridIds,
    ],
  );

  useEffect(
    () => {
      viewModeRef.current =
        viewMode;

      vectorLayerRef.current?.changed();

      setHoveredCandidate(
        null,
      );
    },
    [
      viewMode,
    ],
  );

  useEffect(
    () => {
      if (
        mapElementRef.current ===
        null
      ) {
        return;
      }

      const features =
        candidates.map(
          (
            candidate,
          ) => {
            const feature =
              new Feature({
                geometry:
                  new Point(
                    fromLonLat(
                      [
                        candidate
                          .location
                          .longitude,
                        candidate
                          .location
                          .latitude,
                      ],
                    ),
                  ),
              });

            feature.set(
              "gridId",
              candidate.grid_id,
            );

            feature.set(
              "selectionRank",
              candidate.selection_rank,
            );

            feature.set(
              "suitabilityScore",
              candidate.suitability.score,
            );

            feature.set(
              "mlConsensus",
              candidate.ml_support.consensus_percentile,
            );

            feature.set(
              "modelSpread",
              candidate.ml_support.model_percentile_spread,
            );

            return feature;
          },
        );

      const vectorSource =
        new VectorSource({
          features,
        });

      vectorSourceRef.current =
        vectorSource;

      const vectorLayer =
        new VectorLayer({
          source:
            vectorSource,
          style: (
            feature,
          ) => {
            const gridId =
              String(
                feature.get(
                  "gridId",
                ),
              );

            const selectionRank =
              String(
                feature.get(
                  "selectionRank",
                ),
              );

            const suitabilityScore =
              Number(
                feature.get(
                  "suitabilityScore",
                ),
              );

            const mlConsensus =
              Number(
                feature.get(
                  "mlConsensus",
                ),
              );

            const modelSpread =
              Number(
                feature.get(
                  "modelSpread",
                ),
              );

            const selected =
              gridId ===
              selectedGridIdRef.current;

            const compared =
              compareGridIdsRef.current.includes(
                gridId,
              );

            const fillColor =
              markerFillColor(
                viewModeRef.current,
                suitabilityScore,
                mlConsensus,
                modelSpread,
              );

            return new Style({
              image:
                new CircleStyle({
                  radius:
                    selected
                      ? 13
                      : compared
                        ? 11
                        : 9,
                  fill:
                    new Fill({
                      color:
                        fillColor,
                    }),
                  stroke:
                    new Stroke({
                      color:
                        selected
                          ? "#f6b73c"
                          : compared
                            ? "#2f855a"
                            : "#ffffff",
                      width:
                        selected
                          ? 5
                          : compared
                            ? 4
                            : 2,
                    }),
                }),
              text:
                new Text({
                  text:
                    selectionRank,
                  font:
                    "600 11px Inter, system-ui, sans-serif",
                  fill:
                    new Fill({
                      color:
                        "#ffffff",
                    }),
                  stroke:
                    new Stroke({
                      color:
                        "rgba(16, 42, 67, 0.55)",
                      width:
                        2,
                    }),
                }),
            });
          },
        });

      vectorLayerRef.current =
        vectorLayer;

      const map =
        new Map({
          target:
            mapElementRef.current,
          layers: [
            new TileLayer({
              source:
                new OSM(),
            }),
            vectorLayer,
          ],
          view:
            new View({
              center:
                fromLonLat(
                  [
                    32.85,
                    39.92,
                  ],
                ),
              zoom:
                7.3,
              minZoom:
                6,
              maxZoom:
                16,
            }),
        });

      map.on(
        "singleclick",
        (
          event,
        ) => {
          const feature =
            map.forEachFeatureAtPixel(
              event.pixel,
              (
                candidateFeature,
              ) =>
                candidateFeature,
              {
                hitTolerance:
                  8,
              },
            );

          if (
            feature
          ) {
            const gridId =
              feature.get(
                "gridId",
              );

            if (
              typeof gridId ===
              "string"
            ) {
              onSelect(
                gridId,
              );
            }
          }
        },
      );

      map.on(
        "pointermove",
        (
          event,
        ) => {
          if (
            !mapElementRef.current
          ) {
            return;
          }

          const feature =
            map.forEachFeatureAtPixel(
              event.pixel,
              (
                candidateFeature,
              ) =>
                candidateFeature,
              {
                hitTolerance:
                  8,
              },
            );

          const gridId =
            feature?.get(
              "gridId",
            );

          const candidate =
            typeof gridId ===
            "string"
              ? candidates.find(
                  (
                    currentCandidate,
                  ) =>
                    currentCandidate.grid_id ===
                    gridId,
                )
              : undefined;

          if (
            candidate
          ) {
            mapElementRef.current.style.cursor =
              "pointer";

            setHoveredCandidate({
              candidate,
              x:
                event.pixel[0] +
                14,
              y:
                event.pixel[1] +
                14,
            });

            return;
          }

          mapElementRef.current.style.cursor =
            "";

          setHoveredCandidate(
            null,
          );
        },
      );

      if (
        features.length >
        0
      ) {
        const extent =
          vectorSource.getExtent();

        if (
          extent !==
          null
        ) {
          map.getView().fit(
            extent,
            {
              padding: [
                56,
                56,
                56,
                56,
              ],
              maxZoom:
                9,
              duration:
                0,
            },
          );
        }
      }

      mapRef.current =
        map;

      return () => {
        map.setTarget(
          undefined,
        );

        mapRef.current =
          null;

        vectorLayerRef.current =
          null;

        vectorSourceRef.current =
          null;
      };
    },
    [
      candidates,
      onSelect,
    ],
  );

  const handleFocusSelected =
    () => {
      const selectedCandidate =
        candidates.find(
          (
            candidate,
          ) =>
            candidate.grid_id ===
            selectedGridId,
        );

      if (
        !selectedCandidate ||
        !mapRef.current
      ) {
        return;
      }

      mapRef.current
        .getView()
        .animate({
          center:
            fromLonLat(
              [
                selectedCandidate
                  .location
                  .longitude,
                selectedCandidate
                  .location
                  .latitude,
              ],
            ),
          zoom:
            12,
          duration:
            450,
        });
    };

  const handleShowAll =
    () => {
      const map =
        mapRef.current;

      const source =
        vectorSourceRef.current;

      if (
        !map ||
        !source ||
        candidates.length ===
          0
      ) {
        return;
      }

      const extent =
        source.getExtent();

      if (
        extent ===
        null
      ) {
        return;
      }

      map.getView().fit(
        extent,
        {
          padding: [
            56,
            56,
            56,
            56,
          ],
          maxZoom:
            9,
          duration:
            450,
        },
      );
    };

  const selectedExists =
    Boolean(
      selectedGridId &&
      candidates.some(
        (
          candidate,
        ) =>
          candidate.grid_id ===
          selectedGridId,
      ),
    );

  return (
    <section className="map-panel">
      <div className="map-panel__header">
        <div>
          <p className="eyebrow">
            Ankara · Türkiye
          </p>

          <h2>
            Aday haritası
          </h2>
        </div>

        <div
          className="map-status-legend"
          aria-label="Harita seçim lejandı"
        >
          <span>
            <i className="map-dot" />
            aday
          </span>

          <span>
            <i className="map-dot map-dot--compare" />
            karşılaştırma
          </span>

          <span>
            <i className="map-dot map-dot--selected" />
            seçili
          </span>
        </div>
      </div>

      <div className="map-toolbar">
        <div className="map-toolbar__mode">
          <span className="map-toolbar__label">
            Görünüm
          </span>

          <div
            className="map-mode-switch"
            role="group"
            aria-label="Harita görünüm metriği"
          >
            <button
              type="button"
              aria-pressed={
                viewMode ===
                "suitability"
              }
              onClick={
                () => {
                  setViewMode(
                    "suitability",
                  );
                }
              }
            >
              Uygunluk
            </button>

            <button
              type="button"
              aria-pressed={
                viewMode ===
                "ml_consensus"
              }
              onClick={
                () => {
                  setViewMode(
                    "ml_consensus",
                  );
                }
              }
            >
              ML uzlaşısı
            </button>

            <button
              type="button"
              aria-pressed={
                viewMode ===
                "model_disagreement"
              }
              onClick={
                () => {
                  setViewMode(
                    "model_disagreement",
                  );
                }
              }
            >
              Model uyuşmazlığı
            </button>
          </div>
        </div>

        <div className="map-toolbar__actions">
          <button
            type="button"
            className="map-tool-button"
            disabled={
              !selectedExists
            }
            onClick={
              handleFocusSelected
            }
          >
            Seçili adaya odaklan
          </button>

          <button
            type="button"
            className="map-tool-button"
            disabled={
              candidates.length ===
              0
            }
            onClick={
              handleShowAll
            }
          >
            Tüm adayları göster
          </button>
        </div>
      </div>

      <div className="map-metric-legend">
        <div>
          <strong>
            {VIEW_MODE_LABELS[
              viewMode
            ]}
          </strong>

          <span>
            {VIEW_MODE_DESCRIPTIONS[
              viewMode
            ]}
          </span>
        </div>

        <div
          className="map-scale"
          aria-label={`${VIEW_MODE_LABELS[viewMode]} ölçeği`}
        >
          {
            VIEW_MODE_LEGENDS[
              viewMode
            ].map(
              (
                item,
              ) => (
                <span
                  key={
                    item.label
                  }
                >
                  <i
                    className={`map-scale-dot ${item.className}`}
                  />
                  {item.label}
                </span>
              ),
            )
          }
        </div>
      </div>

      <div className="map-canvas-wrap">
        <div
          ref={
            mapElementRef
          }
          className="map-canvas"
          aria-label="Etkileşimli Ankara aday haritası"
        />

        {
          hoveredCandidate
            ? (
              <div
                className="map-tooltip"
                role="status"
                style={{
                  left:
                    hoveredCandidate.x,
                  top:
                    hoveredCandidate.y,
                }}
              >
                <div className="map-tooltip__topline">
                  <strong>
                    {
                      hoveredCandidate
                        .candidate
                        .grid_id
                    }
                  </strong>

                  <span>
                    #
                    {
                      hoveredCandidate
                        .candidate
                        .selection_rank
                    }
                  </span>
                </div>

                <div className="map-tooltip__metric">
                  <span>
                    {
                      VIEW_MODE_LABELS[
                        viewMode
                      ]
                    }
                  </span>

                  <strong>
                    {metricValue(
                      hoveredCandidate
                        .candidate,
                      viewMode,
                    )}
                  </strong>
                </div>

                <div className="map-tooltip__facts">
                  <span>
                    Uygunluk{" "}
                    <strong>
                      {
                        hoveredCandidate
                          .candidate
                          .suitability
                          .score
                          .toFixed(
                            1,
                          )
                      }
                    </strong>
                  </span>

                  <span>
                    ML{" "}
                    <strong>
                      {
                        hoveredCandidate
                          .candidate
                          .ml_support
                          .consensus_percentile
                          .toFixed(
                            1,
                          )
                      }
                    </strong>
                  </span>

                  <span>
                    Fark{" "}
                    <strong>
                      {
                        hoveredCandidate
                          .candidate
                          .ml_support
                          .model_percentile_spread
                          .toFixed(
                            1,
                          )
                      }
                    </strong>
                  </span>
                </div>
              </div>
            )
            : null
        }

        {
          candidates.length ===
          0
            ? (
              <div className="map-empty-overlay">
                Mevcut filtrelerle eşleşen
                aday yok.
              </div>
            )
            : null
        }
      </div>

      <p className="map-caption">
        Renkler seçilen görünüm metriğini
        gösterir. Sarı dış halka seçili adayı,
        yeşil dış halka karşılaştırılan adayları
        belirtir. ML desteği uygunluk skoruna
        karıştırılmaz.
      </p>
    </section>
  );
}
