import {
  useEffect,
  useRef,
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

interface MapPanelProps {
  candidates: DecisionSupportCandidate[];
  selectedGridId: string | null;
  onSelect: (gridId: string) => void;
  compareGridIds?: string[];
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

  const selectedGridIdRef =
    useRef<string | null>(
      selectedGridId,
    );

  const compareGridIdsRef =
    useRef<string[]>(
      compareGridIds,
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
              "priorityBand",
              candidate.suitability.priority_band,
            );

            return feature;
          },
        );

      const vectorSource =
        new VectorSource({
          features,
        });

      const vectorLayer =
        new VectorLayer({
          source: vectorSource,
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

            const selected =
              gridId ===
              selectedGridIdRef.current;

            const compared =
              compareGridIdsRef.current.includes(
                gridId,
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
                        selected
                          ? "#f6b73c"
                          : compared
                            ? "#2f855a"
                            : "#173f5f",
                    }),
                  stroke:
                    new Stroke({
                      color:
                        "#ffffff",
                      width:
                        selected
                          ? 4
                          : compared
                            ? 3
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
                        selected
                          ? "#14213d"
                          : "#ffffff",
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
              zoom: 7.3,
              minZoom: 6,
              maxZoom: 16,
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
                hitTolerance: 8,
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

          const hasFeature =
            map.hasFeatureAtPixel(
              event.pixel,
              {
                hitTolerance: 8,
              },
            );

          mapElementRef.current.style.cursor =
            hasFeature
              ? "pointer"
              : "";
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
              maxZoom: 9,
              duration: 0,
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
      };
    },
    [
      candidates,
      onSelect,
    ],
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

        <div className="map-legend">
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

      <div className="map-canvas-wrap">
        <div
          ref={
            mapElementRef
          }
          className="map-canvas"
          aria-label="Etkileşimli Ankara aday haritası"
        />

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
        İşaretçi numaraları 25 km
        mekânsal çeşitlilik seçim sırasını gösterir.
        Filtreler hem kısa listeyi hem de
        haritayı günceller.
      </p>
    </section>
  );
}
