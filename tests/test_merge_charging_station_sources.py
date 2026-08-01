from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OSM_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations.gpkg"
)

MERGED_GPKG_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.gpkg"
)

MERGED_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cankaya_charging_stations_merged.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "cankaya_charging_station_source_merge_summary.md"
)

OSM_LAYER_NAME = "charging_stations"

MERGED_LAYER_NAME = (
    "charging_stations_merged"
)

DUPLICATE_DISTANCE_THRESHOLD_METERS = 100.0


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a required merge output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run merge_charging_station_sources.py first."
        )


def normalize_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize GIS boolean fields read as bool, number or text."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().any():
        return numeric.fillna(0).gt(0)

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
        }
    )


@pytest.fixture(scope="session")
def osm_stations() -> gpd.GeoDataFrame:
    """Load the original OSM station inventory."""

    ensure_file_exists(
        OSM_GPKG_PATH,
        "OSM charging-station GeoPackage",
    )

    return gpd.read_file(
        OSM_GPKG_PATH,
        layer=OSM_LAYER_NAME,
    )


@pytest.fixture(scope="session")
def merged_stations() -> gpd.GeoDataFrame:
    """Load the merged OSM and EPDK station inventory."""

    ensure_file_exists(
        MERGED_GPKG_PATH,
        "Merged charging-station GeoPackage",
    )

    return gpd.read_file(
        MERGED_GPKG_PATH,
        layer=MERGED_LAYER_NAME,
    )


@pytest.fixture(scope="session")
def merged_csv() -> pd.DataFrame:
    """Load the non-spatial merged CSV output."""

    ensure_file_exists(
        MERGED_CSV_PATH,
        "Merged charging-station CSV",
    )

    return pd.read_csv(
        MERGED_CSV_PATH,
        encoding="utf-8-sig",
    )


def test_merge_output_files_exist() -> None:
    """All merge outputs must exist and contain data."""

    required_outputs = (
        OSM_GPKG_PATH,
        MERGED_GPKG_PATH,
        MERGED_CSV_PATH,
        SUMMARY_PATH,
    )

    for output_path in required_outputs:
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_merged_station_count_and_unique_ids(
    osm_stations: gpd.GeoDataFrame,
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """Eighteen OSM and one EPDK station must produce nineteen rows."""

    assert len(osm_stations) == 18
    assert len(merged_stations) == 19

    assert merged_stations[
        "station_id"
    ].notna().all()

    assert merged_stations[
        "station_id"
    ].is_unique

    assert not merged_stations.geometry.isna().any()

    assert merged_stations.crs is not None


def test_source_distribution_and_provenance(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """Every merged record must retain source-provenance flags."""

    source_counts = (
        merged_stations["data_source"]
        .value_counts()
        .to_dict()
    )

    assert source_counts == {
        "OSM": 18,
        "EPDK": 1,
    }

    osm_rows = merged_stations.loc[
        merged_stations[
            "data_source"
        ].eq("OSM")
    ]

    epdk_rows = merged_stations.loc[
        merged_stations[
            "data_source"
        ].eq("EPDK")
    ]

    assert normalize_boolean_series(
        osm_rows["source_osm"]
    ).all()

    assert not normalize_boolean_series(
        osm_rows["source_epdk"]
    ).any()

    assert not normalize_boolean_series(
        epdk_rows["source_osm"]
    ).any()

    assert normalize_boolean_series(
        epdk_rows["source_epdk"]
    ).all()


def test_all_osm_station_ids_are_preserved(
    osm_stations: gpd.GeoDataFrame,
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """The merge must not remove or rename OSM station records."""

    original_ids = set(
        osm_stations["station_id"]
        .astype(str)
    )

    merged_osm_ids = set(
        merged_stations.loc[
            merged_stations[
                "data_source"
            ].eq("OSM"),
            "station_id",
        ]
        .astype(str)
    )

    assert merged_osm_ids == original_ids


def test_epdk_2622_metadata(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """The verified 365 AVM EPDK record must appear once."""

    epdk = merged_stations.loc[
        merged_stations[
            "station_id"
        ]
        .astype(str)
        .eq("epdk_srj_2622")
    ]

    assert len(epdk) == 1

    row = epdk.iloc[0]

    assert row["data_source"] == "EPDK"
    assert row["name"] == "Çankaya 365 AVM"
    assert row["epdk_station_no"] == "ŞRJ/2622"

    assert int(
        row["epdk_socket_count"]
    ) == 2

    assert int(
        row["epdk_ac_socket_count"]
    ) == 2

    assert int(
        row["epdk_dc_socket_count"]
    ) == 0

    assert float(
        row["epdk_total_socket_power_kw"]
    ) == pytest.approx(
        44.0,
        abs=0.01,
    )

    assert (
        str(
            row["coordinate_confidence"]
        )
        .strip()
        .lower()
        == "medium"
    )

    assert not bool(
        int(
            row[
                "coordinate_is_official_epdk"
            ]
        )
    )

    assert (
        str(
            row["coordinate_source"]
        ).strip()
        != ""
    )


def test_epdk_capacity_and_connector_flags(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """The EPDK station must contribute capacity and AC information."""

    epdk = merged_stations.loc[
        merged_stations[
            "station_id"
        ]
        .astype(str)
        .eq("epdk_srj_2622")
    ].copy()

    assert len(epdk) == 1

    capacity = pd.to_numeric(
        epdk["capacity_numeric"],
        errors="coerce",
    )

    assert capacity.notna().all()

    assert float(
        capacity.iloc[0]
    ) == pytest.approx(
        2.0,
        abs=0.01,
    )

    assert normalize_boolean_series(
        epdk["has_ac_connector"]
    ).iloc[0]

    assert not normalize_boolean_series(
        epdk["has_dc_connector"]
    ).iloc[0]


def test_epdk_geometry_matches_verified_coordinate(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """The merged EPDK point must retain the accepted coordinate."""

    epdk = merged_stations.loc[
        merged_stations[
            "station_id"
        ]
        .astype(str)
        .eq("epdk_srj_2622")
    ].copy()

    assert len(epdk) == 1

    assert epdk.geometry.iloc[0].geom_type == (
        "Point"
    )

    epdk_wgs84 = epdk.to_crs(
        "EPSG:4326"
    )

    point = epdk_wgs84.geometry.iloc[0]

    assert point.x == pytest.approx(
        32.8698771,
        abs=0.00001,
    )

    assert point.y == pytest.approx(
        39.8758028,
        abs=0.00001,
    )


def test_epdk_station_is_not_an_osm_duplicate(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """The accepted EPDK point must be beyond the duplicate threshold."""

    epdk = merged_stations.loc[
        merged_stations[
            "station_id"
        ]
        .astype(str)
        .eq("epdk_srj_2622")
    ]

    assert len(epdk) == 1

    nearest_distance = float(
        epdk.iloc[0][
            "nearest_osm_distance_m"
        ]
    )

    assert nearest_distance > (
        DUPLICATE_DISTANCE_THRESHOLD_METERS
    )

    assert nearest_distance == pytest.approx(
        1844.18,
        abs=0.05,
    )


def test_merged_csv_matches_geopackage(
    merged_stations: gpd.GeoDataFrame,
    merged_csv: pd.DataFrame,
) -> None:
    """CSV and GeoPackage outputs must contain the same records."""

    assert len(merged_csv) == len(
        merged_stations
    )

    csv_ids = set(
        merged_csv["station_id"]
        .astype(str)
    )

    gpkg_ids = set(
        merged_stations["station_id"]
        .astype(str)
    )

    assert csv_ids == gpkg_ids

    csv_source_counts = (
        merged_csv["data_source"]
        .value_counts()
        .to_dict()
    )

    gpkg_source_counts = (
        merged_stations["data_source"]
        .value_counts()
        .to_dict()
    )

    assert csv_source_counts == (
        gpkg_source_counts
    )


def test_only_one_epdk_record_is_added(
    merged_stations: gpd.GeoDataFrame,
) -> None:
    """Exactly one verified EPDK-only station must be added."""

    epdk_rows = merged_stations.loc[
        merged_stations[
            "data_source"
        ].eq("EPDK")
    ]

    assert len(epdk_rows) == 1

    assert epdk_rows[
        "station_id"
    ].tolist() == [
        "epdk_srj_2622"
    ]


def test_merge_summary_documents_counts_and_caveat() -> None:
    """The summary must document counts and coordinate limitations."""

    ensure_file_exists(
        SUMMARY_PATH,
        "Charging source merge summary",
    )

    summary = SUMMARY_PATH.read_text(
        encoding="utf-8"
    )

    required_text = (
        "OpenStreetMap station records: 18",
        "Verified EPDK station records: 1",
        "New EPDK-only stations added: 1",
        "Final merged station count: 19",
        "not an official coordinate",
    )

    for text in required_text:
        assert text in summary
