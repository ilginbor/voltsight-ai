from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATION_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_stations.csv"
)

SOCKET_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_sockets.csv"
)

REPORT_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_report.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "epdk_cankaya_charging_report_summary.md"
)

REQUIRED_STATION_COLUMNS = {
    "station_order",
    "station_no",
    "station_name",
    "service_type",
    "brand",
    "network_operator",
    "station_operator",
    "green_station_text",
    "is_green_station",
    "address",
    "socket_count",
    "ac_socket_count",
    "dc_socket_count",
    "known_power_socket_count",
    "total_socket_power_kw",
    "maximum_socket_power_kw",
    "socket_current_types",
    "connector_types",
    "has_ac_socket",
    "has_dc_socket",
}

REQUIRED_SOCKET_COLUMNS = {
    "station_no",
    "station_name",
    "socket_no",
    "socket_current_type",
    "connector_type",
    "socket_power_kw",
    "is_ac_socket",
    "is_dc_socket",
}


def ensure_file_exists(
    path: Path,
    description: str,
) -> None:
    """Fail clearly when a required pipeline output is missing."""

    if not path.exists():
        pytest.fail(
            f"{description} does not exist: {path}\n"
            "Run prepare_epdk_charging_report.py first."
        )


@pytest.fixture(scope="session")
def stations() -> pd.DataFrame:
    """Load normalized EPDK station records."""

    ensure_file_exists(
        STATION_CSV_PATH,
        "EPDK station CSV",
    )

    return pd.read_csv(
        STATION_CSV_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def sockets() -> pd.DataFrame:
    """Load normalized EPDK socket records."""

    ensure_file_exists(
        SOCKET_CSV_PATH,
        "EPDK socket CSV",
    )

    return pd.read_csv(
        SOCKET_CSV_PATH,
        encoding="utf-8-sig",
    )


@pytest.fixture(scope="session")
def report() -> dict:
    """Load normalized EPDK JSON output."""

    ensure_file_exists(
        REPORT_JSON_PATH,
        "EPDK report JSON",
    )

    return json.loads(
        REPORT_JSON_PATH.read_text(
            encoding="utf-8-sig"
        )
    )


def test_epdk_output_files_exist() -> None:
    """All reproducible EPDK report outputs must exist."""

    required_outputs = (
        STATION_CSV_PATH,
        SOCKET_CSV_PATH,
        REPORT_JSON_PATH,
        SUMMARY_PATH,
    )

    for output_path in required_outputs:
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_station_dataset_schema_and_count(
    stations: pd.DataFrame,
) -> None:
    """The report must contain ten normalized stations."""

    assert len(stations) == 10

    assert REQUIRED_STATION_COLUMNS.issubset(
        stations.columns
    )

    assert stations["station_no"].notna().all()

    assert stations["station_no"].is_unique


def test_socket_dataset_schema_and_count(
    sockets: pd.DataFrame,
) -> None:
    """The report must contain eighteen physical sockets."""

    assert len(sockets) == 18

    assert REQUIRED_SOCKET_COLUMNS.issubset(
        sockets.columns
    )

    assert sockets["socket_no"].notna().all()

    assert sockets["socket_no"].is_unique


def test_every_socket_references_a_known_station(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Every socket must belong to one normalized station."""

    station_numbers = set(
        stations["station_no"].astype(str)
    )

    socket_station_numbers = set(
        sockets["station_no"].astype(str)
    )

    assert socket_station_numbers.issubset(
        station_numbers
    )

    assert socket_station_numbers == station_numbers


def test_socket_current_type_distribution(
    sockets: pd.DataFrame,
) -> None:
    """The official report contains seventeen AC and one DC socket."""

    current_types = (
        sockets["socket_current_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
        .to_dict()
    )

    assert current_types.get("AC", 0) == 17
    assert current_types.get("DC", 0) == 1

    ac_flags = pd.to_numeric(
        sockets["is_ac_socket"],
        errors="coerce",
    ).fillna(0)

    dc_flags = pd.to_numeric(
        sockets["is_dc_socket"],
        errors="coerce",
    ).fillna(0)

    assert int(ac_flags.sum()) == 17
    assert int(dc_flags.sum()) == 1


def test_socket_power_totals(
    sockets: pd.DataFrame,
) -> None:
    """Socket-level power must reproduce the official totals."""

    power = pd.to_numeric(
        sockets["socket_power_kw"],
        errors="coerce",
    )

    assert power.notna().all()

    assert float(power.sum()) == pytest.approx(
        368.0,
        abs=0.01,
    )

    assert float(power.max()) == pytest.approx(
        60.0,
        abs=0.01,
    )


def test_station_aggregates_match_socket_rows(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Station aggregates must be reproducible from socket rows."""

    prepared_sockets = sockets.copy()

    numeric_columns = (
        "socket_power_kw",
        "is_ac_socket",
        "is_dc_socket",
    )

    for column in numeric_columns:
        prepared_sockets[column] = pd.to_numeric(
            prepared_sockets[column],
            errors="coerce",
        )

    socket_aggregates = (
        prepared_sockets
        .groupby("station_no")
        .agg(
            socket_count=(
                "socket_no",
                "count",
            ),
            ac_socket_count=(
                "is_ac_socket",
                "sum",
            ),
            dc_socket_count=(
                "is_dc_socket",
                "sum",
            ),
            known_power_socket_count=(
                "socket_power_kw",
                "count",
            ),
            total_socket_power_kw=(
                "socket_power_kw",
                "sum",
            ),
            maximum_socket_power_kw=(
                "socket_power_kw",
                "max",
            ),
        )
        .sort_index()
    )

    station_aggregates = (
        stations
        .set_index("station_no")
        [
            [
                "socket_count",
                "ac_socket_count",
                "dc_socket_count",
                "known_power_socket_count",
                "total_socket_power_kw",
                "maximum_socket_power_kw",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .sort_index()
    )

    pd.testing.assert_frame_equal(
        station_aggregates,
        socket_aggregates,
        check_dtype=False,
        check_names=True,
        rtol=1e-9,
        atol=0.01,
    )


def test_all_stations_have_addresses(
    stations: pd.DataFrame,
) -> None:
    """All ten official station records must retain an address."""

    addresses = (
        stations["address"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert addresses.ne("").all()

    assert addresses.str.contains(
        "Çankaya",
        case=False,
        regex=False,
    ).all()


def test_json_metadata_matches_csv_outputs(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
    report: dict,
) -> None:
    """JSON counts and arrays must match the normalized CSV files."""

    assert report["source_file"] == (
        "sarjIstasyonlari.xls"
    )

    assert report["station_count"] == len(
        stations
    )

    assert report["socket_count"] == len(
        sockets
    )

    assert len(report["stations"]) == len(
        stations
    )

    assert len(report["sockets"]) == len(
        sockets
    )

    assert report["station_count"] == 10
    assert report["socket_count"] == 18


def test_epdk_2622_station_metadata(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Çankaya 365 AVM must retain its verified EPDK metadata."""

    station = stations.loc[
        stations["station_no"]
        .astype(str)
        .eq("ŞRJ/2622")
    ]

    assert len(station) == 1

    row = station.iloc[0]

    assert row["station_name"] == (
        "Çankaya 365 AVM"
    )

    assert int(row["socket_count"]) == 2
    assert int(row["ac_socket_count"]) == 2
    assert int(row["dc_socket_count"]) == 0

    assert float(
        row["total_socket_power_kw"]
    ) == pytest.approx(
        44.0,
        abs=0.01,
    )

    assert float(
        row["maximum_socket_power_kw"]
    ) == pytest.approx(
        22.0,
        abs=0.01,
    )

    station_sockets = sockets.loc[
        sockets["station_no"]
        .astype(str)
        .eq("ŞRJ/2622")
    ]

    assert len(station_sockets) == 2

    assert set(
        station_sockets[
            "socket_current_type"
        ]
        .astype(str)
        .str.upper()
    ) == {"AC"}
