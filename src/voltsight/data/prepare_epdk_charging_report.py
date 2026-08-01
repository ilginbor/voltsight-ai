from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sarjIstasyonlari.xls"
)

STATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_stations.csv"
)

SOCKET_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_sockets.csv"
)

JSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "epdk_cankaya_charging_report.json"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "epdk_cankaya_charging_report_summary.md"
)

EXPECTED_REPORT_COLUMNS = {
    "Sıra No",
    "İstasyon No",
    "İstasyon Adı",
    "Hizmet Şekli",
    "Marka",
    "Şarj Ağı İşletmecisi",
    "Şarj İstasyonu İşletmecisi",
    "Yeşil Şarj İstasyonu mu",
    "Adres",
    "Soket Bilgileri",
    "Unnamed: 10",
    "Unnamed: 11",
    "Unnamed: 12",
}

COLUMN_RENAME_MAP = {
    "Sıra No": "station_order",
    "İstasyon No": "station_no",
    "İstasyon Adı": "station_name",
    "Hizmet Şekli": "service_type",
    "Marka": "brand",
    "Şarj Ağı İşletmecisi": "network_operator",
    "Şarj İstasyonu İşletmecisi": "station_operator",
    "Yeşil Şarj İstasyonu mu": "green_station_text",
    "Adres": "address",
    "Soket Bilgileri": "socket_no",
    "Unnamed: 10": "socket_current_type",
    "Unnamed: 11": "connector_type",
    "Unnamed: 12": "socket_power_kw",
}

STATION_METADATA_COLUMNS = [
    "station_order",
    "station_no",
    "station_name",
    "service_type",
    "brand",
    "network_operator",
    "station_operator",
    "green_station_text",
    "address",
]

TEXT_COLUMNS = [
    "station_no",
    "station_name",
    "service_type",
    "brand",
    "network_operator",
    "station_operator",
    "green_station_text",
    "address",
    "socket_no",
    "socket_current_type",
    "connector_type",
]


def create_output_directories() -> None:
    """Create directories used by the EPDK report pipeline."""

    directories = {
        STATION_OUTPUT_PATH.parent,
        SOCKET_OUTPUT_PATH.parent,
        JSON_OUTPUT_PATH.parent,
        SUMMARY_OUTPUT_PATH.parent,
    }

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_input_file() -> None:
    """Ensure that the downloaded EPDK report exists."""

    if not RAW_REPORT_PATH.exists():
        raise FileNotFoundError(
            "The EPDK charging-station report was not found:\n"
            f"{RAW_REPORT_PATH}"
        )

    if RAW_REPORT_PATH.stat().st_size == 0:
        raise ValueError(
            "The EPDK charging-station report is empty."
        )


def clean_text(value: Any) -> str:
    """Convert an Excel value into normalized text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def parse_non_negative_float(
    value: Any,
) -> float:
    """Parse a non-negative numeric Excel value."""

    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass

    text = (
        str(value)
        .strip()
        .replace(",", ".")
    )

    if not text:
        return np.nan

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if match is None:
        return np.nan

    try:
        number = float(
            match.group(0)
        )
    except ValueError:
        return np.nan

    if (
        not np.isfinite(number)
        or number < 0
    ):
        return np.nan

    return number


def normalize_green_station(
    value: Any,
) -> int:
    """Convert the green-station field to zero or one."""

    normalized = (
        clean_text(value)
        .casefold()
        .replace("ı", "i")
    )

    positive_values = {
        "evet",
        "yes",
        "true",
        "1",
    }

    return int(
        normalized in positive_values
    )


def normalize_socket_current_type(
    value: Any,
) -> str:
    """Normalize the socket current type as AC, DC or UNKNOWN."""

    normalized = (
        clean_text(value)
        .upper()
    )

    if normalized == "AC":
        return "AC"

    if normalized == "DC":
        return "DC"

    return "UNKNOWN"


def join_unique_text(
    values: pd.Series,
) -> str:
    """Join unique non-empty strings while preserving order."""

    unique_values: list[str] = []

    for value in values:
        text = clean_text(
            value
        )

        if (
            text
            and text not in unique_values
        ):
            unique_values.append(
                text
            )

    return ";".join(
        unique_values
    )


def load_raw_report() -> pd.DataFrame:
    """Load and validate the downloaded EPDK XLS report."""

    dataframe = pd.read_excel(
        RAW_REPORT_PATH,
        engine="xlrd",
    )

    if dataframe.empty:
        raise ValueError(
            "The EPDK report contains no rows."
        )

    missing_columns = (
        EXPECTED_REPORT_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "The EPDK report schema has changed. "
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.rename(
        columns=COLUMN_RENAME_MAP
    )

    for column in TEXT_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .apply(clean_text)
        )

    dataframe[
        "socket_power_kw"
    ] = dataframe[
        "socket_power_kw"
    ].apply(
        parse_non_negative_float
    )

    print(
        "Raw EPDK report rows: "
        f"{len(dataframe):,}"
    )

    print(
        "Raw EPDK report columns: "
        f"{len(dataframe.columns):,}"
    )

    return dataframe


def build_station_rows(
    raw_report: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one metadata row for each charging station."""

    station_rows = raw_report[
        raw_report["station_no"]
        .ne("")
    ].copy()

    station_rows = station_rows[
        station_rows["station_no"]
        .str.casefold()
        .ne("istasyon no")
    ].copy()

    if station_rows.empty:
        raise ValueError(
            "No charging-station rows were detected."
        )

    station_rows[
        "station_order"
    ] = pd.to_numeric(
        station_rows["station_order"],
        errors="coerce",
    )

    station_rows.sort_values(
        by=[
            "station_order",
            "station_no",
        ],
        na_position="last",
        inplace=True,
    )

    station_rows.drop_duplicates(
        subset=["station_no"],
        keep="first",
        inplace=True,
    )

    station_rows[
        "is_green_station"
    ] = station_rows[
        "green_station_text"
    ].apply(
        normalize_green_station
    )

    selected_columns = [
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
    ]

    station_rows = station_rows[
        selected_columns
    ].copy()

    station_rows.reset_index(
        drop=True,
        inplace=True,
    )

    return station_rows


def build_socket_rows(
    raw_report: pd.DataFrame,
) -> pd.DataFrame:
    """Convert nested socket rows into a normalized socket table."""

    expanded_report = (
        raw_report.copy()
    )

    for column in STATION_METADATA_COLUMNS:
        empty_mask = (
            expanded_report[column]
            .eq("")
            if expanded_report[column].dtype == object
            else expanded_report[column].isna()
        )

        expanded_report.loc[
            empty_mask,
            column,
        ] = np.nan

        expanded_report[column] = (
            expanded_report[column]
            .ffill()
        )

    expanded_report[
        "socket_no"
    ] = expanded_report[
        "socket_no"
    ].apply(clean_text)

    socket_rows = expanded_report[
        expanded_report["socket_no"]
        .ne("")
    ].copy()

    socket_rows = socket_rows[
        ~socket_rows["socket_no"]
        .str.casefold()
        .isin(
            {
                "soket no",
                "socket no",
            }
        )
    ].copy()

    socket_rows = socket_rows[
        socket_rows["socket_no"]
        .str.upper()
        .str.startswith("SKT/")
    ].copy()

    if socket_rows.empty:
        raise ValueError(
            "No EPDK socket rows were detected."
        )

    socket_rows[
        "socket_current_type"
    ] = socket_rows[
        "socket_current_type"
    ].apply(
        normalize_socket_current_type
    )

    socket_rows[
        "connector_type"
    ] = socket_rows[
        "connector_type"
    ].apply(clean_text)

    socket_rows[
        "socket_power_kw"
    ] = socket_rows[
        "socket_power_kw"
    ].apply(
        parse_non_negative_float
    )

    socket_rows[
        "is_ac_socket"
    ] = (
        socket_rows[
            "socket_current_type"
        ]
        == "AC"
    ).astype(int)

    socket_rows[
        "is_dc_socket"
    ] = (
        socket_rows[
            "socket_current_type"
        ]
        == "DC"
    ).astype(int)

    socket_rows.drop_duplicates(
        subset=["socket_no"],
        keep="first",
        inplace=True,
    )

    selected_columns = [
        "station_no",
        "station_name",
        "socket_no",
        "socket_current_type",
        "connector_type",
        "socket_power_kw",
        "is_ac_socket",
        "is_dc_socket",
    ]

    socket_rows = socket_rows[
        selected_columns
    ].copy()

    socket_rows.sort_values(
        by=[
            "station_no",
            "socket_no",
        ],
        inplace=True,
    )

    socket_rows.reset_index(
        drop=True,
        inplace=True,
    )

    return socket_rows


def aggregate_socket_features(
    sockets: pd.DataFrame,
) -> pd.DataFrame:
    """Create station-level features from socket records."""

    socket_features = (
        sockets
        .groupby(
            "station_no",
            as_index=False,
        )
        .agg(
            socket_count=(
                "socket_no",
                "nunique",
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
            socket_current_types=(
                "socket_current_type",
                join_unique_text,
            ),
            connector_types=(
                "connector_type",
                join_unique_text,
            ),
        )
    )

    integer_columns = [
        "socket_count",
        "ac_socket_count",
        "dc_socket_count",
        "known_power_socket_count",
    ]

    for column in integer_columns:
        socket_features[column] = (
            socket_features[column]
            .fillna(0)
            .astype(int)
        )

    numeric_columns = [
        "total_socket_power_kw",
        "maximum_socket_power_kw",
    ]

    for column in numeric_columns:
        socket_features[column] = (
            socket_features[column]
            .fillna(0)
            .round(2)
        )

    socket_features[
        "has_ac_socket"
    ] = (
        socket_features[
            "ac_socket_count"
        ]
        > 0
    ).astype(int)

    socket_features[
        "has_dc_socket"
    ] = (
        socket_features[
            "dc_socket_count"
        ]
        > 0
    ).astype(int)

    return socket_features


def build_station_dataset(
    station_rows: pd.DataFrame,
    socket_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Merge station metadata with aggregated socket features."""

    socket_features = (
        aggregate_socket_features(
            socket_rows
        )
    )

    stations = station_rows.merge(
        socket_features,
        on="station_no",
        how="left",
        validate="one_to_one",
    )

    zero_fill_columns = [
        "socket_count",
        "ac_socket_count",
        "dc_socket_count",
        "known_power_socket_count",
        "total_socket_power_kw",
        "maximum_socket_power_kw",
        "has_ac_socket",
        "has_dc_socket",
    ]

    stations[
        zero_fill_columns
    ] = stations[
        zero_fill_columns
    ].fillna(0)

    integer_columns = [
        "socket_count",
        "ac_socket_count",
        "dc_socket_count",
        "known_power_socket_count",
        "has_ac_socket",
        "has_dc_socket",
        "is_green_station",
    ]

    for column in integer_columns:
        stations[column] = (
            stations[column]
            .astype(int)
        )

    stations[
        "total_socket_power_kw"
    ] = stations[
        "total_socket_power_kw"
    ].astype(float).round(2)

    stations[
        "maximum_socket_power_kw"
    ] = stations[
        "maximum_socket_power_kw"
    ].astype(float).round(2)

    stations[
        "socket_current_types"
    ] = stations[
        "socket_current_types"
    ].fillna("")

    stations[
        "connector_types"
    ] = stations[
        "connector_types"
    ].fillna("")

    stations.sort_values(
        by=[
            "station_order",
            "station_no",
        ],
        na_position="last",
        inplace=True,
    )

    stations.reset_index(
        drop=True,
        inplace=True,
    )

    return stations


def validate_outputs(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Validate normalized station and socket outputs."""

    if stations.empty:
        raise ValueError(
            "The normalized station dataset is empty."
        )

    if sockets.empty:
        raise ValueError(
            "The normalized socket dataset is empty."
        )

    if (
        stations["station_no"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate station numbers were found."
        )

    if (
        sockets["socket_no"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate socket numbers were found."
        )

    station_ids = set(
        stations["station_no"]
    )

    socket_station_ids = set(
        sockets["station_no"]
    )

    unknown_station_ids = (
        socket_station_ids
        - station_ids
    )

    if unknown_station_ids:
        raise ValueError(
            "Sockets reference unknown station numbers: "
            f"{sorted(unknown_station_ids)}"
        )

    count_columns = [
        "socket_count",
        "ac_socket_count",
        "dc_socket_count",
        "known_power_socket_count",
    ]

    for column in count_columns:
        values = stations[
            column
        ].to_numpy(dtype=float)

        if (
            values < 0
        ).any():
            raise ValueError(
                f"{column} contains negative values."
            )

        if not np.allclose(
            values,
            np.round(values),
        ):
            raise ValueError(
                f"{column} contains fractional values."
            )

    if (
        stations["ac_socket_count"]
        + stations["dc_socket_count"]
        > stations["socket_count"]
    ).any():
        raise ValueError(
            "AC and DC socket counts exceed "
            "the total socket count."
        )

    socket_count_total = int(
        stations[
            "socket_count"
        ].sum()
    )

    if socket_count_total != len(
        sockets
    ):
        raise ValueError(
            "Aggregated socket count does not match "
            "the normalized socket table."
        )

    if (
        sockets["socket_power_kw"]
        .dropna()
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Negative socket-power values were found."
        )

    print(
        "EPDK report validation "
        "completed successfully."
    )


def save_outputs(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Save normalized CSV and JSON outputs."""

    stations.to_csv(
        STATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sockets.to_csv(
        SOCKET_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    json_output = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_file": (
            RAW_REPORT_PATH.name
        ),
        "station_count": int(
            len(stations)
        ),
        "socket_count": int(
            len(sockets)
        ),
        "stations": (
            stations
            .replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "sockets": (
            sockets
            .replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    JSON_OUTPUT_PATH.write_text(
        json.dumps(
            json_output,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "Station CSV saved:"
    )

    print(
        STATION_OUTPUT_PATH
    )

    print(
        "Socket CSV saved:"
    )

    print(
        SOCKET_OUTPUT_PATH
    )

    print(
        "Normalized JSON saved:"
    )

    print(
        JSON_OUTPUT_PATH
    )


def create_summary(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Create a Markdown summary of the EPDK report."""

    station_count = len(
        stations
    )

    socket_count = len(
        sockets
    )

    ac_socket_count = int(
        sockets[
            "is_ac_socket"
        ].sum()
    )

    dc_socket_count = int(
        sockets[
            "is_dc_socket"
        ].sum()
    )

    unknown_socket_count = (
        socket_count
        - ac_socket_count
        - dc_socket_count
    )

    stations_with_ac = int(
        stations[
            "has_ac_socket"
        ].sum()
    )

    stations_with_dc = int(
        stations[
            "has_dc_socket"
        ].sum()
    )

    stations_with_address = int(
        stations[
            "address"
        ].ne("")
        .sum()
    )

    known_power_count = int(
        sockets[
            "socket_power_kw"
        ].notna()
        .sum()
    )

    total_socket_power = float(
        sockets[
            "socket_power_kw"
        ].sum()
    )

    maximum_socket_power = float(
        sockets[
            "socket_power_kw"
        ].max()
    )

    service_type_counts = (
        stations[
            "service_type"
        ]
        .replace("", "UNKNOWN")
        .value_counts()
    )

    service_lines = "\n".join(
        f"- {service_type}: {int(count):,}"
        for (
            service_type,
            count,
        ) in service_type_counts.items()
    )

    summary = f"""# EPDK Çankaya Charging Report Summary

## Source

- Source organization: EPDK
- Source file: `{RAW_REPORT_PATH.name}`
- File format: legacy Microsoft Excel (`.xls`)
- Generated at: {datetime.now(timezone.utc).isoformat()}

## Normalized Records

- Charging station count: {station_count:,}
- Socket count: {socket_count:,}
- AC socket count: {ac_socket_count:,}
- DC socket count: {dc_socket_count:,}
- Unknown current-type socket count: {unknown_socket_count:,}
- Stations containing an AC socket: {stations_with_ac:,}
- Stations containing a DC socket: {stations_with_dc:,}
- Stations containing an address: {stations_with_address:,}
- Sockets with known power: {known_power_count:,}
- Total reported socket power: {total_socket_power:,.2f} kW
- Maximum reported socket power: {maximum_socket_power:,.2f} kW

## Service Types

{service_lines}

## Generated Outputs

- `data/interim/epdk_cankaya_charging_stations.csv`
- `data/interim/epdk_cankaya_charging_sockets.csv`
- `data/interim/epdk_cankaya_charging_report.json`

## Parsing Method

The downloaded EPDK report contains hierarchical rows. A station
metadata row is followed by one or more socket rows. Station metadata
was forward-filled only for the purpose of associating socket records
with their parent station.

Each station is represented once in the normalized station output.
Each physical EPDK socket is represented once in the normalized socket
output.

## Spatial Limitation

The downloaded report does not contain latitude or longitude columns.
Station addresses must therefore be geocoded and spatially validated
before these records can be merged with OpenStreetMap charging
stations.
"""

    SUMMARY_OUTPUT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "EPDK summary saved:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )


def print_statistics(
    stations: pd.DataFrame,
    sockets: pd.DataFrame,
) -> None:
    """Print the normalized report statistics."""

    print("-" * 70)

    print(
        "Normalized charging-station count: "
        f"{len(stations):,}"
    )

    print(
        "Normalized socket count: "
        f"{len(sockets):,}"
    )

    print(
        "AC socket count: "
        f"{int(sockets['is_ac_socket'].sum()):,}"
    )

    print(
        "DC socket count: "
        f"{int(sockets['is_dc_socket'].sum()):,}"
    )

    print(
        "Stations with AC sockets: "
        f"{int(stations['has_ac_socket'].sum()):,}"
    )

    print(
        "Stations with DC sockets: "
        f"{int(stations['has_dc_socket'].sum()):,}"
    )

    print(
        "Total reported socket power: "
        f"{sockets['socket_power_kw'].sum():,.2f} kW"
    )

    print(
        "Maximum reported socket power: "
        f"{sockets['socket_power_kw'].max():,.2f} kW"
    )

    print(
        "Stations with address: "
        f"{int(stations['address'].ne('').sum()):,}"
    )


def main() -> None:
    """Normalize the downloaded EPDK charging report."""

    print("=" * 70)

    print(
        "VoltSight - "
        "EPDK Çankaya Charging Report Pipeline"
    )

    print("=" * 70)

    try:
        create_output_directories()
        validate_input_file()

        raw_report = load_raw_report()

        station_rows = build_station_rows(
            raw_report
        )

        socket_rows = build_socket_rows(
            raw_report
        )

        stations = build_station_dataset(
            station_rows,
            socket_rows,
        )

        validate_outputs(
            stations,
            socket_rows,
        )

        save_outputs(
            stations,
            socket_rows,
        )

        create_summary(
            stations,
            socket_rows,
        )

        print_statistics(
            stations,
            socket_rows,
        )

    except Exception as error:
        print(
            "EPDK charging report pipeline failed: "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error

    print("=" * 70)

    print(
        "EPDK charging report pipeline "
        "completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()