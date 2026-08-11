from __future__ import annotations


TARGET_COLUMN = "has_existing_charging_station"

ROAD_FEATURE_COLUMNS = (
    "road_length_m",
    "road_segment_count",
    "main_road_length_m",
    "main_road_segment_count",
    "road_density_km_per_km2",
    "distance_to_main_road_m",
)

PARKING_FEATURE_COLUMNS = (
    "parking_count",
    "parking_area_m2",
    "parking_area_ratio",
    "distance_to_nearest_parking_m",
    "parking_count_within_500m",
    "parking_count_within_1000m",
    "known_parking_capacity",
    "parking_capacity_record_count",
)

CHARGING_CONTEXT_COLUMNS = (
    "distance_to_nearest_charging_station_m",
    "charging_station_count_within_1000m",
    "charging_station_count_within_2000m",
    "ac_station_count_within_1000m",
    "dc_station_count_within_1000m",
)

CHARGING_LEAKAGE_COLUMNS = (
    "charging_station_count",
    "known_charging_capacity",
    "charging_capacity_record_count",
)

HISTORICAL_FULL_14_FEATURE_COLUMNS = (
    *ROAD_FEATURE_COLUMNS,
    *PARKING_FEATURE_COLUMNS,
)

REDUNDANT_SCALE_FEATURE_COLUMNS = (
    "road_length_m",
    "parking_area_m2",
)

NORMALIZED_12_FEATURE_COLUMNS = tuple(
    feature
    for feature in HISTORICAL_FULL_14_FEATURE_COLUMNS
    if feature not in REDUNDANT_SCALE_FEATURE_COLUMNS
)

ACTIVITY_CONTEXT_FEATURE_COLUMNS = (
    "poi_count",
    "poi_count_within_1000m",
    "poi_count_within_2000m",
)

CANONICAL_ML_FEATURE_COLUMNS = (
    *NORMALIZED_12_FEATURE_COLUMNS,
    *ACTIVITY_CONTEXT_FEATURE_COLUMNS,
)

FEATURE_SET_REGISTRY = {
    "historical_full_14": HISTORICAL_FULL_14_FEATURE_COLUMNS,
    "normalized_12": NORMALIZED_12_FEATURE_COLUMNS,
    "canonical_activity_15": CANONICAL_ML_FEATURE_COLUMNS,
}


def validate_feature_architecture() -> None:
    """Validate the fixed Ankara ML feature-set architecture."""

    expected_lengths = {
        "historical_full_14": 14,
        "normalized_12": 12,
        "canonical_activity_15": 15,
    }

    for (
        feature_set_name,
        expected_length,
    ) in expected_lengths.items():
        feature_columns = FEATURE_SET_REGISTRY[
            feature_set_name
        ]

        if len(
            feature_columns
        ) != expected_length:
            raise ValueError(
                f"{feature_set_name} must contain "
                f"{expected_length} predictors."
            )

        if len(
            set(
                feature_columns
            )
        ) != len(
            feature_columns
        ):
            raise ValueError(
                f"{feature_set_name} contains duplicate predictors."
            )

    if (
        set(
            REDUNDANT_SCALE_FEATURE_COLUMNS
        )
        & set(
            NORMALIZED_12_FEATURE_COLUMNS
        )
    ):
        raise ValueError(
            "Normalized-12 still contains a removed scale duplicate."
        )

    if not set(
        ACTIVITY_CONTEXT_FEATURE_COLUMNS
    ).issubset(
        CANONICAL_ML_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Canonical-15 is missing required activity context."
        )

    if not set(
        NORMALIZED_12_FEATURE_COLUMNS
    ).issubset(
        CANONICAL_ML_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Canonical-15 must extend normalized-12."
        )

    forbidden = {
        *CHARGING_CONTEXT_COLUMNS,
        *CHARGING_LEAKAGE_COLUMNS,
    }

    leaked = forbidden & set(
        CANONICAL_ML_FEATURE_COLUMNS
    )

    if leaked:
        raise ValueError(
            "Charging-derived leakage entered the canonical ML set: "
            f"{sorted(leaked)}"
        )


validate_feature_architecture()
