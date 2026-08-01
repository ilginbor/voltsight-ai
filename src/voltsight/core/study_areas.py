from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECTED_CRS = "EPSG:32636"

SUPPORTED_GRID_SIZES_METERS = (
    250,
    500,
    1_000,
)


@dataclass(
    frozen=True,
    slots=True,
)
class StudyAreaConfig:
    """Configuration for one reproducible spatial study area."""

    key: str
    display_name: str
    boundary_scope: str
    grid_prefix: str
    city_name: str
    district_name: str | None
    default_grid_size_m: int
    projected_crs: str
    osm_place_queries: tuple[str, ...]
    minimum_expected_area_km2: float
    maximum_expected_area_km2: float
    chunk_features_by_district: bool

    def __post_init__(self) -> None:
        """Validate configuration values at import time."""

        if not self.key:
            raise ValueError(
                "Study-area key cannot be empty."
            )

        if self.key != self.key.lower():
            raise ValueError(
                "Study-area key must be lowercase."
            )

        normalized_key = self.key.replace(
            "_",
            "",
        )

        if not normalized_key.isalnum():
            raise ValueError(
                "Study-area key must contain only "
                "letters, numbers or underscores."
            )

        if not self.display_name.strip():
            raise ValueError(
                "Study-area display name cannot be empty."
            )

        if not self.grid_prefix.isalnum():
            raise ValueError(
                "Grid prefix must be alphanumeric."
            )

        if self.grid_prefix != self.grid_prefix.upper():
            raise ValueError(
                "Grid prefix must be uppercase."
            )

        if self.default_grid_size_m not in (
            SUPPORTED_GRID_SIZES_METERS
        ):
            raise ValueError(
                "Unsupported default grid size: "
                f"{self.default_grid_size_m}"
            )

        if not self.projected_crs.startswith(
            "EPSG:"
        ):
            raise ValueError(
                "Projected CRS must use EPSG notation."
            )

        if not self.osm_place_queries:
            raise ValueError(
                "At least one OSM place query is required."
            )

        if any(
            not query.strip()
            for query in self.osm_place_queries
        ):
            raise ValueError(
                "OSM place queries cannot be empty."
            )

        if self.minimum_expected_area_km2 <= 0:
            raise ValueError(
                "Minimum expected area must be positive."
            )

        if (
            self.maximum_expected_area_km2
            <= self.minimum_expected_area_km2
        ):
            raise ValueError(
                "Maximum expected area must be greater "
                "than minimum expected area."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class StudyAreaPaths:
    """Resolved output paths for one grid configuration."""

    boundary_geojson: Path
    grid_gpkg: Path
    grid_geojson: Path
    grid_preview_png: Path
    grid_summary_md: Path
    cache_directory: Path
    grid_layer_name: str
    grid_size_m: int


STUDY_AREAS: dict[str, StudyAreaConfig] = {
    "cankaya": StudyAreaConfig(
        key="cankaya",
        display_name="Çankaya",
        boundary_scope="district",
        grid_prefix="CKY",
        city_name="Ankara",
        district_name="Çankaya",
        default_grid_size_m=250,
        projected_crs=DEFAULT_PROJECTED_CRS,
        osm_place_queries=(
            "Çankaya, Ankara, Türkiye",
            "Cankaya, Ankara, Turkey",
        ),
        minimum_expected_area_km2=350.0,
        maximum_expected_area_km2=650.0,
        chunk_features_by_district=False,
    ),
    "ankara": StudyAreaConfig(
        key="ankara",
        display_name="Ankara",
        boundary_scope="province",
        grid_prefix="ANK",
        city_name="Ankara",
        district_name=None,
        default_grid_size_m=500,
        projected_crs=DEFAULT_PROJECTED_CRS,
        osm_place_queries=(
            "Ankara ili, Türkiye",
            "Ankara Province, Turkey",
            "Ankara, Türkiye",
        ),
        minimum_expected_area_km2=20_000.0,
        maximum_expected_area_km2=30_000.0,
        chunk_features_by_district=True,
    ),
}


def list_study_area_keys() -> tuple[str, ...]:
    """Return available study-area keys in deterministic order."""

    return tuple(
        sorted(
            STUDY_AREAS
        )
    )


def get_study_area(
    study_area_key: str,
) -> StudyAreaConfig:
    """Return one study-area configuration."""

    normalized_key = (
        str(study_area_key)
        .strip()
        .lower()
    )

    try:
        return STUDY_AREAS[
            normalized_key
        ]
    except KeyError as error:
        available = ", ".join(
            list_study_area_keys()
        )

        raise ValueError(
            "Unknown study area "
            f"{study_area_key!r}. "
            f"Available values: {available}"
        ) from error


def resolve_grid_size(
    study_area: StudyAreaConfig,
    requested_grid_size_m: int | None = None,
) -> int:
    """Resolve and validate the grid size for one execution."""

    if requested_grid_size_m is None:
        return study_area.default_grid_size_m

    try:
        grid_size_m = int(
            requested_grid_size_m
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Grid size must be an integer."
        ) from error

    if grid_size_m not in (
        SUPPORTED_GRID_SIZES_METERS
    ):
        supported = ", ".join(
            str(value)
            for value
            in SUPPORTED_GRID_SIZES_METERS
        )

        raise ValueError(
            f"Unsupported grid size: {grid_size_m}. "
            f"Supported values: {supported}"
        )

    return grid_size_m


def build_study_area_paths(
    project_root: Path,
    study_area: StudyAreaConfig,
    requested_grid_size_m: int | None = None,
) -> StudyAreaPaths:
    """Build deterministic grid input and output paths."""

    project_root = Path(
        project_root
    ).resolve()

    grid_size_m = resolve_grid_size(
        study_area,
        requested_grid_size_m,
    )

    raw_directory = (
        project_root
        / "data"
        / "raw"
    )

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    docs_directory = (
        project_root
        / "docs"
    )

    cache_directory = (
        project_root
        / "cache"
        / study_area.key
    )

    grid_stem = (
        f"{study_area.key}_grid_"
        f"{grid_size_m}m"
    )

    return StudyAreaPaths(
        boundary_geojson=(
            raw_directory
            / f"{study_area.key}_boundary_osm.geojson"
        ),
        grid_gpkg=(
            processed_directory
            / f"{grid_stem}.gpkg"
        ),
        grid_geojson=(
            processed_directory
            / f"{grid_stem}.geojson"
        ),
        grid_preview_png=(
            docs_directory
            / f"{grid_stem}_preview.png"
        ),
        grid_summary_md=(
            docs_directory
            / f"{grid_stem}_summary.md"
        ),
        cache_directory=cache_directory,
        grid_layer_name=grid_stem,
        grid_size_m=grid_size_m,
    )


def validate_study_area_registry() -> None:
    """Validate registry-wide uniqueness constraints."""

    keys = [
        config.key
        for config in STUDY_AREAS.values()
    ]

    prefixes = [
        config.grid_prefix
        for config in STUDY_AREAS.values()
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Study-area keys must be unique."
        )

    if len(prefixes) != len(set(prefixes)):
        raise ValueError(
            "Grid prefixes must be unique."
        )


validate_study_area_registry()
