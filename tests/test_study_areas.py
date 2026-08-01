from __future__ import annotations

from pathlib import Path

import pytest

from voltsight.core.study_areas import (
    STUDY_AREAS,
    build_study_area_paths,
    get_study_area,
    list_study_area_keys,
    resolve_grid_size,
)


def test_available_study_area_keys() -> None:
    """Both pilot and province-scale study areas must exist."""

    assert list_study_area_keys() == (
        "ankara",
        "cankaya",
    )


def test_lookup_is_case_insensitive() -> None:
    """Study-area lookup must normalize user input."""

    config = get_study_area(
        "  ANKARA  "
    )

    assert config.key == "ankara"
    assert config.display_name == "Ankara"


def test_unknown_study_area_is_rejected() -> None:
    """Unsupported study-area values must fail clearly."""

    with pytest.raises(
        ValueError,
        match="Unknown study area",
    ):
        get_study_area(
            "istanbul"
        )


def test_default_grid_sizes() -> None:
    """Çankaya remains detailed while Ankara defaults to 500 m."""

    cankaya = get_study_area(
        "cankaya"
    )

    ankara = get_study_area(
        "ankara"
    )

    assert cankaya.default_grid_size_m == 250
    assert ankara.default_grid_size_m == 500


def test_expected_area_ranges() -> None:
    """Boundary validation ranges must separate district and province."""

    cankaya = get_study_area(
        "cankaya"
    )

    ankara = get_study_area(
        "ankara"
    )

    assert (
        cankaya.minimum_expected_area_km2
        < 500
        < cankaya.maximum_expected_area_km2
    )

    assert (
        ankara.minimum_expected_area_km2
        < 25_000
        < ankara.maximum_expected_area_km2
    )


def test_grid_prefixes_are_unique() -> None:
    """Each study area must generate distinguishable grid IDs."""

    prefixes = [
        config.grid_prefix
        for config in STUDY_AREAS.values()
    ]

    assert len(prefixes) == len(
        set(prefixes)
    )

    assert get_study_area(
        "cankaya"
    ).grid_prefix == "CKY"

    assert get_study_area(
        "ankara"
    ).grid_prefix == "ANK"


def test_cankaya_default_paths(
    tmp_path: Path,
) -> None:
    """Existing Çankaya naming must remain reproducible."""

    paths = build_study_area_paths(
        tmp_path,
        get_study_area(
            "cankaya"
        ),
    )

    assert paths.grid_size_m == 250

    assert paths.boundary_geojson.name == (
        "cankaya_boundary_osm.geojson"
    )

    assert paths.grid_gpkg.name == (
        "cankaya_grid_250m.gpkg"
    )

    assert paths.grid_layer_name == (
        "cankaya_grid_250m"
    )


def test_ankara_default_paths(
    tmp_path: Path,
) -> None:
    """Ankara outputs must use province-scale deterministic names."""

    paths = build_study_area_paths(
        tmp_path,
        get_study_area(
            "ankara"
        ),
    )

    assert paths.grid_size_m == 500

    assert paths.boundary_geojson.name == (
        "ankara_boundary_osm.geojson"
    )

    assert paths.grid_gpkg.name == (
        "ankara_grid_500m.gpkg"
    )

    assert paths.grid_layer_name == (
        "ankara_grid_500m"
    )

    assert paths.cache_directory.name == (
        "ankara"
    )


def test_ankara_grid_size_override(
    tmp_path: Path,
) -> None:
    """Ankara must support an explicit 250 m detailed run."""

    paths = build_study_area_paths(
        tmp_path,
        get_study_area(
            "ankara"
        ),
        requested_grid_size_m=250,
    )

    assert paths.grid_size_m == 250

    assert paths.grid_gpkg.name == (
        "ankara_grid_250m.gpkg"
    )

    assert paths.grid_layer_name == (
        "ankara_grid_250m"
    )


def test_invalid_grid_size_is_rejected() -> None:
    """Only explicitly supported grid resolutions may be used."""

    ankara = get_study_area(
        "ankara"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported grid size",
    ):
        resolve_grid_size(
            ankara,
            300,
        )
