"""Cross-validate the region hierarchy + inflation indexes against C++ v1.43.

Probe: ``v143/indexes/inflation``.

Regions are pure ``(name, code)`` payloads, but the name feeds straight into
``InflationIndex.name()`` (``"<region> <family>"``), which is the IndexManager
key past fixings are stored under — so a wrong region string silently renames
every index built on it. Both halves are walked: the region payloads on their
own, and the full identity of each concrete index.

The test walks the reference JSON by key and asserts the key sets match, so no
probe entry can be silently skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.indexes.inflation.au_cpi import AUCPI, YYAUCPI
from pquantlib.indexes.inflation.eu_hicp import (
    EUHICP,
    EUHICPXT,
    YYEUHICP,
    YYEUHICPXT,
)
from pquantlib.indexes.inflation.fr_hicp import FRHICP, YYFRHICP
from pquantlib.indexes.inflation.inflation_index import InflationIndex
from pquantlib.indexes.inflation.region import (
    AustraliaRegion,
    CustomRegion,
    EURegion,
    FranceRegion,
    GenericRegion,
    Region,
    UKRegion,
    USRegion,
    ZARegion,
)
from pquantlib.indexes.inflation.uk_hicp import UKHICP
from pquantlib.indexes.inflation.uk_rpi import UKRPI, YYUKRPI
from pquantlib.indexes.inflation.us_cpi import USCPI, YYUSCPI
from pquantlib.indexes.inflation.za_cpi import YYZACPI, ZACPI
from pquantlib.testing import reference_reader
from pquantlib.time.frequency import Frequency

_REGIONS: dict[str, Region] = {
    "australia": AustraliaRegion(),
    "eu": EURegion(),
    "france": FranceRegion(),
    "uk": UKRegion(),
    "us": USRegion(),
    "za": ZARegion(),
    "generic": GenericRegion(),
    "custom": CustomRegion("Atlantis", "AT"),
}

_ZERO: dict[str, Callable[[], InflationIndex]] = {
    "AUCPI_quarterly": lambda: AUCPI(Frequency.Quarterly, False),
    "AUCPI_monthly_revised": lambda: AUCPI(Frequency.Monthly, True),
    "EUHICP": EUHICP,
    "EUHICPXT": EUHICPXT,
    "FRHICP": FRHICP,
    "UKHICP": UKHICP,
    "UKRPI": UKRPI,
    "USCPI": USCPI,
    "ZACPI": ZACPI,
}

_YOY: dict[str, Callable[[], InflationIndex]] = {
    "YYAUCPI_quarterly": lambda: YYAUCPI(Frequency.Quarterly, False),
    "YYAUCPI_monthly_revised": lambda: YYAUCPI(Frequency.Monthly, True),
    "YYEUHICP": YYEUHICP,
    "YYEUHICPXT": YYEUHICPXT,
    "YYFRHICP": YYFRHICP,
    "YYUKRPI": YYUKRPI,
    "YYUSCPI": YYUSCPI,
    "YYZACPI": YYZACPI,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/indexes/inflation")


def test_every_reference_key_is_covered(cpp: dict[str, Any]) -> None:
    """No probe entry may be silently skipped by the walks below."""
    assert set(cpp["regions"]) == set(_REGIONS)
    assert set(cpp["zero_indexes"]) == set(_ZERO)
    assert set(cpp["yoy_indexes"]) == set(_YOY)


@pytest.mark.parametrize("key", sorted(_REGIONS))
def test_region_payload_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    ref = cpp["regions"][key]
    region = _REGIONS[key]
    assert region.name() == ref["name"]
    assert region.code() == ref["code"]


def test_region_equality_matches_cpp(cpp: dict[str, Any]) -> None:
    """C++ ``operator==`` compares ``name()`` and ignores ``code()``."""
    ref = cpp["region_equality"]
    assert (EURegion() == EURegion()) is ref["eu_vs_eu"]
    assert (EURegion() == USRegion()) is ref["eu_vs_us"]
    assert (CustomRegion("EU", "XX") == EURegion()) is ref[
        "custom_same_name_different_code_vs_eu"
    ]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # pins v1.43-deprecated interpolated()
@pytest.mark.parametrize("key", sorted(_ZERO))
def test_zero_inflation_index_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    _check_index(_ZERO[key](), cpp["zero_indexes"][key])
    # ZeroInflationIndex is always non-interpolated.
    assert _ZERO[key]().interpolated() is False  # pyright: ignore[reportDeprecated]


@pytest.mark.parametrize("key", sorted(_YOY))
def test_yoy_inflation_index_matches_cpp(key: str, cpp: dict[str, Any]) -> None:
    _check_index(_YOY[key](), cpp["yoy_indexes"][key])


def _check_index(idx: InflationIndex, ref: dict[str, Any]) -> None:
    assert idx.name() == ref["name"]
    assert idx.family_name() == ref["family_name"]
    assert idx.region().name() == ref["region_name"]
    assert idx.region().code() == ref["region_code"]
    assert idx.revised() is ref["revised"]
    assert int(idx.frequency()) == ref["frequency"]
    assert idx.availability_lag().length == ref["availability_lag_months"]
    assert idx.currency().code == ref["currency_code"]
