"""Tests for the 5 zero-inflation region concretes + 4 YoY siblings.

Cross-validates against the C++ default-construction probe in
``migration-harness/references/l7a/foundations.json``. Every accessor
is compared directly to the C++ values; this guarantees the Python
default-market parameterization matches v1.42.1 bit-by-bit.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.indexes.inflation.eu_hicp import EUHICP, YoYEUHICP
from pquantlib.indexes.inflation.fr_hicp import FRHICP, YoYFRHICP
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
)
from pquantlib.indexes.inflation.region import Region
from pquantlib.indexes.inflation.uk_hicp import UKHICP
from pquantlib.indexes.inflation.uk_rpi import UKRPI, YoYUKRPI
from pquantlib.indexes.inflation.us_cpi import USCPI, YoYUSCPI
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load_reference("l7a/foundations")


def _check_zero(idx: ZeroInflationIndex, probe: dict[str, Any]) -> None:
    """Assert a ZeroInflationIndex matches a probe entry."""
    assert idx.name() == probe["name"]
    assert idx.family_name() == probe["family_name"]
    assert idx.region().region_name() == probe["region_name"]
    assert idx.region().region_code() == probe["region_code"]
    assert idx.revised() is probe["revised"]
    assert int(idx.frequency()) == probe["frequency"]
    assert idx.availability_lag().length == probe["availability_lag_months"]
    assert idx.currency().code == probe["currency_code"]
    # ZeroInflationIndex always non-interpolated.
    assert idx.interpolated() is False


def _check_yoy(idx: YoYInflationIndex, probe: dict[str, Any]) -> None:
    """Assert a YoYInflationIndex matches a probe entry."""
    assert idx.name() == probe["name"]
    assert idx.family_name() == probe["family_name"]
    assert idx.region().region_name() == probe["region_name"]
    assert idx.region().region_code() == probe["region_code"]
    assert idx.revised() is probe["revised"]
    assert int(idx.frequency()) == probe["frequency"]
    assert idx.availability_lag().length == probe["availability_lag_months"]
    assert idx.currency().code == probe["currency_code"]
    assert idx.interpolated() is probe["interpolated"]
    assert idx.ratio() is probe["ratio"]


# ---- zero-inflation concretes ---------------------------------------


def test_euhicp_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_zero(EUHICP(), reference["zero_indexes"]["EUHICP"])
    assert EUHICP().region() == Region.Europe


def test_frhicp_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_zero(FRHICP(), reference["zero_indexes"]["FRHICP"])
    assert FRHICP().region() == Region.France


def test_ukrpi_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_zero(UKRPI(), reference["zero_indexes"]["UKRPI"])
    assert UKRPI().region() == Region.UnitedKingdom


def test_ukhicp_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_zero(UKHICP(), reference["zero_indexes"]["UKHICP"])
    assert UKHICP().region() == Region.UnitedKingdom


def test_uscpi_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_zero(USCPI(), reference["zero_indexes"]["USCPI"])
    assert USCPI().region() == Region.UnitedStates


# ---- YoY siblings (no YYUKHICP — does not exist upstream) -----------


def test_yoyeu_hicp_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_yoy(YoYEUHICP(), reference["yoy_indexes"]["YYEUHICP"])


def test_yoyfr_hicp_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_yoy(YoYFRHICP(), reference["yoy_indexes"]["YYFRHICP"])


def test_yoyuk_rpi_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_yoy(YoYUKRPI(), reference["yoy_indexes"]["YYUKRPI"])


def test_yoyus_cpi_default_matches_cpp(reference: dict[str, Any]) -> None:
    _check_yoy(YoYUSCPI(), reference["yoy_indexes"]["YYUSCPI"])


# ---- interpolated kwarg threads through -----------------------------


def test_yoy_index_interpolated_kwarg_is_respected() -> None:
    """Passing interpolated=True on the YoY concrete reaches the abstract."""
    yoy = YoYEUHICP(interpolated=True)
    assert yoy.interpolated() is True
    assert yoy.ratio() is False  # quoted mode regardless of interpolation


# ---- monthly frequency means each index maturities at end-of-month --


def test_us_cpi_maturity_date_is_end_of_month() -> None:
    """Monthly USCPI maturity_date returns the last day of the input month."""
    matures = USCPI().maturity_date(Date.from_ymd(15, Month.July, 2021))
    assert matures.day_of_month() == 31
    assert matures.month() == Month.July
    assert matures.year() == 2021


def test_all_concretes_share_monthly_frequency() -> None:
    """C++ ships these 5 concretes with Frequency.Monthly by default."""
    for ctor in (EUHICP, FRHICP, UKRPI, UKHICP, USCPI):
        assert ctor().frequency() == Frequency.Monthly
