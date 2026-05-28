"""Tests for InflationIndex / ZeroInflationIndex / YoYInflationIndex abstracts.

Cross-validates against ``migration-harness/references/l7a/foundations.json``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.inflation.inflation_index import (
    InflationIndex,
    YoYInflationIndex,
    ZeroInflationIndex,
    inflation_period,
)
from pquantlib.indexes.inflation.region import Region
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load_reference("l7a/foundations")


# ---- inflation_period helper -----------------------------------------


def test_inflation_period_monthly_matches_probe(reference: dict[str, Any]) -> None:
    """Monthly inflation period brackets the calendar month containing the date."""
    expected = reference["inflation_period"]["monthly_2020_05_15"]
    start, end = inflation_period(Date.from_ymd(15, Month.May, 2020), Frequency.Monthly)
    assert start.serial == expected["start_serial"]
    assert end.serial == expected["end_serial"]


def test_inflation_period_quarterly_matches_probe(reference: dict[str, Any]) -> None:
    """Quarterly groups three months together (Jul → Jul..Sep)."""
    expected = reference["inflation_period"]["quarterly_2021_07_20"]
    start, end = inflation_period(Date.from_ymd(20, Month.July, 2021), Frequency.Quarterly)
    assert start.serial == expected["start_serial"]
    assert end.serial == expected["end_serial"]


def test_inflation_period_semiannual_matches_probe(reference: dict[str, Any]) -> None:
    expected = reference["inflation_period"]["semiannual_2019_11_30"]
    start, end = inflation_period(Date.from_ymd(30, Month.November, 2019), Frequency.Semiannual)
    assert start.serial == expected["start_serial"]
    assert end.serial == expected["end_serial"]


def test_inflation_period_annual_matches_probe(reference: dict[str, Any]) -> None:
    expected = reference["inflation_period"]["annual_2022_03_01"]
    start, end = inflation_period(Date.from_ymd(1, Month.March, 2022), Frequency.Annual)
    assert start.serial == expected["start_serial"]
    assert end.serial == expected["end_serial"]


def test_inflation_period_every_fourth_month_matches_probe(reference: dict[str, Any]) -> None:
    expected = reference["inflation_period"]["every_fourth_month_2020_08_10"]
    start, end = inflation_period(
        Date.from_ymd(10, Month.August, 2020), Frequency.EveryFourthMonth
    )
    assert start.serial == expected["start_serial"]
    assert end.serial == expected["end_serial"]


def test_inflation_period_unsupported_frequency_raises() -> None:
    """OtherFrequency must raise a LibraryException."""
    with pytest.raises(LibraryException):
        inflation_period(Date.from_ymd(1, Month.January, 2020), Frequency.OtherFrequency)


# ---- ZeroInflationIndex (anchored on a stub family) ------------------


def _zero_stub() -> ZeroInflationIndex:
    return ZeroInflationIndex(
        family_name="HICP",
        region=Region.Europe,
        revised=False,
        frequency=Frequency.Monthly,
        availability_lag=Period(1, TimeUnit.Months),
        currency=EURCurrency(),
    )


def test_zero_inflation_index_inspectors() -> None:
    idx = _zero_stub()
    assert idx.family_name() == "HICP"
    assert idx.region() == Region.Europe
    assert idx.revised() is False
    assert idx.frequency() == Frequency.Monthly
    assert idx.availability_lag() == Period(1, TimeUnit.Months)
    assert idx.currency().code == "EUR"
    assert idx.name() == "EU HICP"
    # ZeroInflationIndex always non-interpolated (C++ parity).
    assert idx.interpolated() is False
    # InflationIndex uses NullCalendar.
    assert isinstance(idx.fixing_calendar(), NullCalendar)
    assert idx.is_valid_fixing_date(Date.from_ymd(15, Month.May, 2020)) is True


def test_zero_inflation_index_maturity_date_end_of_month() -> None:
    """Monthly maturity = end of the month containing the fixing date."""
    idx = _zero_stub()
    matures = idx.maturity_date(Date.from_ymd(15, Month.May, 2020))
    # End of May 2020 = serial 43982 from probe.
    assert matures.serial == 43982


def test_zero_inflation_index_past_fixing_lookup_and_missing() -> None:
    """Stored fixings are returned by ``fixing()`` at any date in the period."""
    idx = _zero_stub()
    # Storing on day 5 (period start = 1) and retrieving via day 20:
    idx.add_fixing(Date.from_ymd(1, Month.May, 2020), 105.5)
    assert idx.fixing(Date.from_ymd(20, Month.May, 2020)) == 105.5
    # Unknown period raises.
    with pytest.raises(LibraryException):
        idx.fixing(Date.from_ymd(15, Month.September, 2020))
    idx.clear_fixings()


# ---- YoYInflationIndex (quoted + ratio modes) ------------------------


def test_yoy_inflation_index_quoted_mode_inspectors() -> None:
    idx = YoYInflationIndex(
        family_name="YY_HICP",
        region=Region.Europe,
        revised=False,
        interpolated=False,
        frequency=Frequency.Monthly,
        availability_lag=Period(1, TimeUnit.Months),
        currency=EURCurrency(),
    )
    assert idx.ratio() is False
    assert idx.interpolated() is False
    assert idx.underlying_index() is None
    assert idx.name() == "EU YY_HICP"


def test_yoy_inflation_index_ratio_mode_inspectors() -> None:
    underlying = _zero_stub()
    yoy = YoYInflationIndex.from_underlying(underlying, interpolated=False)
    assert yoy.ratio() is True
    assert yoy.underlying_index() is underlying
    # C++ parity: ratio-mode family_name = "YYR_" + underlying.family_name().
    assert yoy.family_name() == "YYR_HICP"
    assert yoy.region() == Region.Europe
    assert yoy.frequency() == Frequency.Monthly
    assert yoy.name() == "EU YYR_HICP"


def test_yoy_inflation_index_ratio_fixing_from_underlying() -> None:
    """In ratio mode, YoY fixing = underlying(d) / underlying(d-12m) - 1."""
    underlying = _zero_stub()
    # Store two past fixings 12 months apart at their period starts.
    underlying.add_fixing(Date.from_ymd(1, Month.May, 2019), 100.0)
    underlying.add_fixing(Date.from_ymd(1, Month.May, 2020), 102.5)
    yoy = YoYInflationIndex.from_underlying(underlying, interpolated=False)
    # 102.5 / 100.0 - 1.0 = 0.025
    rate = yoy.fixing(Date.from_ymd(15, Month.May, 2020))
    tight(rate, 0.025)
    underlying.clear_fixings()


# ---- abstract InflationIndex shouldn't be directly instantiable ------


def test_inflation_index_abstract_cannot_be_instantiated() -> None:
    """The abstract base must require a concrete ``fixing()`` override."""
    with pytest.raises(TypeError):
        # cast bypass: pyright would otherwise reject this construction.
        cast(Any, InflationIndex)(
            "HICP",
            Region.Europe,
            False,
            False,
            Frequency.Monthly,
            Period(1, TimeUnit.Months),
            EURCurrency(),
        )
