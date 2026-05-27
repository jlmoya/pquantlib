"""Tests for pquantlib.termstructures.volatility_term_structure.

Exercises the VolatilityTermStructure abstract via a minimal concrete
stub (we don't yet have BlackConstantVol at this commit).
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility_term_structure import VolatilityTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class _StubVolTS(VolatilityTermStructure):
    """Minimal concrete VolatilityTermStructure for behavior tests."""

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)

    def min_strike(self) -> float:
        return 0.0

    def max_strike(self) -> float:
        return 1_000.0


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        VolatilityTermStructure(  # type: ignore[abstract]
            business_day_convention=BusinessDayConvention.Following
        )


def test_business_day_convention_round_trip() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    assert ts.business_day_convention() == BusinessDayConvention.ModifiedFollowing


def test_option_date_from_tenor_advances_with_calendar() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=ref,
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    # 1 year out — NullCalendar makes every day a business day.
    d = ts.option_date_from_tenor(Period(1, TimeUnit.Years))
    assert d == Date.from_ymd(15, Month.June, 2027)


def test_check_strike_accepts_in_range() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    ts.check_strike(500.0, extrapolate=False)  # in [0, 1000] — no raise.


def test_check_strike_rejects_below_min() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException, match="outside the curve domain"):
        ts.check_strike(-1.0, extrapolate=False)


def test_check_strike_rejects_above_max() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    with pytest.raises(LibraryException, match="outside the curve domain"):
        ts.check_strike(2_000.0, extrapolate=False)


def test_check_strike_allows_extrapolation_via_arg() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    ts.check_strike(-1.0, extrapolate=True)  # no raise


def test_check_strike_allows_extrapolation_via_flag() -> None:
    ts = _StubVolTS(
        business_day_convention=BusinessDayConvention.Following,
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    ts.enable_extrapolation()
    ts.check_strike(2_000.0, extrapolate=False)  # no raise — flag wins
