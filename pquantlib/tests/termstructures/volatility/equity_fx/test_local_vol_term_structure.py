"""Tests for the LocalVolTermStructure abstract base."""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class _StubLocalVol(LocalVolTermStructure):
    """Minimal LocalVolTermStructure stub for behavior tests."""

    def max_date(self) -> Date:
        return Date.from_ymd(31, Month.December, 2050)

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        # Returns t + S so we can verify both args reach the impl.
        return t + underlying_level


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        LocalVolTermStructure()  # type: ignore[abstract]


def test_default_business_day_convention_is_following() -> None:
    ts = _StubLocalVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    assert ts.business_day_convention() == BusinessDayConvention.Following


def test_local_vol_at_time_passes_t_and_strike_to_impl() -> None:
    ts = _StubLocalVol(
        reference_date=Date.from_ymd(15, Month.June, 2026),
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )
    tolerance.tight(ts.local_vol_at_time(1.5, 100.0), 101.5)


def test_local_vol_by_date_converts_to_time() -> None:
    ref = Date.from_ymd(15, Month.June, 2026)
    one_year = Date.from_ymd(15, Month.June, 2027)
    ts = _StubLocalVol(reference_date=ref, calendar=NullCalendar(), day_counter=Actual365Fixed())
    # t(1y) == 1.0 under Actual/365 Fixed; impl returns 1.0 + 100.0 = 101.0
    tolerance.tight(ts.local_vol(one_year, 100.0), 101.0)
