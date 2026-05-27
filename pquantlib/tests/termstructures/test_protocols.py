"""Tests for pquantlib.termstructures.protocols (cross-cluster Protocols)."""

from __future__ import annotations

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.protocols import (
    IborIndexProtocol,
    OvernightIndexProtocol,
    SwapIndexProtocol,
    YieldTermStructureProtocol,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# --- mock concretes ---------------------------------------------------------


class _MockYieldCurve:
    def reference_date(self) -> Date:
        return Date.from_ymd(15, Month.June, 2026)

    def max_date(self) -> Date:
        return Date.from_ymd(15, Month.June, 2076)

    def day_counter(self) -> DayCounter:
        return Actual360()

    def discount(self, t: float | Date, extrapolate: bool = False) -> float:
        del t, extrapolate
        return 0.95

    def zero_rate(self, arg: float | Date, extrapolate: bool = False) -> float:
        del arg, extrapolate
        return 0.05

    def forward_rate(
        self,
        t1: float | Date,
        t2: float | Date,
        extrapolate: bool = False,
    ) -> float:
        del t1, t2, extrapolate
        return 0.05


class _MockIborIndex:
    def name(self) -> str:
        return "MOCK3M"

    def tenor(self) -> Period:
        return Period(3, TimeUnit.Months)

    def fixing_days(self) -> int:
        return 2

    def currency(self) -> object:
        return None

    def fixing_calendar(self) -> Calendar:
        return WeekendsOnly()

    def day_counter(self) -> DayCounter:
        return Actual360()

    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention.ModifiedFollowing

    def end_of_month(self) -> bool:
        return False

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        del fixing_date, forecast_todays_fixing
        return 0.035

    def maturity_date(self, value_date: Date) -> Date:
        return value_date + Period(3, TimeUnit.Months)


class _MockOvernightIndex:
    def name(self) -> str:
        return "MOCKOIS"

    def currency(self) -> object:
        return None

    def fixing_calendar(self) -> Calendar:
        return WeekendsOnly()

    def day_counter(self) -> DayCounter:
        return Actual360()

    def fixing(self, fixing_date: Date, forecast_todays_fixing: bool = False) -> float:
        del fixing_date, forecast_todays_fixing
        return 0.04


class _MockSwapIndex:
    def name(self) -> str:
        return "MOCKSWAP10Y"

    def tenor(self) -> Period:
        return Period(10, TimeUnit.Years)

    def fixing_days(self) -> int:
        return 2

    def currency(self) -> object:
        return None

    def fixed_leg_tenor(self) -> Period:
        return Period(1, TimeUnit.Years)

    def fixed_leg_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention.Unadjusted

    def fixed_leg_day_counter(self) -> DayCounter:
        return Actual360()

    def ibor_index(self) -> IborIndexProtocol:
        return _MockIborIndex()


# --- structural-typing tests ------------------------------------------------


def test_yield_term_structure_protocol_satisfied_structurally() -> None:
    curve = _MockYieldCurve()
    assert isinstance(curve, YieldTermStructureProtocol)


def test_ibor_index_protocol_satisfied_structurally() -> None:
    idx = _MockIborIndex()
    assert isinstance(idx, IborIndexProtocol)


def test_overnight_index_protocol_satisfied_structurally() -> None:
    idx = _MockOvernightIndex()
    assert isinstance(idx, OvernightIndexProtocol)


def test_swap_index_protocol_satisfied_structurally() -> None:
    idx = _MockSwapIndex()
    assert isinstance(idx, SwapIndexProtocol)


def test_missing_methods_fail_protocol_check() -> None:
    class _Incomplete:
        def name(self) -> str:
            return "x"

    obj = _Incomplete()
    assert not isinstance(obj, IborIndexProtocol)
