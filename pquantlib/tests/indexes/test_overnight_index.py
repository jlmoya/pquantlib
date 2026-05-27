"""Tests for OvernightIndex abstract base."""

from __future__ import annotations

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.indexes._mock_curves import FlatForwardMock


def _make_overnight() -> OvernightIndex:
    return OvernightIndex(
        "Stub", 0, EURCurrency(), TARGET(), Actual360(),
    )


def test_overnight_index_tenor_is_one_day() -> None:
    idx = _make_overnight()
    assert idx.tenor() == Period(1, TimeUnit.Days)


def test_overnight_index_default_conventions() -> None:
    idx = _make_overnight()
    assert idx.business_day_convention() == BusinessDayConvention.Following
    assert idx.end_of_month() is False
    assert idx.fixing_days() == 0


def test_overnight_index_name_uses_on_suffix() -> None:
    idx = _make_overnight()
    assert idx.name() == "StubON Actual/360"


def test_overnight_index_value_date_is_same_as_fixing_date() -> None:
    idx = _make_overnight()
    fix = Date.from_ymd(17, Month.January, 2024)
    # With fixing_days=0, valueDate == fixingDate.
    assert idx.value_date(fix) == fix


def test_overnight_index_maturity_is_next_business_day() -> None:
    idx = _make_overnight()
    fix = Date.from_ymd(17, Month.January, 2024)
    value = idx.value_date(fix)
    maturity = idx.maturity_date(value)
    # +1 day from Wed Jan 17 → Thu Jan 18.
    assert maturity == Date.from_ymd(18, Month.January, 2024)


def test_overnight_index_forecast_fixing() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = OvernightIndex("Stub", 0, EURCurrency(), TARGET(), Actual360(), ts)
    fix = eval_date
    fixing = idx.forecast_fixing(fix)
    # Probe (with SOFR — same calendar shape): 0.05000347238...
    # Compute discount-ratio directly here as cross-check:
    d1 = idx.value_date(fix)
    d2 = idx.maturity_date(d1)
    t = idx.day_counter().year_fraction(d1, d2)
    expected = (ts.discount(d1) / ts.discount(d2) - 1.0) / t
    tight(fixing, expected)


def test_overnight_index_clone_returns_overnight_subtype() -> None:
    a = FlatForwardMock(Date.from_ymd(17, Month.January, 2024), 0.05, Actual360())
    b = FlatForwardMock(Date.from_ymd(17, Month.January, 2024), 0.04, Actual360())
    idx = OvernightIndex("Stub", 0, EURCurrency(), TARGET(), Actual360(), a)
    cloned = idx.clone(b)
    assert isinstance(cloned, OvernightIndex)
    assert cloned.forecast_term_structure() is b
    assert cloned.tenor() == Period(1, TimeUnit.Days)
