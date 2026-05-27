"""Tests for IborIndex abstract base."""

from __future__ import annotations

import math

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.indexes._mock_curves import FlatForwardMock


def _make_ibor_3m(
    forecast_curve: FlatForwardMock | None = None,
) -> IborIndex:
    return IborIndex(
        "Euribor",
        Period(3, TimeUnit.Months),
        2,
        EURCurrency(),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        True,
        Actual360(),
        forecast_curve,
    )


def test_maturity_date_advances_by_tenor() -> None:
    idx = _make_ibor_3m()
    # 2024-01-19 Fri (value date for Jan 17 fix) → +3M MF/EOM=True under TARGET.
    value = Date.from_ymd(19, Month.January, 2024)
    maturity = idx.maturity_date(value)
    # Expected per probe: serial 45401 = 2024-04-19 Fri.
    assert maturity == Date.from_ymd(19, Month.April, 2024)


def test_forecast_fixing_uses_term_structure() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = _make_ibor_3m(ts)
    # From probe: euribor_3m_fixing = 0.050317307618303497
    fix = Date.from_ymd(17, Month.January, 2024)
    fixing = idx.forecast_fixing(fix)
    tight(fixing, 0.050317307618303497)


def test_forecast_fixing_requires_term_structure() -> None:
    idx = _make_ibor_3m()
    with pytest.raises(LibraryException, match="null term structure"):
        idx.forecast_fixing(Date.from_ymd(17, Month.January, 2024))


def test_business_day_convention_round_trip() -> None:
    idx = _make_ibor_3m()
    assert idx.business_day_convention() == BusinessDayConvention.ModifiedFollowing
    assert idx.end_of_month() is True


def test_forecast_term_structure_inspector() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = _make_ibor_3m(ts)
    assert idx.forecast_term_structure() is ts


def test_clone_swaps_term_structure() -> None:
    a = FlatForwardMock(Date.from_ymd(17, Month.January, 2024), 0.05, Actual360())
    b = FlatForwardMock(Date.from_ymd(17, Month.January, 2024), 0.04, Actual360())
    idx = _make_ibor_3m(a)
    cloned = idx.clone(b)
    assert cloned.forecast_term_structure() is b
    # Original untouched:
    assert idx.forecast_term_structure() is a
    # Same conventions:
    assert cloned.business_day_convention() == idx.business_day_convention()
    assert cloned.end_of_month() == idx.end_of_month()


def test_fixing_uses_forecast_when_today_flag_true() -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = _make_ibor_3m(ts)
    fixing = idx.fixing(eval_date, forecast_todays_fixing=True)
    tight(fixing, 0.050317307618303497)


def test_forecast_fixing_uses_discount_factor_ratio() -> None:
    """Verify the forecast-fixing formula via independent discount-factor math."""
    eval_date = Date.from_ymd(17, Month.January, 2024)
    rate = 0.04
    ts = FlatForwardMock(eval_date, rate, Actual360())
    idx = _make_ibor_3m(ts)
    fix = Date.from_ymd(17, Month.January, 2024)
    d1 = idx.value_date(fix)
    d2 = idx.maturity_date(d1)
    t = idx.day_counter().year_fraction(d1, d2)
    expected = (ts.discount(d1) / ts.discount(d2) - 1.0) / t
    tight(idx.forecast_fixing(fix), expected)
    assert math.isfinite(expected)
