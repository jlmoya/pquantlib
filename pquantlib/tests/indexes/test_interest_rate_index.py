"""Tests for InterestRateIndex abstract base."""

from __future__ import annotations

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.interest_rate_index import (
    InterestRateIndex,
    _short_period,  # pyright: ignore[reportPrivateUsage]
)
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class _ConcreteIRI(InterestRateIndex):
    """Minimal subclass that fulfils abstract methods so we can test the base."""

    def maturity_date(self, value_date: Date) -> Date:
        return value_date

    def forecast_fixing(self, fixing_date: Date) -> float:
        del fixing_date
        return 0.0


def test_short_period_letters() -> None:
    assert _short_period(Period(3, TimeUnit.Months)) == "3M"
    assert _short_period(Period(1, TimeUnit.Years)) == "1Y"
    assert _short_period(Period(1, TimeUnit.Weeks)) == "1W"
    assert _short_period(Period(1, TimeUnit.Days)) == "1D"


def test_name_uses_short_period_for_non_daily_tenors() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    assert idx.name() == "Euribor3M Actual/360"


def test_name_normalizes_twelve_months_to_one_year() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(12, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    assert idx.tenor() == Period(1, TimeUnit.Years)
    assert idx.name() == "Euribor1Y Actual/360"


def test_name_uses_on_for_daily_tenor_with_fixing_days_0() -> None:
    idx = _ConcreteIRI(
        "Eonia", Period(1, TimeUnit.Days), 0,
        EURCurrency(), TARGET(), Actual360(),
    )
    assert idx.name() == "EoniaON Actual/360"


def test_name_uses_tn_for_daily_tenor_with_fixing_days_1() -> None:
    idx = _ConcreteIRI(
        "Stub", Period(1, TimeUnit.Days), 1,
        EURCurrency(), TARGET(), Actual360(),
    )
    assert idx.name() == "StubTN Actual/360"


def test_name_uses_sn_for_daily_tenor_with_fixing_days_2() -> None:
    idx = _ConcreteIRI(
        "Stub", Period(1, TimeUnit.Days), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    assert idx.name() == "StubSN Actual/360"


def test_inspectors_round_trip_constructor_args() -> None:
    cal = TARGET()
    dc = Actual360()
    cur = EURCurrency()
    idx = _ConcreteIRI("Euribor", Period(6, TimeUnit.Months), 2, cur, cal, dc)
    assert idx.family_name() == "Euribor"
    assert idx.tenor() == Period(6, TimeUnit.Months)
    assert idx.fixing_days() == 2
    assert idx.currency() == cur
    assert idx.fixing_calendar() is cal
    assert idx.day_counter() is dc


def test_is_valid_fixing_date_checks_business_days() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    # 2024-01-01 was a TARGET holiday (New Year). 2024-01-02 was a Tuesday.
    assert not idx.is_valid_fixing_date(Date.from_ymd(1, Month.January, 2024))
    assert idx.is_valid_fixing_date(Date.from_ymd(2, Month.January, 2024))


def test_value_date_advances_by_fixing_days() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    fix = Date.from_ymd(17, Month.January, 2024)  # Wed
    # +2 business days on TARGET: Fri Jan 19 2024.
    assert idx.value_date(fix) == Date.from_ymd(19, Month.January, 2024)


def test_fixing_date_rolls_back_by_fixing_days() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    value = Date.from_ymd(19, Month.January, 2024)  # Fri
    # -2 business days on TARGET: Wed Jan 17.
    assert idx.fixing_date(value) == Date.from_ymd(17, Month.January, 2024)


def test_value_date_rejects_holiday() -> None:
    idx = _ConcreteIRI(
        "Euribor", Period(3, TimeUnit.Months), 2,
        EURCurrency(), TARGET(), Actual360(),
    )
    with pytest.raises(LibraryException):
        idx.value_date(Date.from_ymd(1, Month.January, 2024))


def test_name_uses_day_counter_name() -> None:
    idx = _ConcreteIRI(
        "GBPLibor", Period(6, TimeUnit.Months), 0,
        EURCurrency(), TARGET(), Actual365Fixed(),
    )
    assert idx.name() == "GBPLibor6M Actual/365 (Fixed)"
