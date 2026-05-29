"""Tests for ConstantCapFloorTermVolatility.

Cross-validated against L8-C C++ probe (cluster/l8c.json).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.constant_capfloor_term_vol import (
    ConstantCapFloorTermVolatility,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_CCFV = _REF["constant_capfloor_term_vol"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_ccfv(vol: float = 0.18) -> ConstantCapFloorTermVolatility:
    return ConstantCapFloorTermVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=vol,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )


def test_volatility_at_known_date_node_matches_probe() -> None:
    cv = _new_ccfv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, 0.05, True), _CCFV["v_at_2y_5pct"])


def test_volatility_strike_independent() -> None:
    cv = _new_ccfv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, 0.03, True), _CCFV["v_at_2y_3pct"])


def test_volatility_at_period_matches_date_overload() -> None:
    cv = _new_ccfv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    period_vol = cv.volatility(Period(2, TimeUnit.Years), 0.04, True)
    date_vol = cv.volatility(d2y, 0.04, True)
    tolerance.tight(period_vol, date_vol)


def test_volatility_at_time_matches_constant() -> None:
    cv = _new_ccfv()
    # Time overload — pass float (year fraction).
    v = cv.volatility(2.0, 0.04, True)
    tolerance.tight(v, 0.18)


def test_max_date_matches_probe() -> None:
    cv = _new_ccfv()
    assert cv.max_date().serial_number() == _CCFV["max_date_serial"]


def test_min_max_strike_are_infinite() -> None:
    cv = _new_ccfv()
    assert cv.min_strike() == -math.inf
    assert cv.max_strike() == math.inf


def test_quote_constructor_round_trip() -> None:
    q = SimpleQuote(0.18)
    cv = ConstantCapFloorTermVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=q,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, 0.04, True), 0.18)
    # Updating the quote propagates through.
    q.set_value(0.22)
    tolerance.tight(cv.volatility(d2y, 0.04, True), 0.22)


def test_strike_range_check_raises_when_extrapolation_disabled() -> None:
    # Constant vol has [-inf, +inf] strike range, so range check never
    # fires. The range check applies to *time*, so we test that one.
    cv = _new_ccfv()
    # Past reference date — must raise.
    past = Date.from_ymd(1, Month.January, 2000)
    with pytest.raises(LibraryException):
        cv.volatility(past, 0.04, False)
