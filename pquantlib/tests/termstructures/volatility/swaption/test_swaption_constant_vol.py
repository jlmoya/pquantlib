"""Tests for SwaptionConstantVolatility.

Cross-validated against L8-C C++ probe (cluster/l8c.json).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_REF = reference_reader.load("cluster/l8c")
_CSV = _REF["constant_swaption_vol"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_csv(vol: float = 0.20) -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=vol,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )


def test_volatility_constant_at_date_period_strike() -> None:
    cv = _new_csv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    v = cv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True)
    tolerance.tight(v, _CSV["v"])


def test_black_variance_matches_probe() -> None:
    cv = _new_csv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    bv = cv.black_variance(d2y, Period(5, TimeUnit.Years), 0.04, True)
    tolerance.tight(bv, _CSV["black_variance"])


def test_volatility_strike_independent() -> None:
    cv = _new_csv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    v_low = cv.volatility(d2y, Period(5, TimeUnit.Years), 0.001, True)
    v_high = cv.volatility(d2y, Period(5, TimeUnit.Years), 100.0, True)
    tolerance.tight(v_low, v_high)


def test_volatility_swap_tenor_independent() -> None:
    cv = _new_csv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    v1 = cv.volatility(d2y, Period(1, TimeUnit.Years), 0.04, True)
    v10 = cv.volatility(d2y, Period(10, TimeUnit.Years), 0.04, True)
    tolerance.tight(v1, v10)


def test_max_date_returns_date_max() -> None:
    cv = _new_csv()
    assert cv.max_date() == Date.max_date()


def test_max_swap_tenor_default_100y() -> None:
    cv = _new_csv()
    assert cv.max_swap_tenor() == Period(100, TimeUnit.Years)


def test_min_max_strike_infinite() -> None:
    cv = _new_csv()
    assert cv.min_strike() == -math.inf
    assert cv.max_strike() == math.inf


def test_quote_constructor_round_trip_and_update() -> None:
    q = SimpleQuote(0.20)
    cv = SwaptionConstantVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=q,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), 0.20)
    q.set_value(0.30)
    tolerance.tight(cv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), 0.30)


def test_shift_returns_constructor_shift() -> None:
    cv = SwaptionConstantVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.20,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
        shift=0.01,
    )
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.shift(d2y, Period(5, TimeUnit.Years), True), 0.01)


def test_volatility_type_default_and_normal() -> None:
    cv = _new_csv()
    assert cv.volatility_type() == VolatilityType.ShiftedLognormal
    cv2 = SwaptionConstantVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.01,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
        volatility_type=VolatilityType.Normal,
    )
    assert cv2.volatility_type() == VolatilityType.Normal


def test_past_date_without_extrapolation_raises() -> None:
    cv = _new_csv()
    past = Date.from_ymd(1, Month.January, 2000)
    with pytest.raises(LibraryException):
        cv.volatility(past, Period(5, TimeUnit.Years), 0.04, False)


def test_negative_swap_tenor_raises() -> None:
    cv = _new_csv()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    with pytest.raises(LibraryException):
        cv.volatility(d2y, Period(0, TimeUnit.Years), 0.04, True)
