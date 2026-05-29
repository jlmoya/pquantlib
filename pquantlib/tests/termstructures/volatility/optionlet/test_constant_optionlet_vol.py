"""Tests for ConstantOptionletVolatility.

Cross-validated against L8-C C++ probe (cluster/l8c.json).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
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
_COV = _REF["constant_optionlet_vol"]


def _eval_date() -> Date:
    return Date(_REF["setup"]["eval_date_serial"])


def _new_cov(vol: float = 0.20) -> ConstantOptionletVolatility:
    return ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=vol,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )


def test_volatility_constant_at_date() -> None:
    cv = _new_cov()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, 0.04, True), _COV["v"])


def test_black_variance_matches_probe() -> None:
    cv = _new_cov()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.black_variance(d2y, 0.04, True), _COV["black_variance"])


def test_black_variance_time_scaling() -> None:
    cv = _new_cov()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    v = cv.volatility(d2y, 0.04, True)
    t = Actual365Fixed().year_fraction(_eval_date(), d2y)
    tolerance.tight(cv.black_variance(d2y, 0.04, True), v * v * t)


def test_volatility_strike_independent() -> None:
    cv = _new_cov()
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    v_low = cv.volatility(d2y, 0.001, True)
    v_high = cv.volatility(d2y, 100.0, True)
    tolerance.tight(v_low, v_high)


def test_min_max_strike_infinite() -> None:
    cv = _new_cov()
    assert cv.min_strike() == -math.inf
    assert cv.max_strike() == math.inf


def test_max_date_returns_date_max() -> None:
    cv = _new_cov()
    assert cv.max_date() == Date.max_date()


def test_quote_constructor_round_trip_and_update() -> None:
    q = SimpleQuote(0.20)
    cv = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=q,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
    )
    d2y = TARGET().advance(_eval_date(), 2, TimeUnit.Years)
    tolerance.tight(cv.volatility(d2y, 0.04, True), 0.20)
    q.set_value(0.30)
    tolerance.tight(cv.volatility(d2y, 0.04, True), 0.30)


def test_volatility_type_default() -> None:
    cv = _new_cov()
    assert cv.volatility_type() == VolatilityType.ShiftedLognormal
    assert cv.displacement() == 0.0


def test_period_overload_matches_date_overload() -> None:
    cv = _new_cov()
    p = Period(2, TimeUnit.Years)
    d = cv.option_date_from_tenor(p)
    tolerance.tight(
        cv.volatility(p, 0.04, True), cv.volatility(d, 0.04, True)
    )


def test_normal_type_constructor() -> None:
    cv = ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.01,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=_eval_date(),
        volatility_type=VolatilityType.Normal,
        displacement=0.0,
    )
    assert cv.volatility_type() == VolatilityType.Normal


def test_past_date_without_extrapolation_raises() -> None:
    cv = _new_cov()
    past = Date.from_ymd(1, Month.January, 2000)
    with pytest.raises(LibraryException):
        cv.volatility(past, 0.04, False)
