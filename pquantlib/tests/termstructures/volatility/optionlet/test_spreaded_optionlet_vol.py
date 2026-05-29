"""Tests for SpreadedOptionletVolatility."""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.optionlet.constant_optionlet_vol import (
    ConstantOptionletVolatility,
)
from pquantlib.termstructures.volatility.optionlet.spreaded_optionlet_vol import (
    SpreadedOptionletVolatility,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit


def _base() -> ConstantOptionletVolatility:
    return ConstantOptionletVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.20,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=Date.from_ymd(15, Month.January, 2024),
    )


def test_spread_added_at_evaluation() -> None:
    base = _base()
    spread = SimpleQuote(0.02)
    sov = SpreadedOptionletVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    tolerance.tight(sov.volatility(d2y, 0.04, True), 0.22)


def test_spread_update_propagates() -> None:
    base = _base()
    spread = SimpleQuote(0.02)
    sov = SpreadedOptionletVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    tolerance.tight(sov.volatility(d2y, 0.04, True), 0.22)
    spread.set_value(-0.05)
    tolerance.tight(sov.volatility(d2y, 0.04, True), 0.15)


def test_forwarded_attributes() -> None:
    base = _base()
    spread = SimpleQuote(0.0)
    sov = SpreadedOptionletVolatility(base, spread)
    assert sov.min_strike() == base.min_strike()
    assert sov.max_strike() == base.max_strike()
    assert sov.volatility_type() == base.volatility_type()
    assert sov.displacement() == base.displacement()
    assert sov.max_date() == base.max_date()
    assert sov.reference_date() == base.reference_date()


def test_black_variance_with_spread() -> None:
    base = _base()
    spread = SimpleQuote(0.01)
    sov = SpreadedOptionletVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    # Var = (v + spread)^2 * t = 0.21^2 * t.
    v_eff = 0.21
    t = Actual365Fixed().year_fraction(Date.from_ymd(15, Month.January, 2024), d2y)
    tolerance.tight(sov.black_variance(d2y, 0.04, True), v_eff * v_eff * t)
