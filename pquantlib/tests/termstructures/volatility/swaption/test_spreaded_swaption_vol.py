"""Tests for SpreadedSwaptionVolatility."""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.spreaded_swaption_vol import (
    SpreadedSwaptionVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    SwaptionConstantVolatility,
)
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _base() -> SwaptionConstantVolatility:
    return SwaptionConstantVolatility(
        business_day_convention=BusinessDayConvention.ModifiedFollowing,
        volatility=0.20,
        calendar=TARGET(),
        day_counter=Actual365Fixed(),
        reference_date=Date.from_ymd(15, Month.January, 2024),
    )


def test_spread_added_at_evaluation() -> None:
    base = _base()
    spread = SimpleQuote(0.02)
    sv = SpreadedSwaptionVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    tolerance.tight(
        sv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), 0.22
    )


def test_spread_update_propagates() -> None:
    base = _base()
    spread = SimpleQuote(0.02)
    sv = SpreadedSwaptionVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    tolerance.tight(
        sv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), 0.22
    )
    spread.set_value(-0.05)
    tolerance.tight(
        sv.volatility(d2y, Period(5, TimeUnit.Years), 0.04, True), 0.15
    )


def test_forwarded_attributes() -> None:
    base = _base()
    spread = SimpleQuote(0.0)
    sv = SpreadedSwaptionVolatility(base, spread)
    assert sv.min_strike() == base.min_strike()
    assert sv.max_strike() == base.max_strike()
    assert sv.max_swap_tenor() == base.max_swap_tenor()
    assert sv.max_date() == base.max_date()
    assert sv.reference_date() == base.reference_date()
    assert sv.volatility_type() == base.volatility_type()


def test_black_variance_with_spread() -> None:
    base = _base()
    spread = SimpleQuote(0.01)
    sv = SpreadedSwaptionVolatility(base, spread)
    d2y = TARGET().advance(Date.from_ymd(15, Month.January, 2024), 2, TimeUnit.Years)
    v_eff = 0.21
    t = Actual365Fixed().year_fraction(Date.from_ymd(15, Month.January, 2024), d2y)
    tolerance.tight(
        sv.black_variance(d2y, Period(5, TimeUnit.Years), 0.04, True),
        v_eff * v_eff * t,
    )
