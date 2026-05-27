"""BlackConstantVol — constant Black volatility (no time or strike dependence).

# C++ parity: ql/termstructures/volatility/equityfx/blackconstantvol.hpp (v1.42.1).

Two constructors:

- raw ``Volatility`` (float) — internally wrapped in a ``SimpleQuote``.
- ``Quote`` reference — registered as an observer so vol changes
  propagate through the term-structure graph.

``max_date`` is ``Date.max_date()`` (the curve never expires).
``min_strike`` / ``max_strike`` are ``-inf`` / ``+inf`` (the curve is
strike-independent).

C++ also has constructors that take ``settlementDays`` (mode 3); those
are deferred — the L2-E pilot ports only the fixed-reference-date mode.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolatilityTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class BlackConstantVol(BlackVolatilityTermStructure):
    """Constant Black volatility term structure (no time / strike dependence)."""

    def __init__(
        self,
        *,
        reference_date: Date,
        calendar: Calendar,
        day_counter: DayCounter,
        volatility: float | Quote,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        self._volatility: Quote = (
            volatility if isinstance(volatility, Quote) else SimpleQuote(float(volatility))
        )
        # Register so vol-quote changes notify our own observers.
        self._volatility.register_with(self)

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _black_vol_impl(self, t: float, strike: float) -> float:
        _ = t, strike  # constant in both
        return self._volatility.value()
