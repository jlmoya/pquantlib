"""ConstantCapFloorTermVolatility — constant cap/floor term vol.

# C++ parity: ql/termstructures/volatility/capfloor/constantcapfloortermvol.{hpp,cpp}
# (v1.42.1).

C++ exposes four constructors (floating/fixed reference x Quote/Vol);
PQuantLib mirrors them via the standard ``Quote | float`` polymorphism
on the ``volatility`` argument plus the optional
``settlement_days`` / ``reference_date`` choice on the parent.

``max_date`` is ``Date.max_date()``; the curve never expires.
``min_strike`` / ``max_strike`` are ``-inf`` / ``+inf`` — the term
vol is strike-independent at this layer.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_volatility_structure import (
    CapFloorTermVolatilityStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class ConstantCapFloorTermVolatility(CapFloorTermVolatilityStructure):
    """Constant cap/floor term volatility (no time / strike dependence)."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        volatility: float | Quote,
        calendar: Calendar,
        day_counter: DayCounter,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._volatility: Quote = (
            volatility if isinstance(volatility, Quote) else SimpleQuote(float(volatility))
        )
        # Vol-quote changes notify our own observers.
        self._volatility.register_with(self)

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _volatility_impl(self, t: float, strike: float) -> float:
        _ = t, strike  # constant in both
        return self._volatility.value()
