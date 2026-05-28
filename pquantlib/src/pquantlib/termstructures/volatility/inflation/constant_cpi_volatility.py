"""ConstantCPIVolatility — constant zero-CPI vol surface (no T or K dependence).

# C++ parity: ql/termstructures/volatility/inflation/constantcpivolatility.{hpp,cpp}
   (v1.42.1).

Two C++ constructors (raw ``Volatility`` and ``Handle<Quote>``); PQuantLib
collapses both into a single one accepting ``float | Quote`` and wraps
floats in a SimpleQuote internally so observer registration works.

Returned ``min_strike`` / ``max_strike`` are ``-inf`` / ``+inf`` — the
constant surface is K-independent. ``max_date`` is ``Date.max_date()``.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.inflation.cpi_volatility_surface import (
    CPIVolatilitySurface,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class ConstantCPIVolatility(CPIVolatilitySurface):
    """Constant zero-CPI volatility surface — no time / strike dependence."""

    def __init__(
        self,
        *,
        vol: float | Quote,
        settlement_days: int,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        day_counter: DayCounter,
        observation_lag: Period,
        frequency: Frequency,
        index_is_interpolated: bool,
    ) -> None:
        super().__init__(
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=business_day_convention,
            day_counter=day_counter,
            observation_lag=observation_lag,
            frequency=frequency,
            index_is_interpolated=index_is_interpolated,
        )
        # # C++ parity: stores ``Handle<Quote>``. Python wraps raw floats
        # # in a SimpleQuote so observer wiring is consistent.
        self._vol: Quote = vol if isinstance(vol, Quote) else SimpleQuote(float(vol))

    # ---- vol surface API ----------------------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: ConstantCPIVolatility::volatilityImpl returns vol_->value().
        del t, strike
        return self._vol.value()

    # ---- limits -------------------------------------------------------

    def max_date(self) -> Date:
        """The curve never expires."""
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf


__all__ = ["ConstantCPIVolatility"]
