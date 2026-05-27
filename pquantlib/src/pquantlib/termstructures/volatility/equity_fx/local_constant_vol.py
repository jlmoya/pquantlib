"""LocalConstantVol — constant local volatility (no time or asset dependence).

# C++ parity: ql/termstructures/volatility/equityfx/localconstantvol.hpp (v1.42.1).

Local vol and Black vol coincide when sigma is at most a function of
time (the integrated-variance equality reduces to a pointwise equality
under no S-dependence). This class is therefore essentially a proxy for
``BlackConstantVol`` from the local-vol side.

C++ supports both a raw ``Volatility`` constructor and a ``Handle<Quote>``
constructor; PQuantLib unifies these into a single keyword that accepts
either ``float`` or ``Quote``.

Construction modes 1 (fixed reference date) only. Mode 3
(``Natural settlementDays``) is deferred as in BlackConstantVol.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date


class LocalConstantVol(LocalVolTermStructure):
    """Constant local-vol term structure (no time / asset dependence)."""

    def __init__(
        self,
        *,
        reference_date: Date,
        volatility: float | Quote,
        day_counter: DayCounter,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            day_counter=day_counter,
        )
        self._volatility: Quote = (
            volatility if isinstance(volatility, Quote) else SimpleQuote(float(volatility))
        )
        self._volatility.register_with(self)

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        _ = t, underlying_level
        return self._volatility.value()
