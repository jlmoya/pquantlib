"""ConstantOptionletVolatility — constant caplet/floorlet vol.

# C++ parity: ql/termstructures/volatility/optionlet/constantoptionletvol.{hpp,cpp}
# (v1.42.1).

C++ exposes four constructors (floating/fixed reference x Quote/Vol);
PQuantLib mirrors them via ``Quote | float`` polymorphism on
``volatility`` plus the standard ``reference_date`` / ``settlement_days``
parent choice.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class ConstantOptionletVolatility(OptionletVolatilityStructure):
    """Constant optionlet volatility (no time / strike dependence)."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        volatility: float | Quote,
        calendar: Calendar,
        day_counter: DayCounter,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
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
        self._volatility.register_with(self)
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def displacement(self) -> float:
        return self._displacement

    def _volatility_impl(self, t: float, strike: float) -> float:
        _ = t, strike
        return self._volatility.value()
