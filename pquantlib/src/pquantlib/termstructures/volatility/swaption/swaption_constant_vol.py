"""ConstantSwaptionVolatility — constant swaption vol.

# C++ parity: ql/termstructures/volatility/swaption/swaptionconstantvol.{hpp,cpp}
# (v1.42.1).
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class SwaptionConstantVolatility(SwaptionVolatilityStructure):
    """Constant swaption volatility (no expiry / tenor / strike dependence)."""

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
        shift: float = 0.0,
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
        self._shift: float = shift
        # # C++ parity: ``maxSwapTenor_ = 100 * Years`` — the C++ default.
        self._max_swap_tenor: Period = Period(100, TimeUnit.Years)

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def max_swap_tenor(self) -> Period:
        return self._max_swap_tenor

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        _ = option_time, swap_length, strike
        return self._volatility.value()

    def _shift_impl(self, option_time: float, swap_length: float) -> float:
        _ = option_time, swap_length
        return self._shift
