"""ConstantYoYOptionletVolatility — constant YoY-rate vol surface.

# C++ parity: ql/termstructures/volatility/inflation/yoyinflationoptionletvolatilitystructure.{hpp,cpp}
   class ``ConstantYoYOptionletVolatility`` (v1.42.1).

Two C++ constructors (raw ``Volatility`` and ``Handle<Quote>``); PQuantLib
collapses both into a single one accepting ``float | Quote``.

The constant surface returns the configured vol regardless of ``(time,
strike)`` — used as a smile-less default and by the YoY-cap engines for
flat-vol pricing. ``min_strike``/``max_strike`` are passed at construction
(C++ defaults: -1.0 / 100.0). ``max_date`` is ``Date.max_date()``.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class ConstantYoYOptionletVolatility(YoYOptionletVolatilitySurface):
    """Constant YoY-rate volatility surface — no time / strike dependence."""

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
        min_strike: float = -1.0,
        max_strike: float = 100.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(
            settlement_days=settlement_days,
            calendar=calendar,
            business_day_convention=business_day_convention,
            day_counter=day_counter,
            observation_lag=observation_lag,
            frequency=frequency,
            index_is_interpolated=index_is_interpolated,
            volatility_type=volatility_type,
            displacement=displacement,
        )
        self._vol: Quote = vol if isinstance(vol, Quote) else SimpleQuote(float(vol))
        self._min_strike: float = min_strike
        self._max_strike: float = max_strike

    # ---- vol surface API ----------------------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: ConstantYoYOptionletVolatility::volatilityImpl returns
        # # vol_->value().
        del t, strike
        return self._vol.value()

    # ---- limits -------------------------------------------------------

    def max_date(self) -> Date:
        """The curve never expires."""
        return Date.max_date()

    def min_strike(self) -> float:
        return self._min_strike

    def max_strike(self) -> float:
        return self._max_strike


__all__ = ["ConstantYoYOptionletVolatility"]
