"""CapletVarianceCurve — optionlet vol curve backed by a BlackVarianceCurve.

# C++ parity: ql/termstructures/volatility/optionlet/capletvariancecurve.hpp
# (v1.42.1).

The class wraps a ``BlackVarianceCurve`` (from L2-E) and exposes it
through the optionlet API. The strike axis is ignored (the curve is
ATM-only). Black variance is read from the inner curve; vol is
``sqrt(variance/t)``.

The C++ class composes ``BlackVarianceCurve`` and inherits the
optionlet structure. PQuantLib uses composition + delegation —
Python's MRO doesn't need a multi-base solution here.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class CapletVarianceCurve(OptionletVolatilityStructure):
    """Optionlet variance curve over option dates."""

    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        caplet_vol_curve: Sequence[float],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        self._black_curve: BlackVarianceCurve = BlackVarianceCurve(
            reference_date=reference_date,
            dates=dates,
            black_vol_curve=caplet_vol_curve,
            day_counter=day_counter,
            force_monotone_variance=False,
            calendar=calendar,
        )
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement

    def max_date(self) -> Date:
        return self._black_curve.max_date()

    def min_strike(self) -> float:
        return self._black_curve.min_strike()

    def max_strike(self) -> float:
        return self._black_curve.max_strike()

    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    def displacement(self) -> float:
        return self._displacement

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: CapletVarianceCurve::volatilityImpl — delegates
        # to ``blackCurve_.blackVol(t, strike, true)``.
        return self._black_curve.black_vol_at_time(t, strike, extrapolate=True)
