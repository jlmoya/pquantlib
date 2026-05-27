"""LocalVolCurve — local volatility derived from a BlackVarianceCurve.

# C++ parity: ql/termstructures/volatility/equityfx/localvolcurve.hpp (v1.42.1).

Wraps a BlackVarianceCurve and computes local vol via the Dupire
relation for the strike-independent case::

    sigma_L(t)^2 = d/dt [ sigma_B^2(t) * t ]

i.e. the local volatility is the derivative of total variance w.r.t.
time. C++ uses a one-sided forward finite-difference with
``dt = 1 / 365`` to estimate the derivative; we do the same.

C++ uses ``Handle<BlackVarianceCurve>``; Python passes the curve
directly and registers with it as an observer.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

_DT: float = 1.0 / 365.0


class LocalVolCurve(LocalVolTermStructure):
    """Local volatility curve derived from a BlackVarianceCurve (Dupire 1-D)."""

    def __init__(self, curve: BlackVarianceCurve) -> None:
        super().__init__(
            business_day_convention=curve.business_day_convention(),
            day_counter=curve.day_counter(),
        )
        self._black_variance_curve: BlackVarianceCurve = curve
        self._black_variance_curve.register_with(self)

    # --- delegated TermStructure accessors ---------------------------------

    def reference_date(self) -> Date:
        return self._black_variance_curve.reference_date()

    def calendar(self) -> Calendar:
        # Delegate; BlackVarianceCurve may not have one explicitly — let it raise.
        return self._black_variance_curve.calendar()

    def day_counter(self) -> DayCounter:
        return self._black_variance_curve.day_counter()

    def max_date(self) -> Date:
        return self._black_variance_curve.max_date()

    # --- VolatilityTermStructure accessors ---------------------------------

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    # --- local-vol impl ----------------------------------------------------

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        # Strike-independent Dupire: sigma_L(t) = sqrt(d/dt [ sigma_B^2 * t ]).
        var1 = self._black_variance_curve.black_variance_at_time(
            t, underlying_level, extrapolate=True
        )
        var2 = self._black_variance_curve.black_variance_at_time(
            t + _DT, underlying_level, extrapolate=True
        )
        derivative = (var2 - var1) / _DT
        return math.sqrt(derivative)
