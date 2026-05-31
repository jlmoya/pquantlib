"""SABRVolTermStructure — implied-vol surface backed by a SABR model.

# C++ parity: ql/experimental/volatility/sabrvoltermstructure.hpp (v1.42.1).

A :class:`BlackVolatilityTermStructure` whose Black vol at ``(t, strike)``
is the closed-form Hagan-2002 SABR volatility evaluated at the
log-normal forward ``F(t) = s0 * exp(r t)``:

.. math::

   \\sigma_{Black}(t, K) = \\mathrm{sabrVolatility}(K, F(t), t;\\,
                                                    \\alpha, \\beta, \\gamma, \\rho)

where the C++ parameter ``gamma`` is the SABR vol-of-vol ``nu``.

This is a single-parameter-set term structure (no per-expiry
calibration) — useful for analytic SABR pricing test fixtures.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.sabr_formula import sabr_volatility
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolatilityTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date


class SABRVolTermStructure(BlackVolatilityTermStructure):
    """Black volatility term structure backed by a single SABR parameter set."""

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        gamma: float,
        rho: float,
        s0: float,
        r: float,
        reference_date: Date,
        day_counter: DayCounter,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=NullCalendar(),
            day_counter=day_counter,
        )
        self._alpha: float = alpha
        self._beta: float = beta
        self._gamma: float = gamma
        self._rho: float = rho
        self._s0: float = s0
        self._r: float = r

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return 0.0

    def max_strike(self) -> float:
        return math.inf

    def _black_vol_impl(self, t: float, strike: float) -> float:
        fwd = self._s0 * math.exp(self._r * t)
        # C++ ``gamma`` is the SABR vol-of-vol ``nu``.
        return sabr_volatility(
            strike, fwd, t, self._alpha, self._beta, self._gamma, self._rho
        )


__all__ = ["SABRVolTermStructure"]
