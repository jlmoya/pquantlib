"""BlackVarianceCurve — Black vol curve modelled as a variance curve.

# C++ parity: ql/termstructures/volatility/equityfx/blackvariancecurve.hpp +
#             blackvariancecurve.cpp (v1.42.1).

Inputs: a reference date, a list of dates, a list of Black volatilities
at those dates, and a day counter. The class converts each (date, vol)
to (t, variance) where ``t = day_counter.year_fraction(ref, date)`` and
``variance = t * vol^2``. A linear interpolation on (t, variance) is
the default (settable via ``set_interpolation`` for custom interpolators).

PQuantLib defers two C++ features:

- ``BlackVolTimeExtrapolation`` (FlatVolatility / UseInterpolator /
  LinearVariance). The C++ default is ``FlatVolatility``; PQuantLib
  L2-E exposes only the default behavior — extrapolation beyond the
  last pillar uses ``flat-variance-per-time = variance(t_max) /
  t_max * t``.
- Custom interpolators via ``setInterpolation<Interpolator>``. The
  L2-E port pins Linear; cubic-spline variants will land alongside
  the same in L1 carve-outs (currently deferred).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class BlackVarianceCurve(BlackVarianceTermStructure):
    """Black volatility curve modelled as a variance curve (linear interp)."""

    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        black_vol_curve: Sequence[float],
        day_counter: DayCounter,
        force_monotone_variance: bool = True,
        calendar: Calendar | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        qassert.require(
            len(dates) == len(black_vol_curve),
            "mismatch between date vector and black vol vector",
        )
        qassert.require(len(dates) >= 1, "at least one date required")
        qassert.require(
            dates[0] > reference_date,
            "cannot have dates[0] <= referenceDate",
        )

        # Pre-pend (t=0, variance=0) as C++ does so the linear interp
        # has an anchor at the reference date.
        n = len(dates)
        times: list[float] = [0.0] * (n + 1)
        variances: list[float] = [0.0] * (n + 1)
        for j in range(1, n + 1):
            t_j = day_counter.year_fraction(reference_date, dates[j - 1])
            qassert.require(t_j > times[j - 1], "dates must be sorted unique!")
            times[j] = t_j
            v_j = float(black_vol_curve[j - 1])
            variances[j] = t_j * v_j * v_j
            qassert.require(
                variances[j] >= variances[j - 1] or not force_monotone_variance,
                "variance must be non-decreasing",
            )

        self._max_date: Date = dates[-1]
        self._times: list[float] = times
        self._variances: list[float] = variances
        # default: linear interpolation on (times, variances)
        self._variance_curve: LinearInterpolation = LinearInterpolation(
            np.asarray(times, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
        )

    def max_date(self) -> Date:
        return self._max_date

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def _black_variance_impl(self, t: float, strike: float) -> float:
        _ = strike  # the curve is strike-independent
        if t <= self._times[-1]:
            return max(self._variance_curve(t, allow_extrapolation=True), 0.0)
        # FlatVolatility extrapolation (C++ default):
        # ``max(varianceCurve(t_max), 0) / t_max * t``.
        v_max = max(self._variance_curve(self._times[-1], allow_extrapolation=True), 0.0)
        return v_max / self._times[-1] * t
