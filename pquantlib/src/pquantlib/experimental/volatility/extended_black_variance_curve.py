"""ExtendedBlackVarianceCurve — Quote-backed Black variance curve.

# C++ parity: ql/experimental/volatility/extendedblackvariancecurve.{hpp,cpp}
# (v1.42.1).

Like :class:`~pquantlib.termstructures.volatility.equity_fx.black_variance_curve.BlackVarianceCurve`,
but the input volatilities are *live quotes* (:class:`Quote`) rather
than raw floats, and the interpolator over (time, variance) is
selectable (C++ ``setInterpolation<Interpolator>()``; default Linear).

The (date, vol-quote) pillars are converted to (t, variance) where
``variance[j] = t[j] * vol[j-1]^2`` with a ``(t=0, variance=0)`` anchor
prepended. ``update()`` re-reads every quote and refits the interpolator,
so a downstream quote change refreshes the curve.

Divergences from C++:

* **No ``Handle`` wrapper.** We take :class:`Quote` objects directly and
  register as their observer (matching the C++ ``registerWith`` calls).
* **Selectable interpolator via ``interpolator=`` factory.** C++ exposes
  ``template <class Interpolator> setInterpolation(...)``; we accept an
  ``interpolator: type[Interpolation]`` factory (precedent: L9-C
  ``CapFloorTermVolCurve``). Default is :class:`LinearInterpolation`
  (C++ default).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date


class ExtendedBlackVarianceCurve(BlackVarianceTermStructure):
    """Black variance curve modelled from Quote-backed vols (selectable interp)."""

    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        volatilities: Sequence[Quote],
        day_counter: DayCounter,
        force_monotone_variance: bool = True,
        interpolator: type[Interpolation] = LinearInterpolation,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            day_counter=day_counter,
        )
        qassert.require(
            len(dates) == len(volatilities),
            "size mismatch between dates and volatilities",
        )
        qassert.require(len(dates) >= 1, "at least one date required")
        qassert.require(
            dates[0] > reference_date,
            "cannot have dates_[0] <= referenceDate",
        )

        n = len(dates)
        times: list[float] = [0.0] * (n + 1)
        for j in range(1, n + 1):
            t_j = day_counter.year_fraction(reference_date, dates[j - 1])
            qassert.require(t_j > times[j - 1], "dates must be sorted unique!")
            times[j] = t_j

        self._max_date: Date = dates[-1]
        self._times: list[float] = times
        self._volatilities: list[Quote] = list(volatilities)
        self._force_monotone_variance: bool = force_monotone_variance
        self._interpolator: type[Interpolation] = interpolator
        self._variances: list[float] = [0.0] * (n + 1)

        self._set_variances()
        self._set_interpolation()

        # Register with each quote so the curve refreshes on quote changes.
        for q in self._volatilities:
            q.register_with(self)

    # --- internal ----------------------------------------------------------

    def _set_variances(self) -> None:
        """Recompute (t, variance) from the current quote values.

        # C++ parity: ExtendedBlackVarianceCurve::setVariances.
        """
        self._variances[0] = 0.0
        for j in range(1, len(self._volatilities) + 1):
            sigma = self._volatilities[j - 1].value()
            self._variances[j] = self._times[j] * sigma * sigma
            qassert.require(
                self._variances[j] >= self._variances[j - 1]
                or not self._force_monotone_variance,
                "variance must be non-decreasing",
            )

    def _set_interpolation(self) -> None:
        self._variance_curve: Interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._variances, dtype=np.float64),
        )

    def update(self) -> None:
        """Observer.update — re-read quotes, refit, propagate.

        # C++ parity: ExtendedBlackVarianceCurve::update.
        """
        self._set_variances()
        self._set_interpolation()
        self.notify_observers()

    # --- TermStructure / VolatilityTermStructure ---------------------------

    def max_date(self) -> Date:
        return self._max_date

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    # --- BlackVarianceTermStructure ----------------------------------------

    def _black_variance_impl(self, t: float, strike: float) -> float:
        _ = strike  # strike-independent
        if t <= self._times[-1]:
            return self._variance_curve(t, allow_extrapolation=True)
        # flat-variance-per-time extrapolation beyond the last pillar.
        # C++ parity: ``varianceCurve_(times_.back(), true)*t/times_.back()``.
        return (
            self._variance_curve(self._times[-1], allow_extrapolation=True)
            * t
            / self._times[-1]
        )


__all__ = ["ExtendedBlackVarianceCurve"]
