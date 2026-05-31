"""ExtendedBlackVarianceSurface — Quote-backed Black variance surface.

# C++ parity: ql/experimental/volatility/extendedblackvariancesurface.{hpp,cpp}
# (v1.42.1).

Like :class:`~pquantlib.termstructures.volatility.equity_fx.black_variance_surface.BlackVarianceSurface`,
but the input volatilities are *live quotes* (:class:`Quote`) and the
2-D interpolator over (time, strike) -> variance is selectable
(default Bilinear).

The (date, strike) pillars are converted to (t, variance) where
``variance[i][j] = t[j] * vol[i][j-1]^2`` with a ``t=0`` anchor column.
``update()`` re-reads every quote and refits, so a downstream quote
change refreshes the surface.

Divergences from C++:

* **C++ v1.42.1 ``ExtendedBlackVarianceSurface`` is broken.** Its
  ``setVariances()`` loops ``times_.size()+1`` columns into a matrix with
  only ``dates+1`` columns and indexes the flat quote vector as
  ``i*times_.size()+j-1`` instead of ``i*nDates+(j-1)``, so the C++ class
  reads out of bounds and aborts on construction. It also stores the
  quote vector as ``const std::vector<Handle<Quote>>&`` (a reference to a
  ``std::move``'d temporary — a dangling-reference bug). PQuantLib ports
  the *documented correct* behaviour: ``variance[strike_i][date_j] =
  t[j] * vol[strike_i][date_j-1]^2`` with a Bilinear interpolation, the
  same z-grid convention as the working core ``BlackVarianceSurface``.
* **No ``Handle`` wrapper.** Quotes are held directly; the surface
  registers as their observer.
* **Selectable interpolator via ``interpolator=`` factory** (precedent:
  L9-C ``CapFloorTermVolSurface``). Default Bilinear (C++ default).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

# Structural type for a 2-D interpolator factory (see L9-C
# CapFloorTermVolSurface): ``(xs, ys, z) -> callable(x, y[, *, kw])``.
_Interp2DFactory = Callable[[Array, Array, Matrix], Callable[..., float]]


class Extrapolation(IntEnum):
    """Strike-extrapolation mode below / above the strike grid.

    # C++ parity: ExtendedBlackVarianceSurface::Extrapolation.
    """

    ConstantExtrapolation = 0
    InterpolatorDefaultExtrapolation = 1


class ExtendedBlackVarianceSurface(BlackVarianceTermStructure):
    """Black variance surface modelled from Quote-backed vols (selectable interp)."""

    def __init__(
        self,
        *,
        reference_date: Date,
        calendar: Calendar,
        dates: Sequence[Date],
        strikes: Sequence[float],
        vol_matrix: Matrix | Sequence[Sequence[Quote]],
        day_counter: DayCounter,
        lower_extrapolation: Extrapolation = Extrapolation.InterpolatorDefaultExtrapolation,
        upper_extrapolation: Extrapolation = Extrapolation.InterpolatorDefaultExtrapolation,
        interpolator: _Interp2DFactory = BilinearInterpolation,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        n_dates = len(dates)
        n_strikes = len(strikes)
        # vol_matrix is a (n_strikes x n_dates) grid of Quotes (rows =
        # strikes, columns = dates) — the same layout as the core
        # BlackVarianceSurface vol matrix.
        quote_grid: list[list[Quote]] = [list(row) for row in vol_matrix]
        qassert.require(
            len(quote_grid) == n_strikes
            and all(len(row) == n_dates for row in quote_grid),
            f"vol matrix shape must be (n_strikes={n_strikes}, n_dates={n_dates})",
        )
        qassert.require(
            dates[0] > reference_date,
            "cannot have dates_[0] <= referenceDate_",
        )

        times: list[float] = [0.0] * (n_dates + 1)
        for j in range(1, n_dates + 1):
            t_j = day_counter.year_fraction(reference_date, dates[j - 1])
            qassert.require(t_j > times[j - 1], "dates must be sorted unique")
            times[j] = t_j

        self._strikes: Array = np.asarray(strikes, dtype=np.float64)
        self._times: Array = np.asarray(times, dtype=np.float64)
        self._max_date: Date = dates[-1]
        self._lower_extrapolation: Extrapolation = lower_extrapolation
        self._upper_extrapolation: Extrapolation = upper_extrapolation
        self._quote_grid: list[list[Quote]] = quote_grid
        self._n_dates: int = n_dates
        self._n_strikes: int = n_strikes
        self._interpolator: _Interp2DFactory = interpolator
        self._variances: Matrix = np.zeros((n_strikes, n_dates + 1), dtype=np.float64)

        self._set_variances()
        self._set_interpolation()

        # Register with every quote so the surface refreshes on changes.
        for row in self._quote_grid:
            for q in row:
                q.register_with(self)

    # --- internal ----------------------------------------------------------

    def _set_variances(self) -> None:
        """Recompute the (strike x time) variance grid from quote values.

        # C++ parity (corrected): ``variance[i][j] = t[j]*vol[i][j-1]^2``,
        # anchor column 0 = 0. See the module docstring on the C++ bug.
        """
        for i in range(self._n_strikes):
            self._variances[i, 0] = 0.0
            for j in range(1, self._n_dates + 1):
                sigma = self._quote_grid[i][j - 1].value()
                self._variances[i, j] = float(self._times[j]) * sigma * sigma
                qassert.require(
                    self._variances[i, j] >= self._variances[i, j - 1],
                    "variance must be non-decreasing",
                )

    def _set_interpolation(self) -> None:
        # Bilinear indexes z as ``[y, x]``; strikes on y-axis, times on
        # x-axis matches the variances[strike_idx, time_idx] convention.
        surface = self._interpolator(self._times, self._strikes, self._variances)
        enable = getattr(surface, "enable_extrapolation", None)
        if callable(enable):
            enable(True)
        self._variance_surface: Callable[..., float] = surface

    def update(self) -> None:
        """Observer.update — re-read quotes, refit, propagate.

        # C++ parity: ExtendedBlackVarianceSurface::update.
        """
        self._set_variances()
        self._set_interpolation()
        self.notify_observers()

    # --- TermStructure / VolatilityTermStructure ---------------------------

    def max_date(self) -> Date:
        return self._max_date

    def min_strike(self) -> float:
        return float(self._strikes[0])

    def max_strike(self) -> float:
        return float(self._strikes[-1])

    # --- BlackVarianceTermStructure ----------------------------------------

    def _black_variance_impl(self, t: float, strike: float) -> float:
        if t == 0.0:
            return 0.0

        # enforce constant strike extrapolation when configured.
        k = strike
        if (
            k < float(self._strikes[0])
            and self._lower_extrapolation == Extrapolation.ConstantExtrapolation
        ):
            k = float(self._strikes[0])
        if (
            k > float(self._strikes[-1])
            and self._upper_extrapolation == Extrapolation.ConstantExtrapolation
        ):
            k = float(self._strikes[-1])

        t_max = float(self._times[-1])
        if t <= t_max:
            return self._variance_surface(t, k, allow_extrapolation=True)
        # t > times[-1]: flat-variance-per-time extrapolation.
        return self._variance_surface(t_max, k, allow_extrapolation=True) * t / t_max


__all__ = ["ExtendedBlackVarianceSurface", "Extrapolation"]
