"""BlackVarianceSurface — Black vol surface modelled as a variance surface.

# C++ parity: ql/termstructures/volatility/equityfx/blackvariancesurface.hpp +
#             blackvariancesurface.cpp (v1.42.1).

Inputs: a reference date, calendar, list of dates (length ``M``), list
of strikes (length ``K``), a (``K`` x ``M``) ``Matrix`` of Black
volatilities, and a day counter. The (date, vol) pillars are converted
to (t, variance) where ``variance[i, j+1] = t[j+1] * vol[i, j]^2``.
Bilinear interpolation is the default.

Note the indexing convention: in C++ the matrix is ``[strike_idx,
date_idx]`` (rows = strikes, columns = dates), and the
``Interpolation2D::z[j, i]`` lookup expects ``[y_idx=strike, x_idx=date]``.
PQuantLib's ``BilinearInterpolation`` indexes as ``z[y, x]``, so we
pass the variance matrix with strikes as rows and times as columns —
exactly the C++ layout.

Strike-extrapolation modes (``ConstantExtrapolation`` /
``InterpolatorDefaultExtrapolation``) are ported. Time-extrapolation
beyond the last pillar uses the C++ default (``flat-variance-per-time =
variance(t_max, K) / t_max * t``).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVarianceTermStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class Extrapolation(IntEnum):
    """Strike-extrapolation mode below / above the strike grid.

    # C++ parity: BlackVarianceSurface::Extrapolation
    """

    ConstantExtrapolation = 0
    InterpolatorDefaultExtrapolation = 1


class BlackVarianceSurface(BlackVarianceTermStructure):
    """Black volatility surface modelled as a variance surface."""

    def __init__(
        self,
        *,
        reference_date: Date,
        calendar: Calendar,
        dates: Sequence[Date],
        strikes: Sequence[float],
        black_vol_matrix: Matrix,
        day_counter: DayCounter,
        lower_extrapolation: Extrapolation = Extrapolation.InterpolatorDefaultExtrapolation,
        upper_extrapolation: Extrapolation = Extrapolation.InterpolatorDefaultExtrapolation,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        n_dates = len(dates)
        n_strikes = len(strikes)
        mat = np.ascontiguousarray(black_vol_matrix, dtype=np.float64)
        qassert.require(
            mat.shape == (n_strikes, n_dates),
            f"vol matrix shape {mat.shape} must be (n_strikes={n_strikes}, n_dates={n_dates})",
        )
        qassert.require(
            dates[0] >= reference_date,
            "cannot have dates[0] < referenceDate",
        )

        # Pre-pend t=0 column with zero variance, mirroring C++.
        times: list[float] = [0.0] * (n_dates + 1)
        variances = np.zeros((n_strikes, n_dates + 1), dtype=np.float64)
        for j in range(1, n_dates + 1):
            t_j = day_counter.year_fraction(reference_date, dates[j - 1])
            qassert.require(t_j > times[j - 1], "dates must be sorted unique!")
            times[j] = t_j
            for i in range(n_strikes):
                v = float(mat[i, j - 1])
                variances[i, j] = t_j * v * v

        self._strikes: Array = np.asarray(strikes, dtype=np.float64)
        self._times: Array = np.asarray(times, dtype=np.float64)
        self._variances: Matrix = variances
        self._max_date: Date = dates[-1]
        self._lower_extrapolation: Extrapolation = lower_extrapolation
        self._upper_extrapolation: Extrapolation = upper_extrapolation

        # default: bilinear interpolation. BilinearInterpolation indexes
        # z as ``[y, x]``; with strikes on y-axis and times on x-axis,
        # variances[strike_idx, time_idx] matches the C++ ``zData_[j][i]``
        # convention exactly.
        self._variance_surface: BilinearInterpolation = BilinearInterpolation(
            self._times, self._strikes, variances
        )
        # The surface always queries with allow_extrapolation=True;
        # see _black_variance_impl below. Setting the flag here keeps
        # the public interface aligned with C++ blackvariancesurface.cpp.
        self._variance_surface.enable_extrapolation(True)

    def max_date(self) -> Date:
        return self._max_date

    def min_strike(self) -> float:
        return float(self._strikes[0])

    def max_strike(self) -> float:
        return float(self._strikes[-1])

    def _black_variance_impl(self, t: float, strike: float) -> float:
        if t == 0.0:
            return 0.0

        # enforce constant strike extrapolation when configured
        k = strike
        if k < float(self._strikes[0]) and self._lower_extrapolation == Extrapolation.ConstantExtrapolation:
            k = float(self._strikes[0])
        if k > float(self._strikes[-1]) and self._upper_extrapolation == Extrapolation.ConstantExtrapolation:
            k = float(self._strikes[-1])

        if t <= float(self._times[-1]):
            return self._variance_surface(t, k, allow_extrapolation=True)
        # t > times[-1]: time-flat extrapolation in variance (C++ default).
        return (
            self._variance_surface(float(self._times[-1]), k, allow_extrapolation=True)
            * t
            / float(self._times[-1])
        )
