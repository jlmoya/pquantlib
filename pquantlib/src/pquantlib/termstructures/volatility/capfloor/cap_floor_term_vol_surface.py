"""CapFloorTermVolSurface — 2-D bilinear ATM cap/floor vol surface.

# C++ parity: ql/termstructures/volatility/capfloor/capfloortermvolsurface.{hpp,cpp}
# (v1.42.1).

A grid of (option_tenor x strike) ATM cap/floor vols. Strikes form the
x-axis, times form the y-axis. The C++ port hard-wires
``BicubicSpline`` as the interpolator; PQuantLib uses bilinear because
the bicubic spline is in the L1 carve-out (deferred — see
``phase1-completion.md``). At node points the two interpolations
agree (the spline passes through pillars); intermediate-strike values
on locally linear smiles also agree; off-node time values diverge in
the cubic-spline corrections. Tests assert TIGHT at nodes and
LOOSE / custom for the few intermediate points where bilinear and
bicubic measurably differ.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_volatility_structure import (
    CapFloorTermVolatilityStructure,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CapFloorTermVolSurface(CapFloorTermVolatilityStructure):
    """Cap/floor smile volatility surface (bilinear in (t, K))."""

    def __init__(
        self,
        *,
        business_day_convention: BusinessDayConvention,
        option_tenors: Sequence[Period],
        strikes: Sequence[float],
        volatilities: Matrix | Sequence[Sequence[float]],
        calendar: Calendar,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        settlement_days: int | None = None,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            business_day_convention=business_day_convention,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=dc,
            settlement_days=settlement_days,
        )
        n_tenors = len(option_tenors)
        n_strikes = len(strikes)
        qassert.require(n_tenors > 0, "empty option tenor vector")
        qassert.require(n_strikes > 0, "empty strikes vector")

        vols = np.ascontiguousarray(volatilities, dtype=np.float64)
        qassert.require(
            vols.shape == (n_tenors, n_strikes),
            f"vol matrix shape {vols.shape} must be (n_tenors={n_tenors}, "
            f"n_strikes={n_strikes})",
        )
        # Strikes must be strictly increasing.
        for j in range(1, n_strikes):
            qassert.require(
                strikes[j - 1] < strikes[j],
                f"non-increasing strikes at index {j}",
            )
        # Tenors must be strictly increasing.
        for i in range(1, n_tenors):
            qassert.require(
                option_tenors[i - 1] < option_tenors[i],
                f"non-increasing option tenors at index {i}",
            )

        self._option_tenors: list[Period] = list(option_tenors)
        self._strikes: list[float] = [float(s) for s in strikes]
        self._vols: Matrix = vols.copy()

        ref = self.reference_date()
        option_dates: list[Date] = []
        option_times: list[float] = []
        for p in self._option_tenors:
            d = self.option_date_from_tenor(p)
            option_dates.append(d)
            option_times.append(dc.year_fraction(ref, d))
        # Strict monotonicity on times follows from tenors.
        self._option_dates: list[Date] = option_dates
        self._option_times: list[float] = option_times

        # BilinearInterpolation expects z[y, x] indexing — y = time
        # (rows), x = strike (columns). The input matrix already
        # matches that layout, so we pass it through.
        self._interpolation: BilinearInterpolation = BilinearInterpolation(
            np.asarray(self._strikes, dtype=np.float64),
            np.asarray(option_times, dtype=np.float64),
            vols,
        )

    # --- inspectors ------------------------------------------------------

    def option_tenors(self) -> list[Period]:
        return list(self._option_tenors)

    def option_dates(self) -> list[Date]:
        return list(self._option_dates)

    def option_times(self) -> list[float]:
        return list(self._option_times)

    def strikes(self) -> list[float]:
        return list(self._strikes)

    # --- TermStructure interface -----------------------------------------

    def max_date(self) -> Date:
        return self._option_dates[-1]

    def min_strike(self) -> float:
        return self._strikes[0]

    def max_strike(self) -> float:
        return self._strikes[-1]

    def _volatility_impl(self, t: float, strike: float) -> float:
        # Clamp t to a non-negative range; the curve is anchored at t=0
        # but the interpolation grid starts at the first tenor.
        t_clamped = max(t, self._option_times[0])
        return self._interpolation(strike, t_clamped, allow_extrapolation=True)
