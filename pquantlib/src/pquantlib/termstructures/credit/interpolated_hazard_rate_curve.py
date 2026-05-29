"""InterpolatedHazardRateCurve — hazard rates at known dates.

# C++ parity: ql/termstructures/credit/interpolatedhazardratecurve.hpp
   (v1.42.1).

C++ default interpolator is ``BackwardFlat`` (piecewise-constant hazard
rate between nodes, taking the right-hand value). Python takes an
``InterpolationFactory`` callable defaulting to
``BackwardFlatInterpolation``.

Validation at construction:
- Each ``hazard_rate >= 0``.

Flat-hazard-rate extrapolation past the last knot:
``h(t) = h_last`` for ``t > t_last``;
``S(t) = exp(- (primitive(t_last) + h_last * (t - t_last)))``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.hazard_rate_structure import HazardRateStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedHazardRateCurve(HazardRateStructure):
    """Default-probability curve from interpolated hazard rates."""

    def __init__(
        self,
        dates: Sequence[Date],
        hazard_rates: Sequence[float],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
        interpolator: InterpolationFactory = BackwardFlatInterpolation,
    ) -> None:
        qassert.require(len(dates) >= 1, "no input dates given")
        super().__init__(
            reference_date=dates[0],
            calendar=calendar,
            day_counter=day_counter,
            jumps=jumps,
            jump_dates=jump_dates,
        )
        self._dates: list[Date] = list(dates)
        self._data: list[float] = list(hazard_rates)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
        for i in range(len(self._dates)):
            qassert.require(self._data[i] >= 0.0, "negative hazard rate")
        ref = self._dates[0]
        dc = self.day_counter()
        self._times = [dc.year_fraction(ref, d) for d in self._dates]
        self._interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- TermStructure interface ------------------------------------------

    def max_date(self) -> Date:
        return self._dates[-1]

    # ---- HazardRateStructure implementation -------------------------------

    def _hazard_rate_impl(self, t: float) -> float:
        # C++ parity: interpolatedhazardratecurve.hpp:149-155.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        return self._data[-1]

    def _survival_probability_impl(self, t: float) -> float:
        # C++ parity: interpolatedhazardratecurve.hpp:158-172.
        if t == 0.0:
            return 1.0
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            integral = self._interpolation.primitive(t, allow_extrapolation=True)
        else:
            integral = (
                self._interpolation.primitive(max_time, allow_extrapolation=True)
                + self._data[-1] * (t - max_time)
            )
        return math.exp(-integral)

    # ---- inspectors --------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def hazard_rates(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))


__all__ = ["InterpolatedHazardRateCurve"]
