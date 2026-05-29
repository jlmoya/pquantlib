"""InterpolatedDefaultDensityCurve — default density at known dates.

# C++ parity: ql/termstructures/credit/interpolateddefaultdensitycurve.hpp
   (v1.42.1).

Default interpolator is ``Linear``. Validation: ``density[i] >= 0``.

S(t) = max(1 - integral_0^t p(tau) d tau, 0).

Flat extrapolation past the last knot: ``p(t) = p_last``;
``integral(t) = primitive(t_last) + p_last * (t - t_last)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.default_density_structure import (
    DefaultDensityStructure,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedDefaultDensityCurve(DefaultDensityStructure):
    """Default-probability curve from interpolated default densities."""

    def __init__(
        self,
        dates: Sequence[Date],
        densities: Sequence[float],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
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
        self._data: list[float] = list(densities)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
        for i in range(len(self._dates)):
            qassert.require(self._data[i] >= 0.0, "negative default density")
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

    # ---- DefaultDensityStructure implementation ---------------------------

    def _default_density_impl(self, t: float) -> float:
        # C++ parity: interpolateddefaultdensitycurve.hpp:147-153.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        return self._data[-1]

    def _survival_probability_impl(self, t: float) -> float:
        # C++ parity: interpolateddefaultdensitycurve.hpp:155-172.
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
        return max(1.0 - integral, 0.0)

    # ---- inspectors --------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def default_densities(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))


__all__ = ["InterpolatedDefaultDensityCurve"]
