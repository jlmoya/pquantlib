"""InterpolatedSurvivalProbabilityCurve — survival probabilities at known dates.

# C++ parity: ql/termstructures/credit/interpolatedsurvivalprobabilitycurve.hpp
   (v1.42.1).

C++ templates ``InterpolatedSurvivalProbabilityCurve<Interpolator>``; Python
takes an ``InterpolationFactory`` callable. Default is
``LogLinearInterpolation`` — i.e. survival probabilities are interpolated
linearly in log-space, equivalent to piecewise-constant *hazard rate*
between nodes (the C++ default for credit curves).

Validation at construction:
- ``probabilities[0] == 1.0`` (first date == reference date).
- ``probabilities[i] > 0.0``.
- ``probabilities[i] <= probabilities[i-1]`` (survival is monotonic).

Flat-hazard-rate extrapolation past the last knot:
``S(t) = sMax * exp(-hazardMax * (t - tMax))`` where
``hazardMax = -interpolation.derivative(tMax) / sMax``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.survival_probability_structure import (
    SurvivalProbabilityStructure,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedSurvivalProbabilityCurve(SurvivalProbabilityStructure):
    """Default-probability curve from interpolated survival probabilities."""

    def __init__(
        self,
        dates: Sequence[Date],
        probabilities: Sequence[float],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
        interpolator: InterpolationFactory = LogLinearInterpolation,
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
        self._data: list[float] = list(probabilities)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
        qassert.require(
            self._data[0] == 1.0,
            "the first probability must be == 1.0 to flag the corresponding "
            "date as reference date",
        )
        for i in range(1, len(self._dates)):
            qassert.require(self._data[i] > 0.0, "negative probability")
            qassert.require(
                self._data[i] <= self._data[i - 1],
                f"negative hazard rate implied by the survival probability "
                f"{self._data[i]} at {self._dates[i]} (t={self._times[i] if i < len(self._times) else 'NA'}) "
                f"after the survival probability {self._data[i - 1]} at "
                f"{self._dates[i - 1]}",
            )
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

    # ---- DefaultProbabilityTermStructure implementation -------------------

    def _survival_probability_impl(self, t: float) -> float:
        # C++ parity: interpolatedsurvivalprobabilitycurve.hpp:146-156.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        s_max = self._data[-1]
        hazard_max = -self._interpolation.derivative(max_time, allow_extrapolation=True) / s_max
        return s_max * math.exp(-hazard_max * (t - max_time))

    def _default_density_impl(self, t: float) -> float:
        # C++ parity: interpolatedsurvivalprobabilitycurve.hpp:158-169.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return -self._interpolation.derivative(t, allow_extrapolation=True)
        s_max = self._data[-1]
        hazard_max = -self._interpolation.derivative(max_time, allow_extrapolation=True) / s_max
        return s_max * hazard_max * math.exp(-hazard_max * (t - max_time))

    # ---- inspectors --------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def survival_probabilities(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))


__all__ = ["InterpolatedSurvivalProbabilityCurve"]
