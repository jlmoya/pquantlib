"""InterpolatedDiscountCurve — yield curve from discount factors directly.

# C++ parity: ql/termstructures/yield/discountcurve.hpp (v1.42.1)

C++ templates ``InterpolatedDiscountCurve<Interpolator>``; Python takes
an ``InterpolationFactory``. Default is ``LogLinearInterpolation`` (the
C++ ``DiscountCurve`` typedef).

Subclasses ``YieldTermStructure`` directly (not ``ZeroYieldStructure``)
because the discount factor is the *primary* curve quantity here.
``_discount_impl(t)`` returns the interpolated value (log-linear by
default → piecewise-constant instantaneous forward).

Validation at construction:
- ``data[0] == 1.0`` (first date == reference date).
- ``data[i] > 0.0`` for all subsequent points.

Flat-forward extrapolation past the last knot: the C++ code computes
``instFwdMax = -derivative(tMax) / dMax`` (i.e. the negative of the
log-derivative scaled by the curve value), then extrapolates
``d(t) = dMax * exp(-instFwdMax * (t - tMax))``.
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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedDiscountCurve(YieldTermStructure):
    """Yield curve from discount factors at known dates."""

    def __init__(
        self,
        dates: Sequence[Date],
        dfs: Sequence[float],
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
        self._data: list[float] = list(dfs)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
        qassert.require(
            self._data[0] == 1.0,
            "the first discount must be == 1.0 to flag the corresponding date as reference date",
        )
        for i in range(1, len(self._dates)):
            qassert.require(self._data[i] > 0.0, "negative discount")
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

    # ---- YieldTermStructure implementation --------------------------------

    def _discount_impl(self, t: float) -> float:
        # C++ parity: ``discountcurve.hpp`` lines 158-168.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        # Flat-forward extrapolation past last knot.
        d_max = self._data[-1]
        inst_fwd_max = -self._interpolation.derivative(max_time, allow_extrapolation=True) / d_max
        return d_max * math.exp(-inst_fwd_max * (t - max_time))

    # ---- inspectors --------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def discounts(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
