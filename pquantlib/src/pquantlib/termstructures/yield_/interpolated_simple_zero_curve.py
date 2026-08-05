"""InterpolatedSimpleZeroCurve — curve interpolating SIMPLE (not compounded) zero rates.

# C++ parity: ql/termstructures/yield/interpolatedsimplezerocurve.hpp (v1.43)

Unlike :class:`InterpolatedZeroCurve`, whose data are continuously-compounded
zero rates, the data here are SIMPLE rates: the discount factor is
``1 / (1 + R t)``, not ``exp(-R t)``.

Past the last node the curve extrapolates flat-forward on the instantaneous
forward implied by the interpolation's slope at the last node::

    instFwdMax = zMax + tMax * interpolation.derivative(tMax)
    R(t)       = (zMax * tMax + instFwdMax * (t - tMax)) / t

(C++ notes in a comment that Bloomberg instead extrapolates the
non-annualised zero flat; QuantLib does not, and neither does this port.)
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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedSimpleZeroCurve(YieldTermStructure):
    """Yield curve interpolating simple (1 / (1 + R t)) zero rates."""

    def __init__(
        self,
        dates: Sequence[Date],
        yields: Sequence[float],
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
        self._data: list[float] = list(yields)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        # C++ parity: ``InterpolatedSimpleZeroCurve<T>::initialize``. C++ checks
        # against T::requiredPoints; the Python interpolation classes carry the
        # same minimum, so the check is delegated to them at construction.
        qassert.require(
            len(self._data) == len(self._dates), "dates/data count mismatch"
        )
        ref = self._dates[0]
        dc = self.day_counter()
        self._times = [dc.year_fraction(ref, d) for d in self._dates]
        self._interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- TermStructure interface -------------------------------------------

    def max_date(self) -> Date:
        return self._dates[-1]

    # ---- YieldTermStructure implementation ---------------------------------

    def _discount_impl(self, t: float) -> float:
        # C++ parity: ``InterpolatedSimpleZeroCurve<T>::discountImpl``.
        assert self._interpolation is not None
        t_max = self._times[-1]
        if t <= t_max:
            r = self._interpolation(t, allow_extrapolation=True)
        else:
            # Flat instantaneous forward past the last pillar.
            z_max = self._data[-1]
            inst_fwd_max = z_max + t_max * self._interpolation.derivative(
                t_max, allow_extrapolation=True
            )
            r = (z_max * t_max + inst_fwd_max * (t - t_max)) / t
        return 1.0 / (1.0 + r * t)

    # ---- inspectors ---------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def zero_rates(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
