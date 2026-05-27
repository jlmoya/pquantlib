"""InterpolatedForwardCurve — yield curve from instantaneous forward rates.

# C++ parity: ql/termstructures/yield/forwardcurve.hpp (v1.42.1)

C++ templates ``InterpolatedForwardCurve<Interpolator>``; Python takes
an ``InterpolationFactory`` callable. Default is
``BackwardFlatInterpolation`` (the C++ ``ForwardCurve`` typedef).

Forward rates are interpolated; zero yields are derived via the
``primitive(t)`` of the interpolation (i.e. ``z(t) = (1/t) *
∫₀ᵗ f(s) ds``). Past the last knot the curve extrapolates the forward
rate flat.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedForwardCurve(ZeroYieldStructure):
    """Yield curve from instantaneous forward rates at known dates.

    Inputs mirror ``InterpolatedZeroCurve`` but the ``forwards``
    parameter carries instantaneous forwards (annual continuous
    compounding) rather than zero rates. Conversion to zero yield is
    automatic via the interpolation's ``primitive``.
    """

    def __init__(
        self,
        dates: Sequence[Date],
        forwards: Sequence[float],
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
        self._data: list[float] = list(forwards)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
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

    # ---- ZeroYieldStructure implementation --------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``forwardcurve.hpp`` lines 155-169.
        assert self._interpolation is not None
        if t == 0.0:
            return self._interpolation(t, allow_extrapolation=True)
        max_time = self._times[-1]
        if t <= max_time:
            integral = self._interpolation.primitive(t, allow_extrapolation=True)
        else:
            # Flat forward extrapolation past the last knot.
            integral = self._interpolation.primitive(max_time, allow_extrapolation=True) + self._data[-1] * (
                t - max_time
            )
        return integral / t

    # ---- inspectors --------------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def forwards(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
