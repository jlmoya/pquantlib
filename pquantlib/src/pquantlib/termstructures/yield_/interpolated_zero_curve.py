"""InterpolatedZeroCurve — yield curve from zero rates at known dates.

# C++ parity: ql/termstructures/yield/zerocurve.hpp (v1.42.1)

C++ templates ``InterpolatedZeroCurve<Interpolator>``; Python takes a
``InterpolationFactory`` callable that maps ``(xs, ys) → Interpolation``
(use ``LinearInterpolation`` directly — its constructor matches the
signature). Default is ``LinearInterpolation`` (the C++ ``ZeroCurve``
typedef specializes the template the same way).

The zero rates are converted to continuously-compounded form internally
when ``compounding`` differs from ``Continuous``, mirroring C++
``initialize()`` (see lines 250-277 of zerocurve.hpp).

Flat-forward extrapolation past the last knot, per C++ ``zeroYieldImpl``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedZeroCurve(ZeroYieldStructure):
    """Yield curve from zero rates at known dates.

    Inputs:
    - ``dates``: sequence with ``dates[0]`` = reference date.
    - ``yields``: zero rates in the supplied compounding/frequency.
    - ``day_counter``: used to convert dates → times.
    - ``calendar``: optional, defaults to None (NullCalendar in C++).
    - ``interpolator``: factory ``(xs, ys) → Interpolation``. Default
      ``LinearInterpolation`` (the C++ ``ZeroCurve`` typedef).
    - ``compounding`` / ``frequency``: input rate convention; converted
      to ``Continuous`` internally if not already.

    Beyond the last date, ``_zero_yield_impl`` extrapolates flatly in
    forward rate (C++ parity, ``zerocurve.hpp`` lines 161-169).
    """

    def __init__(
        self,
        dates: Sequence[Date],
        yields: Sequence[float],
        day_counter: DayCounter,
        calendar: Calendar | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
        interpolator: InterpolationFactory = LinearInterpolation,
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.Annual,
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
        self._data: list[float] = list(yields)  # mutable; may be rewritten if comp != Continuous
        self._interpolator: InterpolationFactory = interpolator
        self._compounding: Compounding = compounding
        self._frequency: Frequency = frequency
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        # C++ parity: ``initialize(comp, freq)`` — adjusts rates if non-Continuous
        # and sets up the interpolation.
        qassert.require(
            len(self._data) == len(self._dates), "dates/data count mismatch"
        )
        # Set up times relative to dates[0].
        ref = self._dates[0]
        dc = self.day_counter()
        self._times = [dc.year_fraction(ref, d) for d in self._dates]

        if self._compounding != Compounding.Continuous:
            # C++ parity: ``initialize()`` lines 261-273 — convert all rates
            # to continuous compounding. For node 0 (time = 0), fall back
            # to ~1 day (dt = 1/365).
            dt = 1.0 / 365.0
            r0 = InterestRate(self._data[0], dc, self._compounding, self._frequency)
            self._data[0] = r0.equivalent_rate(Compounding.Continuous, Frequency.NoFrequency, dt).rate()
            for i in range(1, len(self._dates)):
                ri = InterestRate(self._data[i], dc, self._compounding, self._frequency)
                self._data[i] = ri.equivalent_rate(
                    Compounding.Continuous, Frequency.NoFrequency, self._times[i]
                ).rate()
        # Build the interpolation.
        self._interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- TermStructure interface ------------------------------------------

    def max_date(self) -> Date:
        return self._dates[-1]

    # ---- ZeroYieldStructure implementation --------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``zerocurve.hpp`` lines 160-169.
        assert self._interpolation is not None
        last_t = self._times[-1]
        if t <= last_t:
            return self._interpolation(t, allow_extrapolation=True)
        # Flat-forward extrapolation past the last knot:
        # ``instFwdMax = zMax + tMax * derivative(tMax)``;
        # then z(t) = (zMax*tMax + instFwdMax*(t-tMax)) / t.
        z_max = self._data[-1]
        inst_fwd_max = z_max + last_t * self._interpolation.derivative(last_t, allow_extrapolation=True)
        return (z_max * last_t + inst_fwd_max * (t - last_t)) / t

    # ---- inspectors --------------------------------------------------------

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
