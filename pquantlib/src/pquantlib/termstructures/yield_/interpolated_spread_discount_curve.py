"""InterpolatedSpreadDiscountCurve — base curve * interpolated discount spread.

# C++ parity: ql/termstructures/yield/spreaddiscountcurve.hpp (v1.42.1)
#   typedef InterpolatedSpreadDiscountCurve<LogLinear> SpreadDiscountCurve;

Multiplies the base curve's discount factor by an interpolated spread
factor at each time. The spread is given as a sequence of (date,
discount-spread) pairs; the first spread must equal 1.0 (anchoring at
the reference date). Flat-forward extrapolation past the last knot.

The cluster scope describes this as "DiscountSpreadedTermStructure" —
there is no such class in C++; this is the canonical C++ equivalent and
``DiscountSpreadedTermStructure`` is a type alias under
``pquantlib.termstructures.yield_.discount_spreaded_term_structure``.
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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedSpreadDiscountCurve(YieldTermStructure):
    """Yield curve = base curve * interpolated discount-factor spread."""

    def __init__(
        self,
        base_curve: YieldTermStructure,
        dates: Sequence[Date],
        dfs: Sequence[float],
        interpolator: InterpolationFactory = LogLinearInterpolation,
    ) -> None:
        # C++ parity: ``InterpolatedSpreadDiscountCurve(base, dates, dfs, interp)``.
        # We don't pass a reference_date/calendar/day_counter to the
        # super: they're forwarded to the base curve. But the C++ class
        # extends YieldTermStructure (which extends TermStructure) so we
        # need a TermStructure instance — give it None across the board
        # since the inspectors all delegate.
        YieldTermStructure.__init__(self)
        self._base: YieldTermStructure = base_curve
        self._dates: list[Date] = list(dates)
        self._data: list[float] = list(dfs)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        # Validation mirrors the C++ initializer-list:
        qassert.require(len(self._dates) >= 1, "no input dates given")
        qassert.require(len(self._data) == len(self._dates), "dates/data count mismatch")
        qassert.require(
            self._data[0] == 1.0,
            "the first discount must be == 1.0 to flag the corresponding date as reference date",
        )
        for i in range(1, len(self._dates)):
            qassert.require(self._data[i] > 0.0, "negative discount")
        base_curve.register_with(self)
        self._update_interpolation()

    def _update_interpolation(self) -> None:
        # C++ parity: ``updateInterpolation``. We rebuild on every update
        # since Python doesn't have the C++ ``prevDayCount_`` optimization
        # (and the cost is negligible).
        qassert.require(
            self._dates[0] == self._base.reference_date(),
            "the first date should be the same as in the original curve",
        )
        dc = self.day_counter()
        self._times = [dc.year_fraction(self._dates[0], d) for d in self._dates]
        self._interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- TermStructure overrides forwarded to the base curve ---------------

    def day_counter(self) -> DayCounter:
        return self._base.day_counter()

    def calendar(self) -> Calendar:
        return self._base.calendar()

    def reference_date(self) -> Date:
        return self._base.reference_date()

    def max_date(self) -> Date:
        # C++: min(maxDate_, dates_.back()) — we have no maxDate_ override
        # mechanism, so just return min(base.max_date(), dates_.back()).
        return min(self._base.max_date(), self._dates[-1])

    # ---- Observer -----------------------------------------------------------

    def update(self) -> None:
        if self._dates:
            self._update_interpolation()
        super().update()

    # ---- YieldTermStructure implementation ---------------------------------

    def _calc_spread(self, t: float) -> float:
        # C++ parity: ``calcSpread(t)`` — interpolated spread with flat-fwd
        # extrapolation past the last knot.
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        d_max = self._data[-1]
        inst_fwd_max = -self._interpolation.derivative(max_time, allow_extrapolation=True) / d_max
        return d_max * math.exp(-inst_fwd_max * (t - max_time))

    def _discount_impl(self, t: float) -> float:
        # C++ parity: ``base.discount(t) * calcSpread(t)``.
        return self._base.discount(t) * self._calc_spread(t)

    # ---- inspectors --------------------------------------------------------

    def base_curve(self) -> YieldTermStructure:
        return self._base

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        return list(zip(self._dates, self._data, strict=True))
