"""InterpolatedPiecewiseForwardSpreadedTermStructure — dated forward spreads.

# C++ parity: ql/termstructures/yield/piecewiseforwardspreadedtermstructure.hpp (v1.43)

A base curve plus a term structure of spreads on the INSTANTANEOUS FORWARD
rate, given at dates and interpolated in between. The zero-rate spread at time
t is therefore the average of the forward spread over [0, t] — the
interpolation's primitive divided by t — not the forward spread itself.

Past the last quoted date the primitive is continued linearly at the last
spread value (a flat forward spread); before the first and after the last
date ``calc_spread`` itself is flat.

C++ templates on ``Interpolator``; Python takes an interpolation factory. The
day-counter constructor overload C++ deprecated in v1.41 is not ported.
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
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedPiecewiseForwardSpreadedTermStructure(ZeroYieldStructure):
    """Base curve with an interpolated term structure of forward-rate spreads."""

    def __init__(
        self,
        original_curve: YieldTermStructure,
        spreads: Sequence[Quote],
        dates: Sequence[Date],
        interpolator: InterpolationFactory = LinearInterpolation,
    ) -> None:
        ZeroYieldStructure.__init__(self)
        qassert.require(len(spreads) > 0, "no spreads given")
        qassert.require(
            len(spreads) == len(dates), "spread and date vector have different sizes"
        )
        self._original: YieldTermStructure = original_curve
        self._spreads: list[Quote] = list(spreads)
        self._dates: list[Date] = list(dates)
        self._factory: InterpolationFactory = interpolator
        self._times: list[float] = [0.0] * len(self._dates)
        self._spread_values: list[float] = [0.0] * len(self._dates)
        self._interpolation: Interpolation | None = None
        original_curve.register_with(self)
        for spread in self._spreads:
            spread.register_with(self)
        self._update_interpolation()

    # ---- TermStructure overrides forwarded to the original curve -----------

    def day_counter(self) -> DayCounter:
        return self._original.day_counter()

    def calendar(self) -> Calendar:
        return self._original.calendar()

    def settlement_days(self) -> int:
        return self._original.settlement_days()

    def reference_date(self) -> Date:
        return self._original.reference_date()

    def max_date(self) -> Date:
        return min(self._original.max_date(), self._dates[-1])

    # ---- Observer -----------------------------------------------------------

    def update(self) -> None:
        self._update_interpolation()
        super().update()

    def _update_interpolation(self) -> None:
        # C++ parity: ``updateInterpolation``.
        for i, d in enumerate(self._dates):
            self._times[i] = self.time_from_reference(d)
            self._spread_values[i] = self._spreads[i].value()
        self._interpolation = self._factory(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._spread_values, dtype=np.float64),
        )

    # ---- spread -------------------------------------------------------------

    def _calc_spread(self, t: float) -> float:
        # C++ parity: ``calcSpread`` — flat outside the quoted range.
        if t <= self._times[0]:
            return self._spread_values[0]
        if t >= self._times[-1]:
            return self._spread_values[-1]
        assert self._interpolation is not None
        return self._interpolation(t, allow_extrapolation=True)

    def _calc_spread_primitive(self, t: float) -> float:
        # C++ parity: ``calcSpreadPrimitive`` — the average forward spread over
        # [0, t]. Past the last node the primitive grows linearly at the last
        # spread value.
        if t == 0.0:
            return self._calc_spread(0.0)
        assert self._interpolation is not None
        if t <= self._times[-1]:
            integral = self._interpolation.primitive(t, allow_extrapolation=True)
        else:
            integral = self._interpolation.primitive(
                self._times[-1], allow_extrapolation=True
            ) + self._spread_values[-1] * (t - self._times[-1])
        return integral / t

    # ---- inspectors ---------------------------------------------------------

    def spreads(self) -> list[Quote]:
        return list(self._spreads)

    def dates(self) -> list[Date]:
        return list(self._dates)

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``zeroYieldImpl`` — a bare additive shift on the
        # continuously-compounded zero rate, no compounding round trip.
        spread_primitive = self._calc_spread_primitive(t)
        zero_rate = self._original.zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        )
        return zero_rate.rate() + spread_primitive


class PiecewiseForwardSpreadedTermStructure(
    InterpolatedPiecewiseForwardSpreadedTermStructure
):
    """Linear-interpolated forward-spreaded curve (the C++ Linear typedef)."""

    def __init__(
        self,
        original_curve: YieldTermStructure,
        spreads: Sequence[Quote],
        dates: Sequence[Date],
    ) -> None:
        super().__init__(original_curve, spreads, dates, LinearInterpolation)
