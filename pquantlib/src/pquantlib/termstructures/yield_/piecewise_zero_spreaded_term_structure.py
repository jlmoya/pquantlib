"""InterpolatedPiecewiseZeroSpreadedTermStructure — dated zero-rate spreads.

# C++ parity: ql/termstructures/yield/piecewisezerospreadedtermstructure.hpp (v1.43)

A base curve plus a term structure of zero-rate spreads given at dates and
interpolated in between. Outside the quoted range the spread is held flat at
the first / last value — NOT extrapolated by the interpolator.

C++ templates on ``Interpolator``; Python takes an interpolation factory, the
same shape ``InterpolatedDiscountCurve`` already uses. The C++ ``typedef
PiecewiseZeroSpreadedTermStructure`` (Linear) is provided as a subclass so the
name survives.

The day-counter constructor overload C++ deprecated in v1.41 is not ported.
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
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

InterpolationFactory = Callable[[Array, Array], Interpolation]


class InterpolatedPiecewiseZeroSpreadedTermStructure(ZeroYieldStructure):
    """Base curve with an interpolated term structure of zero-rate spreads."""

    def __init__(
        self,
        original_curve: YieldTermStructure,
        spreads: Sequence[Quote],
        dates: Sequence[Date],
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.NoFrequency,
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
        self._comp: Compounding = compounding
        self._freq: Frequency = frequency
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
        # C++ parity: ``updateInterpolation`` — times and spread values are
        # both re-read, then the interpolation is rebuilt over them.
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

    # ---- inspectors ---------------------------------------------------------

    def spreads(self) -> list[Quote]:
        return list(self._spreads)

    def dates(self) -> list[Date]:
        return list(self._dates)

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``zeroYieldImpl``.
        spread = self._calc_spread(t)
        zero_rate = self._original.zero_rate(t, self._comp, self._freq, extrapolate=True)
        spreaded = InterestRate(
            zero_rate.rate() + spread,
            zero_rate.day_counter(),
            zero_rate.compounding(),
            zero_rate.frequency(),
        )
        return spreaded.equivalent_rate(
            Compounding.Continuous, Frequency.NoFrequency, t
        ).rate()


class PiecewiseZeroSpreadedTermStructure(InterpolatedPiecewiseZeroSpreadedTermStructure):
    """Linear-interpolated zero-spreaded curve.

    # C++ parity: the ``typedef InterpolatedPiecewiseZeroSpreadedTermStructure<Linear>``
    # in piecewisezerospreadedtermstructure.hpp.
    """

    def __init__(
        self,
        original_curve: YieldTermStructure,
        spreads: Sequence[Quote],
        dates: Sequence[Date],
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.NoFrequency,
    ) -> None:
        super().__init__(
            original_curve, spreads, dates, compounding, frequency, LinearInterpolation
        )
