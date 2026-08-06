"""UltimateForwardTermStructure — Dutch (DNB) UFR extrapolation of a base curve.

# C++ parity: ql/termstructures/yield/ultimateforwardtermstructure.hpp (v1.43)

Regulatory term structure for Dutch pension funds. Up to the first smoothing
point the curve reproduces the original curve's continuously-compounded zero
rate; beyond it the forward is extrapolated towards an Ultimate Forward Rate::

    f(t, T_c, T) = UFR + (LLFR - UFR) * B(T - T_c)
    B(dT)        = (1 - exp(-alpha * dT)) / (alpha * dT)

and the zero rate is the time-weighted blend of the base zero up to the
cut-off with that extrapolated forward beyond it.

Optionally the resulting zero rate is rounded to a fixed number of decimals,
under a chosen compounding — the rate is converted out of continuous
compounding, rounded, and converted back.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.math.rounding import ClosestRounding
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class UltimateForwardTermStructure(ZeroYieldStructure):
    """Base curve extended past a smoothing point towards an ultimate forward."""

    def __init__(
        self,
        original_curve: YieldTermStructure,
        last_liquid_forward_rate: Quote,
        ultimate_forward_rate: Quote,
        first_smoothing_point: Period,
        alpha: float,
        rounding_digits: int | None = None,
        compounding: Compounding = Compounding.Compounded,
        frequency: Frequency = Frequency.Annual,
    ) -> None:
        ZeroYieldStructure.__init__(self)
        qassert.require(
            first_smoothing_point.length > 0,
            "first smoothing point must be a period with positive length",
        )
        self._original: YieldTermStructure = original_curve
        self._llfr: Quote = last_liquid_forward_rate
        self._ufr: Quote = ultimate_forward_rate
        self._fsp: Period = first_smoothing_point
        self._alpha: float = alpha
        self._rounding_digits: int | None = rounding_digits
        self._compounding: Compounding = compounding
        self._frequency: Frequency = frequency
        self.enable_extrapolation(original_curve.allows_extrapolation())
        original_curve.register_with(self)
        last_liquid_forward_rate.register_with(self)
        ultimate_forward_rate.register_with(self)

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
        # C++ parity: Date::maxDate() — the whole point is unbounded
        # extrapolation past the base curve.
        return Date.max_date()

    # ---- Observer -----------------------------------------------------------

    def update(self) -> None:
        super().update()
        self.enable_extrapolation(self._original.allows_extrapolation())

    # ---- inspectors ---------------------------------------------------------

    def last_liquid_forward_rate(self) -> Quote:
        return self._llfr

    def ultimate_forward_rate(self) -> Quote:
        return self._ufr

    def first_smoothing_point(self) -> Period:
        return self._fsp

    def alpha(self) -> float:
        return self._alpha

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _apply_rounding(self, r: float, t: float) -> float:
        """Round the zero rate, converting compounding around the rounding.

        # C++ parity: ``UltimateForwardTermStructure::applyRounding``.
        """
        if self._rounding_digits is None:
            return r
        # The input rate is continuously compounded by definition, so when
        # that is also the rounding compounding no conversion is needed.
        if self._compounding == Compounding.Continuous:
            return ClosestRounding(self._rounding_digits)(r)
        equivalent = InterestRate(
            r, self.day_counter(), Compounding.Continuous, Frequency.NoFrequency
        ).equivalent_rate(self._compounding, self._frequency, t).rate()
        rounded = ClosestRounding(self._rounding_digits)(equivalent)
        return (
            InterestRate(rounded, self.day_counter(), self._compounding, self._frequency)
            .equivalent_rate(Compounding.Continuous, Frequency.NoFrequency, t)
            .rate()
        )

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``UltimateForwardTermStructure::zeroYieldImpl``.
        cut_off_time = self._original.time_from_reference(self.reference_date() + self._fsp)
        delta_t = t - cut_off_time
        if delta_t > 0.0:
            base_rate = self._original.zero_rate(
                cut_off_time, Compounding.Continuous, Frequency.NoFrequency
            ).rate()
            beta = (1.0 - math.exp(-self._alpha * delta_t)) / (self._alpha * delta_t)
            ufr = self._ufr.value()
            extrapolated_forward = ufr + (self._llfr.value() - ufr) * beta
            return self._apply_rounding(
                (cut_off_time * base_rate + delta_t * extrapolated_forward) / t, t
            )
        return self._apply_rounding(
            self._original.zero_rate(
                t, Compounding.Continuous, Frequency.NoFrequency
            ).rate(),
            t,
        )
