"""CompositeZeroYieldStructure — two curves combined by a binary function.

# C++ parity: ql/termstructures/yield/compositezeroyieldstructure.hpp (v1.43)

C++ templates on ``BinaryFunction``; Python takes a plain
``Callable[[float, float], float]``. Both curves' zero rates are read under
the same ``compounding`` / ``frequency``, combined, and the result converted
back to continuous compounding.
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

BinaryFunction = Callable[[float, float], float]


class CompositeZeroYieldStructure(ZeroYieldStructure):
    """Zero-yield curve built from two curves and a binary combining function."""

    def __init__(
        self,
        curve1: YieldTermStructure,
        curve2: YieldTermStructure,
        f: BinaryFunction,
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.NoFrequency,
    ) -> None:
        ZeroYieldStructure.__init__(self)
        self._curve1: YieldTermStructure = curve1
        self._curve2: YieldTermStructure = curve2
        self._f: BinaryFunction = f
        self._comp: Compounding = compounding
        self._freq: Frequency = frequency
        self.enable_extrapolation(
            curve1.allows_extrapolation() and curve2.allows_extrapolation()
        )
        curve1.register_with(self)
        curve2.register_with(self)

    # ---- TermStructure overrides forwarded to curve1 -----------------------

    def day_counter(self) -> DayCounter:
        return self._curve1.day_counter()

    def calendar(self) -> Calendar:
        return self._curve1.calendar()

    def settlement_days(self) -> int:
        return self._curve1.settlement_days()

    def reference_date(self) -> Date:
        return self._curve1.reference_date()

    def max_date(self) -> Date:
        return self._curve1.max_date()

    def max_time(self) -> float:
        return self._curve1.max_time()

    # ---- Observer -----------------------------------------------------------

    def update(self) -> None:
        super().update()
        self.enable_extrapolation(
            self._curve1.allows_extrapolation() and self._curve2.allows_extrapolation()
        )

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``CompositeZeroYieldStructure::zeroYieldImpl``. Note that
        # C++ binds curve1's rate as a bare Rate and curve2's as an InterestRate
        # — the binary function sees two numbers either way.
        zero1 = self._curve1.zero_rate(t, self._comp, self._freq, extrapolate=True).rate()
        zero2 = self._curve2.zero_rate(t, self._comp, self._freq, extrapolate=True).rate()
        composite = InterestRate(
            self._f(zero1, zero2), self.day_counter(), self._comp, self._freq
        )
        return composite.equivalent_rate(
            Compounding.Continuous, Frequency.NoFrequency, t
        ).rate()
