"""ZeroSpreadedTermStructure — base curve + Quote-driven spread on zero rate.

# C++ parity: ql/termstructures/yield/zerospreadedtermstructure.hpp (v1.42.1)

Adds the spread to the zero rate of the original curve under the
specified ``compounding`` / ``frequency``. The spreaded rate is then
re-expressed in continuous compounding before being returned, mirroring
the C++ ``equivalentRate(Continuous, NoFrequency, t)`` line.

The deprecated (in C++) day-counter argument is omitted from the
Python port (per the C++ ``[[deprecated]]`` annotation in v1.42.1).
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class ZeroSpreadedTermStructure(ZeroYieldStructure):
    """Adds a Quote-driven spread to a base curve's zero rate."""

    def __init__(
        self,
        original_curve: YieldTermStructure,
        spread: Quote,
        compounding: Compounding = Compounding.Continuous,
        frequency: Frequency = Frequency.NoFrequency,
    ) -> None:
        ZeroYieldStructure.__init__(self)
        self._original: YieldTermStructure = original_curve
        self._spread: Quote = spread
        self._comp: Compounding = compounding
        self._freq: Frequency = frequency
        self.enable_extrapolation(original_curve.allows_extrapolation())
        original_curve.register_with(self)
        spread.register_with(self)

    # ---- TermStructure overrides forwarded to the original curve -----------

    def day_counter(self) -> DayCounter:
        return self._original.day_counter()

    def calendar(self) -> Calendar:
        return self._original.calendar()

    def reference_date(self) -> Date:
        return self._original.reference_date()

    def max_date(self) -> Date:
        return self._original.max_date()

    def max_time(self) -> float:
        return self._original.max_time()

    # ---- Observer -----------------------------------------------------------

    def update(self) -> None:
        super().update()
        self.enable_extrapolation(self._original.allows_extrapolation())

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``zeroYieldImpl`` lines 142-150.
        zero_ir = self._original.zero_rate(t, self._comp, self._freq, extrapolate=True)
        spreaded = InterestRate(
            zero_ir.rate() + self._spread.value(),
            zero_ir.day_counter(),
            zero_ir.compounding(),
            zero_ir.frequency(),
        )
        return spreaded.equivalent_rate(Compounding.Continuous, Frequency.NoFrequency, t).rate()
