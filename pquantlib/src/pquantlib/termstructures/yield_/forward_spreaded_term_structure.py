"""ForwardSpreadedTermStructure — base curve + constant additive spread.

# C++ parity: ql/termstructures/yield/forwardspreadedtermstructure.hpp (v1.42.1)

Despite the name, the C++ implementation adds the spread to the
*continuously-compounded zero rate* of the original curve. This is
algebraically equivalent to adding the spread to the instantaneous
forward when the spread is constant (since the integral relationship
``z(t) = (1/t) ∫ f(s) ds`` shifts uniformly).

The spread is Quote-driven so it can be bumped without rebuilding the
curve. Observability propagates through both the original curve and
the spread quote.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.yield_.zero_yield_structure import ZeroYieldStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class ForwardSpreadedTermStructure(ZeroYieldStructure):
    """Adds a constant Quote-driven spread to a base curve's zero rate."""

    def __init__(self, original_curve: YieldTermStructure, spread: Quote) -> None:
        # C++ parity: ``ForwardSpreadedTermStructure(Handle<YieldTermStructure>, Handle<Quote>)``.
        # We do NOT call super().__init__ with any reference_date /
        # day_counter — those are forwarded to the original curve.
        ZeroYieldStructure.__init__(self)
        self._original: YieldTermStructure = original_curve
        self._spread: Quote = spread
        # Mirror C++: propagate the original curve's extrapolation flag.
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
        # C++ parity: ``update()`` — propagate original's extrapolation flag.
        super().update()
        self.enable_extrapolation(self._original.allows_extrapolation())

    # ---- ZeroYieldStructure implementation ---------------------------------

    def _zero_yield_impl(self, t: float) -> float:
        # C++ parity: ``zeroYieldImpl`` — z_orig(t, Continuous, NoFrequency)
        # + spread. Continuous + NoFrequency is the C++ default for this
        # bumped rate (NoFrequency makes sense for Continuous compounding).
        original_zero = self._original.zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency, extrapolate=True
        ).rate()
        return original_zero + self._spread.value()
