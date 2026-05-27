"""ImpliedTermStructure — forward-shifted view of an existing curve.

# C++ parity: ql/termstructures/yield/impliedtermstructure.hpp (v1.42.1)

Given a base ``YieldTermStructure`` and a future reference date, this
class exposes a curve whose effective reference date is the future
one. Discount factors are computed by ratioing the base curve's
discount at the absolute time corresponding to a relative time ``t``
from the new reference date.

The implied curve forwards the base curve's day-counter / calendar /
max-date; only the reference date is shifted.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class ImpliedTermStructure(YieldTermStructure):
    """Forward-shifted view of an existing yield curve."""

    def __init__(self, original_curve: YieldTermStructure, reference_date: Date) -> None:
        # C++ parity: ``ImpliedTermStructure(Handle<YieldTermStructure>, Date)``.
        # We pass the future reference date to the base class so
        # ``reference_date()`` returns it; ``day_counter`` / ``calendar``
        # forward to the original curve (overridden below).
        YieldTermStructure.__init__(self, reference_date=reference_date)
        self._original: YieldTermStructure = original_curve
        original_curve.register_with(self)

    # ---- forwarded inspectors ----------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._original.day_counter()

    def calendar(self) -> Calendar:
        return self._original.calendar()

    def max_date(self) -> Date:
        return self._original.max_date()

    # ---- YieldTermStructure implementation ---------------------------------

    def _discount_impl(self, t: float) -> float:
        # C++ parity: ``impliedtermstructure.hpp`` lines 90-102.
        # t is relative to *this* curve's reference date; convert to the
        # original curve's time axis by adding the year-fraction between
        # the original's reference date and this curve's reference date.
        ref = self.reference_date()
        original_time = t + self.day_counter().year_fraction(self._original.reference_date(), ref)
        # The original-curve discount at *our* reference date cannot be
        # cached because the original curve may change between calls.
        return self._original.discount(original_time, True) / self._original.discount(ref, True)
