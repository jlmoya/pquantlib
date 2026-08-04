"""ImpliedTermStructure — forward-shifted view of an existing curve.

# C++ parity: ql/termstructures/yield/impliedtermstructure.hpp (v1.43)

Given a base ``YieldTermStructure`` and a future reference date, this
class exposes a curve whose effective reference date is the future
one. Discount factors are computed by ratioing the base curve's
discount at the absolute time corresponding to a relative time ``t``
from the new reference date.

The implied curve forwards the base curve's day-counter / calendar /
max-date; only the reference date is shifted.

v1.43 made two changes here: the reference-date discount and its time offset
are cached until the original curve notifies (they only change when it does,
and recomputing them per query cost a curve lookup each time), and the implied
curve now mirrors the original's extrapolation setting instead of always
starting with extrapolation disabled.
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
        # C++ parity: impliedtermstructure.hpp:78-79 (v1.43) — the implied
        # curve inherits the original's extrapolation setting rather than
        # silently starting with extrapolation off.
        self.enable_extrapolation(original_curve.allows_extrapolation())
        # Cached reference-date discount + time offset; invalidated by update().
        # C++ parity: impliedtermstructure.hpp:68-69 ``refDf_`` / ``refTime_``.
        self._ref_df: float | None = None
        self._ref_time: float = 0.0
        original_curve.register_with(self)

    # ---- Observer interface ------------------------------------------------

    def update(self) -> None:
        """Drop the cached reference discount and re-mirror extrapolation.

        # C++ parity: ``ImpliedTermStructure::update`` (impliedtermstructure.hpp:99-105).
        """
        self._ref_df = None
        self.enable_extrapolation(self._original.allows_extrapolation())
        super().update()

    # ---- forwarded inspectors ----------------------------------------------

    def day_counter(self) -> DayCounter:
        return self._original.day_counter()

    def calendar(self) -> Calendar:
        return self._original.calendar()

    def max_date(self) -> Date:
        return self._original.max_date()

    # ---- YieldTermStructure implementation ---------------------------------

    def _discount_impl(self, t: float) -> float:
        # C++ parity: ``impliedtermstructure.hpp`` lines 107-119 (v1.43).
        # t is relative to *this* curve's reference date; convert to the
        # original curve's time axis by adding the year-fraction between
        # the original's reference date and this curve's reference date.
        #
        # v1.42.1 recomputed the reference-date discount on every query,
        # commenting that it "cannot be cached since the original curve could
        # change between invocations". It can: the implied curve observes the
        # original, so update() is exactly the notification that invalidates it.
        if self._ref_df is None:
            ref = self.reference_date()
            self._ref_time = self.day_counter().year_fraction(
                self._original.reference_date(), ref
            )
            self._ref_df = self._original.discount(ref, True)
        return self._original.discount(t + self._ref_time, True) / self._ref_df
