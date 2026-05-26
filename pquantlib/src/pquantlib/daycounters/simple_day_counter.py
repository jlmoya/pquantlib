"""SimpleDayCounter — clean whole-month year fractions.

# C++ parity: ql/time/daycounters/simpledaycounter.hpp + simpledaycounter.cpp (v1.42.1).

When both endpoints share the same day-of-month (or one is end-of-month
on its respective side), returns ``(Δy + Δm/12)`` exactly. Otherwise
falls back to ``Thirty360(BondBasis)``. Designed to be paired with
``NullCalendar`` so whole-month distances always share day-of-month.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.time.date import Date

_FALLBACK: Thirty360 = Thirty360(Convention.BondBasis)
_MONTHS_PER_YEAR_FLOAT: float = 12.0


class SimpleDayCounter(DayCounter):
    def name(self) -> str:
        return "Simple"

    def day_count(self, d1: Date, d2: Date) -> int:
        return _FALLBACK.day_count(d1, d2)

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        dm1 = d1.day_of_month()
        dm2 = d2.day_of_month()
        # Whole-month detection: same day-of-month, OR shrinking month spans
        # an end-of-month, OR growing month spans an end-of-month.
        if dm1 == dm2 or (dm1 > dm2 and Date.is_end_of_month(d2)) or (dm1 < dm2 and Date.is_end_of_month(d1)):
            return (d2.year() - d1.year()) + (int(d2.month()) - int(d1.month())) / _MONTHS_PER_YEAR_FLOAT
        return _FALLBACK.year_fraction(d1, d2)
