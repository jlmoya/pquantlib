"""Business252 — business/252 day counter (calendar-dependent).

# C++ parity: ql/time/daycounters/business252.hpp + business252.cpp (v1.42.1).

day_count counts business days between d1 and d2 against the supplied
calendar. year_fraction = day_count / 252.

Divergence from C++:
- The C++ default constructor uses ``Brazil()`` calendar. Brazil is not
  yet ported (Stage 4); the Python port requires an explicit calendar.
  Documented inline.
- The C++ impl maintains module-level monthly + yearly business-day caches
  keyed by calendar name to amortize cost over multi-year date ranges.
  The Python port skips the cache and calls ``calendar.business_days_between``
  directly — algorithmically equivalent, just slower for very long
  date spans. Cache can be added later if profiling identifies it as a
  hotspot.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

_BUSINESS_DAYS_PER_YEAR: float = 252.0


class Business252(DayCounter):
    def __init__(self, calendar: Calendar) -> None:
        super().__init__()
        self._calendar: Calendar = calendar

    def name(self) -> str:
        return f"Business/252({self._calendar.name()})"

    def day_count(self, d1: Date, d2: Date) -> int:
        return self._calendar.business_days_between(d1, d2, include_first=True, include_last=False)

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return self.day_count(d1, d2) / _BUSINESS_DAYS_PER_YEAR
