"""OneDayCounter — 1/1 convention (sign-only day count).

# C++ parity: ql/time/daycounters/one.hpp (v1.42.1).

``day_count`` returns the sign of ``d2 - d1`` (+1 if ``d2 >= d1``, -1
otherwise). ``year_fraction`` returns the same value as a float.

Name is the literal string ``"1/1"`` — preserved exactly for C++ equality.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date


class OneDayCounter(DayCounter):
    def name(self) -> str:
        return "1/1"

    def day_count(self, d1: Date, d2: Date) -> int:
        return 1 if d2 >= d1 else -1

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return float(self.day_count(d1, d2))
