"""Actual/364 day counter.

# C++ parity: ql/time/daycounters/actual364.hpp (v1.42.1).

No include-last-day variant in C++; year fraction is (d2 - d1) / 364.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date


class Actual364(DayCounter):
    def name(self) -> str:
        return "Actual/364"

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return (d2 - d1) / 364.0
