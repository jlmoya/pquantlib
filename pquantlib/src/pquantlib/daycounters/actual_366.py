"""Actual/366 day counter.

# C++ parity: ql/time/daycounters/actual366.hpp (v1.42.1).

When ``include_last_day`` is True, day count is ``(d2 - d1) + 1`` and
the name is "Actual/366 (inc)" — preserved exactly for C++ equality.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date


class Actual366(DayCounter):
    def __init__(self, include_last_day: bool = False) -> None:
        super().__init__()
        self._include_last_day: bool = include_last_day

    def name(self) -> str:
        return "Actual/366 (inc)" if self._include_last_day else "Actual/366"

    def day_count(self, d1: Date, d2: Date) -> int:
        return (d2 - d1) + (1 if self._include_last_day else 0)

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return self.day_count(d1, d2) / 366.0
