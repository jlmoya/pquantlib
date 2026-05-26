"""Actual/360 day counter.

# C++ parity: ql/time/daycounters/actual360.hpp (v1.42.1).

When ``include_last_day`` is True, day count is ``(d2 - d1) + 1`` and
the name is "Actual/360 (inc)" — preserved exactly for C++ equality.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date


class Actual360(DayCounter):
    def __init__(self, include_last_day: bool = False) -> None:
        super().__init__()
        self._include_last_day: bool = include_last_day

    def name(self) -> str:
        return "Actual/360 (inc)" if self._include_last_day else "Actual/360"

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
        return self.day_count(d1, d2) / 360.0
