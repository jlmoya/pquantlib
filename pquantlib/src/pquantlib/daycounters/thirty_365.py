"""Thirty365 — 30/365 day counter (ISO 20022 adjustment rules).

# C++ parity: ql/time/daycounters/thirty365.hpp + thirty365.cpp (v1.42.1).

day_count adjustments mirror C++ exactly: d1 = 31 → 30, d2 = 31 → 30.
year_fraction = day_count / 365.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date

_DAYS_PER_MONTH_30: int = 30
_DAYS_PER_YEAR_360_NUMER: int = 360
_YEAR_DIVISOR_365: float = 365.0


class Thirty365(DayCounter):
    def name(self) -> str:
        return "30/365"

    def day_count(self, d1: Date, d2: Date) -> int:
        dd1, dd2 = d1.day_of_month(), d2.day_of_month()
        mm1, mm2 = int(d1.month()), int(d2.month())
        yy1, yy2 = d1.year(), d2.year()
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31:
            dd2 = 30
        return _DAYS_PER_YEAR_360_NUMER * (yy2 - yy1) + _DAYS_PER_MONTH_30 * (mm2 - mm1) + (dd2 - dd1)

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return self.day_count(d1, d2) / _YEAR_DIVISOR_365
