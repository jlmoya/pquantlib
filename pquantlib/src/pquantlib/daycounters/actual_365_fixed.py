"""Actual/365 (Fixed) day counter — 3 conventions.

# C++ parity: ql/time/daycounters/actual365fixed.hpp + actual365fixed.cpp (v1.42.1).

The C++ class has three Impl variants selected by ``Convention``:

- ``Standard``  — year_fraction = (d2 - d1) / 365.
  Name: "Actual/365 (Fixed)".
- ``Canadian``  — uses ref_period_{start,end} to derive coupon frequency;
  short stubs scale by 1/365, long stubs use 1/frequency - (refLen - dcs)/365.
  Name: "Actual/365 (Fixed) Canadian Bond".
- ``NoLeap``    — Feb-29 is folded onto Feb-28 in the day-count (computes
  serial = day_of_month + month_offset[m-1] + year*365 with --serial when
  the input is Feb 29). Name: "Actual/365 (No Leap)".

Python design: instead of three Impl subclasses, ``Actual365Fixed`` dispatches
in its own ``day_count`` / ``year_fraction`` on a ``Convention`` IntEnum.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Cumulative day count through end of month-1, ignoring leap-day for
# NoLeap dispatch. Mirrors C++ MonthOffset[] in actual365fixed.cpp.
_NO_LEAP_MONTH_OFFSET: tuple[int, ...] = (
    0,
    31,
    59,
    90,
    120,
    151,
    181,
    212,
    243,
    273,
    304,
    334,
)


class Convention(IntEnum):
    Standard = 0
    Canadian = 1
    NoLeap = 2


class Actual365Fixed(DayCounter):
    def __init__(self, convention: Convention = Convention.Standard) -> None:
        super().__init__()
        self._convention: Convention = convention

    def name(self) -> str:
        if self._convention == Convention.Canadian:
            return "Actual/365 (Fixed) Canadian Bond"
        if self._convention == Convention.NoLeap:
            return "Actual/365 (No Leap)"
        return "Actual/365 (Fixed)"

    def day_count(self, d1: Date, d2: Date) -> int:
        if self._convention == Convention.NoLeap:
            s1 = d1.day_of_month() + _NO_LEAP_MONTH_OFFSET[int(d1.month()) - 1] + d1.year() * 365
            s2 = d2.day_of_month() + _NO_LEAP_MONTH_OFFSET[int(d2.month()) - 1] + d2.year() * 365
            if d1.month() == Month.February and d1.day_of_month() == 29:
                s1 -= 1
            if d2.month() == Month.February and d2.day_of_month() == 29:
                s2 -= 1
            return s2 - s1
        return d2 - d1

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        if self._convention == Convention.Canadian:
            return self._year_fraction_canadian(d1, d2, ref_period_start, ref_period_end)
        # Standard + NoLeap both divide day_count by 365.
        return self.day_count(d1, d2) / 365.0

    def _year_fraction_canadian(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None,
        ref_period_end: Date | None,
    ) -> float:
        if d1 == d2:
            return 0.0
        qassert.require(ref_period_start is not None, "invalid refPeriodStart")
        qassert.require(ref_period_end is not None, "invalid refPeriodEnd")
        assert ref_period_start is not None
        assert ref_period_end is not None

        dcs = float(d2 - d1)
        dcc = float(ref_period_end - ref_period_start)
        months = round(12 * dcc / 365)
        qassert.require(
            months != 0,
            "invalid reference period for Act/365 Canadian; must be longer than a month",
        )
        frequency = 12 // months
        qassert.require(
            frequency != 0,
            "invalid reference period for Act/365 Canadian; must not be longer than a year",
        )
        if dcs < (365 // frequency):
            return dcs / 365.0
        return 1.0 / frequency - (dcc - dcs) / 365.0
