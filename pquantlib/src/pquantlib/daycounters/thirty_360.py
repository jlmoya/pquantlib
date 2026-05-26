"""Thirty360 — 30/360 day counter (9 conventions).

# C++ parity: ql/time/daycounters/thirty360.hpp + thirty360.cpp (v1.42.1).

The 9 conventions reduce to 6 distinct day-count formulas (C++ aliases
several conventions to the same Impl):

- USA          — US convention.
- BondBasis    } share ISMA Impl ("30/360 (Bond Basis)" name).
- ISMA         }
- European     } share EU Impl ("30E/360 (Eurobond Basis)" name).
- EurobondBasis}
- Italian      — IT Impl.
- German       } share ISDA Impl ("30E/360 (ISDA)" name, with optional
- ISDA         } termination date).
- NASD         — NASD Impl.

The C++ pImpl split into 6 Impl classes collapses in Python to a single
``Thirty360`` class dispatching on a ``Convention`` IntEnum.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_MONTHS_PER_YEAR: int = 12
_DAYS_PER_MONTH_30_360: int = 30
_DAYS_PER_YEAR_30_360: int = 360


class Convention(IntEnum):
    USA = 0
    BondBasis = 1
    European = 2
    EurobondBasis = 3
    Italian = 4
    German = 5
    ISMA = 6
    ISDA = 7
    NASD = 8


def _is_last_of_february(d: int, m: Month, y: int) -> bool:
    """Mirrors C++ anonymous-namespace ``isLastOfFebruary``."""
    return m == Month.February and d == 28 + (1 if Date.is_leap(y) else 0)


class Thirty360(DayCounter):
    def __init__(self, convention: Convention, termination_date: Date | None = None) -> None:
        super().__init__()
        self._convention: Convention = convention
        self._termination_date: Date = termination_date if termination_date is not None else Date()

    def name(self) -> str:
        if self._convention == Convention.USA:
            return "30/360 (US)"
        if self._convention in (Convention.European, Convention.EurobondBasis):
            return "30E/360 (Eurobond Basis)"
        if self._convention == Convention.Italian:
            return "30/360 (Italian)"
        if self._convention in (Convention.ISMA, Convention.BondBasis):
            return "30/360 (Bond Basis)"
        if self._convention in (Convention.ISDA, Convention.German):
            return "30E/360 (ISDA)"
        # Convention.NASD
        return "30/360 (NASD)"

    def day_count(self, d1: Date, d2: Date) -> int:
        c = self._convention
        if c == Convention.USA:
            return _dc_us(d1, d2)
        if c in (Convention.European, Convention.EurobondBasis):
            return _dc_eu(d1, d2)
        if c == Convention.Italian:
            return _dc_it(d1, d2)
        if c in (Convention.ISMA, Convention.BondBasis):
            return _dc_isma(d1, d2)
        if c in (Convention.ISDA, Convention.German):
            return _dc_isda(d1, d2, self._termination_date)
        # Convention.NASD
        return _dc_nasd(d1, d2)

    def year_fraction(
        self,
        d1: Date,
        d2: Date,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> float:
        _ = (ref_period_start, ref_period_end)
        return self.day_count(d1, d2) / _DAYS_PER_YEAR_30_360


def _composed(yy2: int, yy1: int, mm2: int, mm1: int, dd2: int, dd1: int) -> int:
    return _DAYS_PER_YEAR_30_360 * (yy2 - yy1) + _DAYS_PER_MONTH_30_360 * (mm2 - mm1) + (dd2 - dd1)


def _dc_us(d1: Date, d2: Date) -> int:
    """US convention. Order of checks is important."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = d1.month(), d2.month()
    yy1, yy2 = d1.year(), d2.year()
    if _is_last_of_february(dd1, mm1, yy1):
        if _is_last_of_february(dd2, mm2, yy2):
            dd2 = 30
        dd1 = 30
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    if dd1 == 31:
        dd1 = 30
    return _composed(yy2, yy1, int(mm2), int(mm1), dd2, dd1)


def _dc_isma(d1: Date, d2: Date) -> int:
    """ISMA / BondBasis convention."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = d1.month(), d2.month()
    yy1, yy2 = d1.year(), d2.year()
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31 and dd1 == 30:
        dd2 = 30
    return _composed(yy2, yy1, int(mm2), int(mm1), dd2, dd1)


def _dc_eu(d1: Date, d2: Date) -> int:
    """European / EurobondBasis convention."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = d1.month(), d2.month()
    yy1, yy2 = d1.year(), d2.year()
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31:
        dd2 = 30
    return _composed(yy2, yy1, int(mm2), int(mm1), dd2, dd1)


def _dc_it(d1: Date, d2: Date) -> int:
    """Italian convention — European plus Feb-end-> 30 (when day > 27)."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = d1.month(), d2.month()
    yy1, yy2 = d1.year(), d2.year()
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31:
        dd2 = 30
    if mm1 == Month.February and dd1 > 27:
        dd1 = 30
    if mm2 == Month.February and dd2 > 27:
        dd2 = 30
    return _composed(yy2, yy1, int(mm2), int(mm1), dd2, dd1)


def _dc_isda(d1: Date, d2: Date, termination_date: Date) -> int:
    """ISDA / German convention — termination_date suppresses the d2 Feb-end fold."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = d1.month(), d2.month()
    yy1, yy2 = d1.year(), d2.year()
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31:
        dd2 = 30
    if _is_last_of_february(dd1, mm1, yy1):
        dd1 = 30
    if d2 != termination_date and _is_last_of_february(dd2, mm2, yy2):
        dd2 = 30
    return _composed(yy2, yy1, int(mm2), int(mm1), dd2, dd1)


def _dc_nasd(d1: Date, d2: Date) -> int:
    """NASD convention — d2=31 with d1<30 bumps d2 to next month's 1st."""
    dd1, dd2 = d1.day_of_month(), d2.day_of_month()
    mm1, mm2 = int(d1.month()), int(d2.month())
    yy1, yy2 = d1.year(), d2.year()
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31 and dd1 >= 30:
        dd2 = 30
    if dd2 == 31 and dd1 < 30:
        dd2 = 1
        mm2 += 1
    return _composed(yy2, yy1, mm2, mm1, dd2, dd1)
