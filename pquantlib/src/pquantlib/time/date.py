"""Date — serial-day-number date arithmetic.

# C++ parity: ql/time/date.hpp + ql/time/date.cpp (v1.42.1), non-boost
# (``#ifndef QL_HIGH_RESOLUTION_DATE``) branch.

Internal representation is a signed integer ``serial`` of days since the
Excel-compatible epoch of 1899-12-30 (so serial 1 = 1899-12-31, serial 367 =
1901-01-01). The Excel "1900 is leap" bug is preserved for full compatibility
with the C++ source's serial numbers — see ``is_leap(1900) == True``.

Valid serial range: ``[367, 109574]`` → 1901-01-01 to 2199-12-31. The default
``Date()`` returns a "null" date with serial 0 (outside the valid range);
``Date(serial)`` for any non-zero serial validates and raises
``LibraryException`` on out-of-range.

Python idiomatic mappings vs C++:
- C++ mutating ``+=``, ``-=``, ``++``, ``--`` → Python ``+``, ``-`` returning
  new ``Date`` instances (the class is ``@dataclass(frozen=True, slots=True)``).
- C++ ``Date(d, m, y)`` constructor → ``Date.from_ymd(d, m, y)`` classmethod.
- C++ static methods stay as ``@classmethod`` / ``@staticmethod``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _stdlib_date
from typing import Final, NoReturn, overload

from pquantlib import qassert
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday

# --- static tables ---------------------------------------------------------

# Min/max from ql/time/date.cpp ``minimumSerialNumber`` / ``maximumSerialNumber``.
_MIN_SERIAL: Final[int] = 367  # 1901-01-01
_MAX_SERIAL: Final[int] = 109574  # 2199-12-31
_MIN_YEAR: Final[int] = 1901
_MAX_YEAR: Final[int] = 2199

# Days in each month: index 0 unused, [1..12] populated.
_MONTH_LENGTH: Final[tuple[int, ...]] = (
    0,
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
)
_MONTH_LEAP_LENGTH: Final[tuple[int, ...]] = (
    0,
    31,
    29,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
)

# Cumulative days through end of month-1 (i.e., monthOffset[m] = day-of-year
# of first day of month m, minus 1). Index 0 unused; [1..13] populated where
# index 13 = total days (365 / 366).
_MONTH_OFFSET: Final[tuple[int, ...]] = (
    0,
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
    365,
)
_MONTH_LEAP_OFFSET: Final[tuple[int, ...]] = (
    0,
    0,
    31,
    60,
    91,
    121,
    152,
    182,
    213,
    244,
    274,
    305,
    335,
    366,
)


def _build_year_offset() -> tuple[int, ...]:
    """Cumulative serial through Dec 31 of (y-1), indexed by ``y - 1900``.

    The first entry is 0, matching C++ ``yearOffset[0] == 0`` (Dec 31, 1899).
    """
    out: list[int] = [0]
    for y in range(1900, 2201):
        out.append(out[-1] + (366 if is_leap(y) else 365))
    return tuple(out)


def is_leap(year: int) -> bool:
    """Whether ``year`` is a leap year per the QuantLib calendar.

    Preserves the Excel-compatible "1900 is leap" bug so serial numbers
    align with the C++ ``yearOffset`` table.
    """
    if year == 1900:
        return True  # Excel 1900-is-leap bug, preserved for serial compatibility
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


_YEAR_OFFSET: Final[tuple[int, ...]] = _build_year_offset()


def _check_serial(serial: int) -> None:
    qassert.require(
        _MIN_SERIAL <= serial <= _MAX_SERIAL,
        f"Date's serial number ({serial}) outside allowed range [{_MIN_SERIAL}-{_MAX_SERIAL}]",
    )


def _month_length(m: Month | int, leap: bool) -> int:
    return _MONTH_LEAP_LENGTH[int(m)] if leap else _MONTH_LENGTH[int(m)]


def _month_offset(m: Month | int, leap: bool) -> int:
    """Cumulative day-of-year through end of month ``m - 1``.

    Accepts ``int`` rather than just ``Month`` because the ``Date.month``
    algorithm probes ``monthOffset(13, leap)`` as the year-end sentinel
    (C++ allows ``Month(13)`` as an unchecked enum cast; Python ``IntEnum``
    does not).
    """
    idx = int(m)
    return _MONTH_LEAP_OFFSET[idx] if leap else _MONTH_OFFSET[idx]


def _year_offset(year: int) -> int:
    return _YEAR_OFFSET[year - 1900]


def _unknown_unit(u: TimeUnit) -> NoReturn:
    qassert.fail(f"undefined time units ({int(u)})")


# --- the class -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Date:
    """Serial-day-number date.

    The ``serial`` is days since 1899-12-30 (Excel convention). Valid
    non-null range: ``[367, 109574]`` → 1901-01-01..2199-12-31.
    """

    serial: int = 0

    def __post_init__(self) -> None:
        if self.serial != 0:
            _check_serial(self.serial)

    # --- alternate constructors --------------------------------------------

    @classmethod
    def from_ymd(cls, day: int, month: Month, year: int) -> Date:
        """Mirrors C++ ``Date(Day d, Month m, Year y)``."""
        qassert.require(
            _MIN_YEAR <= year <= _MAX_YEAR,
            f"year {year} out of bound. It must be in [{_MIN_YEAR},{_MAX_YEAR}]",
        )
        qassert.require(
            1 <= int(month) <= 12,
            f"month {int(month)} outside January-December range [1,12]",
        )
        leap = is_leap(year)
        length = _month_length(month, leap)
        offset = _month_offset(month, leap)
        qassert.require(
            1 <= day <= length,
            f"day outside month ({int(month)}) day-range [1,{length}]",
        )
        return cls(day + offset + _year_offset(year))

    @classmethod
    def todays_date(cls) -> Date:
        """Today's date in the host's local timezone."""
        t = _stdlib_date.today()
        return cls.from_ymd(t.day, Month(t.month), t.year)

    @classmethod
    def min_date(cls) -> Date:
        return cls(_MIN_SERIAL)

    @classmethod
    def max_date(cls) -> Date:
        return cls(_MAX_SERIAL)

    # --- inspectors --------------------------------------------------------

    def weekday(self) -> Weekday:
        w = self.serial % 7
        return Weekday(7 if w == 0 else w)

    def day_of_year(self) -> int:
        return self.serial - _year_offset(self.year())

    def year(self) -> int:
        y = (self.serial // 365) + 1900
        # yearOffset[y - 1900] is Dec 31 of (y - 1); if our serial is at or
        # before that, we're still in year (y - 1).
        if self.serial <= _year_offset(y):
            y -= 1
        return y

    def month(self) -> Month:
        d = self.day_of_year()
        m = d // 30 + 1
        leap = is_leap(self.year())
        while d <= _month_offset(m, leap):
            m -= 1
        while d > _month_offset(m + 1, leap):
            m += 1
        return Month(m)

    def day_of_month(self) -> int:
        return self.day_of_year() - _month_offset(self.month(), is_leap(self.year()))

    def serial_number(self) -> int:
        """C++ ``serialNumber()`` accessor (alias for the ``serial`` field)."""
        return self.serial

    # --- arithmetic --------------------------------------------------------

    @overload
    def __add__(self, other: int) -> Date: ...
    @overload
    def __add__(self, other: Period) -> Date: ...

    def __add__(self, other: int | Period) -> Date:
        if isinstance(other, Period):
            return _advance(self, other.length, other.units)
        return Date(self.serial + int(other))

    def __radd__(self, other: int) -> Date:
        return self.__add__(other)

    # `Date - int` and `Date - Period` return a new Date.
    # `Date - Date` returns an integer day count.
    @overload
    def __sub__(self, other: int) -> Date: ...
    @overload
    def __sub__(self, other: Period) -> Date: ...
    @overload
    def __sub__(self, other: Date) -> int: ...

    def __sub__(self, other: int | Period | Date) -> Date | int:
        if isinstance(other, Date):
            return self.serial - other.serial
        if isinstance(other, Period):
            return _advance(self, -other.length, other.units)
        return Date(self.serial - int(other))

    # --- ordering / equality (dataclass provides __eq__ / __hash__) --------

    def __lt__(self, other: Date) -> bool:
        return self.serial < other.serial

    def __le__(self, other: Date) -> bool:
        return self.serial <= other.serial

    def __gt__(self, other: Date) -> bool:
        return self.serial > other.serial

    def __ge__(self, other: Date) -> bool:
        return self.serial >= other.serial

    # --- string forms ------------------------------------------------------

    def __str__(self) -> str:
        if self.serial == 0:
            return "Date(null)"
        return f"{self.year():04d}-{int(self.month()):02d}-{self.day_of_month():02d}"

    # --- static helpers ----------------------------------------------------

    @staticmethod
    def is_leap(year: int) -> bool:
        return is_leap(year)

    @classmethod
    def end_of_month(cls, d: Date) -> Date:
        m = d.month()
        y = d.year()
        return cls.from_ymd(_month_length(m, is_leap(y)), m, y)

    @classmethod
    def is_end_of_month(cls, d: Date) -> bool:
        return d.day_of_month() == _month_length(d.month(), is_leap(d.year()))

    @classmethod
    def start_of_month(cls, d: Date) -> Date:
        return cls.from_ymd(1, d.month(), d.year())

    @classmethod
    def is_start_of_month(cls, d: Date) -> bool:
        return d.day_of_month() == 1

    @classmethod
    def next_weekday(cls, d: Date, target: Weekday) -> Date:
        """Smallest ``Date >= d`` whose weekday is ``target``.

        Mirrors C++ ``Date::nextWeekday``: returns ``d`` itself if it already
        has the target weekday.
        """
        wd = int(d.weekday())
        t = int(target)
        days = (7 if wd > t else 0) - wd + t
        return d + days

    @classmethod
    def nth_weekday(cls, n: int, target: Weekday, month: Month, year: int) -> Date:
        """``n``-th occurrence of ``target`` weekday in (``month``, ``year``).

        Mirrors C++ ``Date::nthWeekday``. ``n`` must be in ``[1, 5]``.
        """
        qassert.require(n > 0, "zeroth day of week in a given (month, year) is undefined")
        qassert.require(n < 6, "no more than 5 weekday in a given (month, year)")
        first = int(cls.from_ymd(1, month, year).weekday())
        t = int(target)
        skip = n - (1 if t >= first else 0)
        return cls.from_ymd((1 + t + skip * 7) - first, month, year)


# --- internal: Period advance helper ---------------------------------------


def _advance(date: Date, n: int, units: TimeUnit) -> Date:
    """Mirrors C++ ``Date::advance`` (date.cpp, non-boost branch)."""
    if units == TimeUnit.Days:
        return date + n
    if units == TimeUnit.Weeks:
        return date + 7 * n
    if units == TimeUnit.Months:
        d = date.day_of_month()
        m = int(date.month()) + n
        y = date.year()
        while m > 12:
            m -= 12
            y += 1
        while m < 1:
            m += 12
            y -= 1
        qassert.require(
            _MIN_YEAR - 1 <= y <= _MAX_YEAR,
            f"year {y} out of bounds. It must be in [{_MIN_YEAR},{_MAX_YEAR}]",
        )
        length = _month_length(Month(m), is_leap(y))
        d = min(d, length)
        return Date.from_ymd(d, Month(m), y)
    if units == TimeUnit.Years:
        d = date.day_of_month()
        m = date.month()
        y = date.year() + n
        qassert.require(
            _MIN_YEAR - 1 <= y <= _MAX_YEAR,
            f"year {y} out of bounds. It must be in [{_MIN_YEAR},{_MAX_YEAR}]",
        )
        # Feb 29 in non-leap target year → clip to Feb 28.
        if d == 29 and m == Month.February and not is_leap(y):
            d = 28
        return Date.from_ymd(d, m, y)
    _unknown_unit(units)
