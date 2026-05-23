"""Period — integer length * TimeUnit.

# C++ parity: ql/time/period.hpp + ql/time/period.cpp (v1.42.1).

The Python port is a ``@dataclass(frozen=True, slots=True)`` over
``(length: int, units: TimeUnit)``. Mutation operations on the C++
class (``+=``, ``-=``, ``*=``, ``normalize()``) become methods that
return new ``Period`` instances; classmethod ``from_frequency`` mirrors
the C++ ``explicit Period(Frequency)`` constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from pquantlib import qassert
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit

_DAYS_PER_WEEK: Final[int] = 7
_MONTHS_PER_YEAR: Final[int] = 12


def _unknown_unit(u: TimeUnit) -> NoReturn:
    qassert.fail(f"unknown time unit ({int(u)})")


@dataclass(frozen=True, slots=True)
class Period:
    """A signed integer length of one ``TimeUnit``."""

    length: int = 0
    units: TimeUnit = TimeUnit.Days

    # --- alternate constructor ------------------------------------------------

    @classmethod
    def from_frequency(cls, f: Frequency) -> Period:
        """Mirrors C++ ``explicit Period(Frequency)`` (ql/time/period.cpp)."""
        if f == Frequency.NoFrequency:
            return cls(0, TimeUnit.Days)
        if f == Frequency.Once:
            return cls(0, TimeUnit.Years)
        if f == Frequency.Annual:
            return cls(1, TimeUnit.Years)
        if f in (
            Frequency.Semiannual,
            Frequency.EveryFourthMonth,
            Frequency.Quarterly,
            Frequency.Bimonthly,
            Frequency.Monthly,
        ):
            return cls(_MONTHS_PER_YEAR // int(f), TimeUnit.Months)
        if f in (Frequency.EveryFourthWeek, Frequency.Biweekly, Frequency.Weekly):
            return cls(52 // int(f), TimeUnit.Weeks)
        if f == Frequency.Daily:
            return cls(1, TimeUnit.Days)
        if f == Frequency.OtherFrequency:
            qassert.fail("unknown frequency")
        qassert.fail(f"unknown frequency ({int(f)})")

    # --- inverse: Period → Frequency -----------------------------------------

    def frequency(self) -> Frequency:
        """Mirrors C++ ``Period::frequency() const``."""
        length = abs(self.length)
        if length == 0:
            if self.units == TimeUnit.Years:
                return Frequency.Once
            return Frequency.NoFrequency
        if self.units == TimeUnit.Years:
            return Frequency.Annual if length == 1 else Frequency.OtherFrequency
        if self.units == TimeUnit.Months:
            if _MONTHS_PER_YEAR % length == 0 and length <= _MONTHS_PER_YEAR:
                return Frequency(_MONTHS_PER_YEAR // length)
            return Frequency.OtherFrequency
        if self.units == TimeUnit.Weeks:
            if length == 1:
                return Frequency.Weekly
            if length == 2:
                return Frequency.Biweekly
            if length == 4:
                return Frequency.EveryFourthWeek
            return Frequency.OtherFrequency
        if self.units == TimeUnit.Days:
            return Frequency.Daily if length == 1 else Frequency.OtherFrequency
        _unknown_unit(self.units)

    # --- normalize -----------------------------------------------------------

    def normalized(self) -> Period:
        """Return the canonical-form Period (mirrors C++ ``Period::normalized``)."""
        if self.length == 0:
            return Period(0, TimeUnit.Days)
        if self.units == TimeUnit.Months and self.length % _MONTHS_PER_YEAR == 0:
            return Period(self.length // _MONTHS_PER_YEAR, TimeUnit.Years)
        if self.units == TimeUnit.Days and self.length % _DAYS_PER_WEEK == 0:
            return Period(self.length // _DAYS_PER_WEEK, TimeUnit.Weeks)
        if self.units in (TimeUnit.Weeks, TimeUnit.Years, TimeUnit.Months, TimeUnit.Days):
            return self
        _unknown_unit(self.units)

    # --- arithmetic ----------------------------------------------------------

    def __neg__(self) -> Period:
        return Period(-self.length, self.units)

    def __add__(self, other: Period) -> Period:
        if self.length == 0:
            return Period(other.length, other.units)
        if other.length == 0:
            return self
        if self.units == other.units:
            return Period(self.length + other.length, self.units)
        # Mixed-unit addition: only adjacent unit pairs are well-defined.
        a, b = self.units, other.units
        if a == TimeUnit.Years and b == TimeUnit.Months:
            return Period(self.length * _MONTHS_PER_YEAR + other.length, TimeUnit.Months)
        if a == TimeUnit.Months and b == TimeUnit.Years:
            return Period(self.length + other.length * _MONTHS_PER_YEAR, TimeUnit.Months)
        if a == TimeUnit.Weeks and b == TimeUnit.Days:
            return Period(self.length * _DAYS_PER_WEEK + other.length, TimeUnit.Days)
        if a == TimeUnit.Days and b == TimeUnit.Weeks:
            return Period(self.length + other.length * _DAYS_PER_WEEK, TimeUnit.Days)
        qassert.fail(f"impossible addition between {self} and {other}")

    def __sub__(self, other: Period) -> Period:
        return self + (-other)

    def __mul__(self, n: int) -> Period:
        return Period(self.length * n, self.units)

    def __rmul__(self, n: int) -> Period:
        return self.__mul__(n)

    def __floordiv__(self, n: int) -> Period:
        """Integer division mirroring C++ ``operator/=``.

        Keeps original units when ``length % n == 0``; otherwise converts to a
        finer unit (Years → Months, Weeks → Days) and retries.
        """
        qassert.require(n != 0, "cannot be divided by zero")
        if self.length % n == 0:
            return Period(self.length // n, self.units)
        units = self.units
        length = self.length
        if units == TimeUnit.Years:
            length *= _MONTHS_PER_YEAR
            units = TimeUnit.Months
        elif units == TimeUnit.Weeks:
            length *= _DAYS_PER_WEEK
            units = TimeUnit.Days
        qassert.require(length % n == 0, f"{self} cannot be divided by {n}")
        return Period(length // n, units)

    # --- ordering ------------------------------------------------------------

    def __lt__(self, other: Period) -> bool:
        # Special cases for zero-length periods
        if self.length == 0:
            return other.length > 0
        if other.length == 0:
            return self.length < 0

        # Exact comparisons (same units, or convertible Years/Months, Weeks/Days)
        if self.units == other.units:
            return self.length < other.length
        if self.units == TimeUnit.Months and other.units == TimeUnit.Years:
            return self.length < _MONTHS_PER_YEAR * other.length
        if self.units == TimeUnit.Years and other.units == TimeUnit.Months:
            return _MONTHS_PER_YEAR * self.length < other.length
        if self.units == TimeUnit.Days and other.units == TimeUnit.Weeks:
            return self.length < _DAYS_PER_WEEK * other.length
        if self.units == TimeUnit.Weeks and other.units == TimeUnit.Days:
            return _DAYS_PER_WEEK * self.length < other.length

        # Inexact comparisons (convert to days using min/max bounds)
        a_min, a_max = _days_min_max(self)
        b_min, b_max = _days_min_max(other)
        if a_max < b_min:
            return True
        if a_min > b_max:
            return False
        qassert.fail(f"undecidable comparison between {self} and {other}")

    def __le__(self, other: Period) -> bool:
        return self < other or self == other

    def __gt__(self, other: Period) -> bool:
        return other < self

    def __ge__(self, other: Period) -> bool:
        return other < self or self == other


# --- free-function extractors -----------------------------------------------


def years(p: Period) -> float:
    """Length of ``p`` expressed in years (mirrors C++ free function ``years``)."""
    if p.length == 0:
        return 0.0
    if p.units == TimeUnit.Days:
        qassert.fail("cannot convert Days into Years")
    if p.units == TimeUnit.Weeks:
        qassert.fail("cannot convert Weeks into Years")
    if p.units == TimeUnit.Months:
        return p.length / 12.0
    if p.units == TimeUnit.Years:
        return float(p.length)
    _unknown_unit(p.units)


def months(p: Period) -> float:
    if p.length == 0:
        return 0.0
    if p.units == TimeUnit.Days:
        qassert.fail("cannot convert Days into Months")
    if p.units == TimeUnit.Weeks:
        qassert.fail("cannot convert Weeks into Months")
    if p.units == TimeUnit.Months:
        return float(p.length)
    if p.units == TimeUnit.Years:
        return p.length * 12.0
    _unknown_unit(p.units)


def weeks(p: Period) -> float:
    if p.length == 0:
        return 0.0
    if p.units == TimeUnit.Days:
        return p.length / 7.0
    if p.units == TimeUnit.Weeks:
        return float(p.length)
    if p.units == TimeUnit.Months:
        qassert.fail("cannot convert Months into Weeks")
    if p.units == TimeUnit.Years:
        qassert.fail("cannot convert Years into Weeks")
    _unknown_unit(p.units)


def days(p: Period) -> float:
    if p.length == 0:
        return 0.0
    if p.units == TimeUnit.Days:
        return float(p.length)
    if p.units == TimeUnit.Weeks:
        return p.length * 7.0
    if p.units == TimeUnit.Months:
        qassert.fail("cannot convert Months into Days")
    if p.units == TimeUnit.Years:
        qassert.fail("cannot convert Years into Days")
    _unknown_unit(p.units)


def _days_min_max(p: Period) -> tuple[int, int]:
    """Mirrors the anonymous-namespace ``daysMinMax`` helper in period.cpp."""
    if p.units == TimeUnit.Days:
        return p.length, p.length
    if p.units == TimeUnit.Weeks:
        return 7 * p.length, 7 * p.length
    if p.units == TimeUnit.Months:
        return 28 * p.length, 31 * p.length
    if p.units == TimeUnit.Years:
        return 365 * p.length, 366 * p.length
    _unknown_unit(p.units)
