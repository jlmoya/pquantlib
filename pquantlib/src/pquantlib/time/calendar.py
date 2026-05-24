"""Calendar abstract base class + Western/Orthodox shared bases.

# C++ parity: ql/time/calendar.hpp + ql/time/calendar.cpp (v1.42.1).

The C++ design uses a pImpl Bridge pattern (``Calendar`` holds a
``shared_ptr<Impl>`` whose ``isBusinessDay`` / ``isWeekend`` / ``name`` are
virtual). The Python port collapses this into direct ``abc.ABC`` inheritance:

- ``Calendar`` is the abstract base. Concretes implement ``name()``,
  ``_is_business_day(date)``, and ``_is_weekend(weekday)``.
- ``WesternCalendar`` is an abstract subclass that fixes Sat+Sun as
  weekend and exposes ``easter_monday(year)`` (day-of-year offset).
- ``OrthodoxCalendar`` is the same but uses the Orthodox Easter table.

Per-instance ``added_holidays`` and ``removed_holidays`` are stored
directly on the Python instance. C++ shares them via the Impl pointer;
Python's per-instance model is simpler and matches typical usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from pquantlib import qassert
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib.time.weekday import Weekday


class Calendar(ABC):
    """Abstract base class for market calendars.

    Concretes implement ``name()``, ``_is_business_day(date)``, and
    ``_is_weekend(weekday)``. Public ``is_business_day`` consults
    per-instance added / removed holiday sets before delegating, mirroring
    C++ ``Calendar::isBusinessDay``.
    """

    def __init__(self) -> None:
        self._added_holidays: set[Date] = set()
        self._removed_holidays: set[Date] = set()

    # --- subclass-implemented hooks ----------------------------------------

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _is_business_day(self, d: Date) -> bool: ...

    @abstractmethod
    def _is_weekend(self, w: Weekday) -> bool: ...

    # --- public API --------------------------------------------------------

    def is_business_day(self, d: Date) -> bool:
        if d in self._added_holidays:
            return False
        if d in self._removed_holidays:
            return True
        return self._is_business_day(d)

    def is_holiday(self, d: Date) -> bool:
        return not self.is_business_day(d)

    def is_weekend(self, w: Weekday) -> bool:
        return self._is_weekend(w)

    @property
    def added_holidays(self) -> frozenset[Date]:
        return frozenset(self._added_holidays)

    @property
    def removed_holidays(self) -> frozenset[Date]:
        return frozenset(self._removed_holidays)

    def add_holiday(self, d: Date) -> None:
        self._removed_holidays.discard(d)
        if self._is_business_day(d):
            self._added_holidays.add(d)

    def remove_holiday(self, d: Date) -> None:
        self._added_holidays.discard(d)
        if not self._is_business_day(d):
            self._removed_holidays.add(d)

    def reset_added_and_removed_holidays(self) -> None:
        self._added_holidays.clear()
        self._removed_holidays.clear()

    # --- start_of_month / end_of_month -------------------------------------

    def start_of_month(self, d: Date) -> Date:
        return self.adjust(Date.start_of_month(d), BusinessDayConvention.Following)

    def is_start_of_month(self, d: Date) -> bool:
        return d <= self.start_of_month(d)

    def end_of_month(self, d: Date) -> Date:
        return self.adjust(Date.end_of_month(d), BusinessDayConvention.Preceding)

    def is_end_of_month(self, d: Date) -> bool:
        return d >= self.end_of_month(d)

    # --- adjust ------------------------------------------------------------

    def adjust(
        self,
        d: Date,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
    ) -> Date:
        qassert.require(d != Date(), "null date")

        if convention == BusinessDayConvention.Unadjusted:
            return d

        c = convention
        if c in (
            BusinessDayConvention.Following,
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.HalfMonthModifiedFollowing,
        ):
            d1 = d
            while self.is_holiday(d1):
                d1 = d1 + 1
            if c in (
                BusinessDayConvention.ModifiedFollowing,
                BusinessDayConvention.HalfMonthModifiedFollowing,
            ):
                if d1.month() != d.month():
                    return self.adjust(d, BusinessDayConvention.Preceding)
                if (
                    c == BusinessDayConvention.HalfMonthModifiedFollowing
                    and d.day_of_month() <= 15
                    and d1.day_of_month() > 15
                ):
                    return self.adjust(d, BusinessDayConvention.Preceding)
            return d1

        if c in (BusinessDayConvention.Preceding, BusinessDayConvention.ModifiedPreceding):
            d1 = d
            while self.is_holiday(d1):
                d1 = d1 - 1
            if c == BusinessDayConvention.ModifiedPreceding and d1.month() != d.month():
                return self.adjust(d, BusinessDayConvention.Following)
            return d1

        if c == BusinessDayConvention.Nearest:
            d1 = d
            d2 = d
            while self.is_holiday(d1) and self.is_holiday(d2):
                d1 = d1 + 1
                d2 = d2 - 1
            return d2 if self.is_holiday(d1) else d1

        qassert.fail("unknown business-day convention")

    # --- advance -----------------------------------------------------------

    def advance(
        self,
        d: Date,
        n: int,
        unit: TimeUnit,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
        end_of_month: bool = False,
    ) -> Date:
        qassert.require(d != Date(), "null date")
        if n == 0:
            return self.adjust(d, convention)
        if unit == TimeUnit.Days:
            d1 = d
            if n > 0:
                while n > 0:
                    d1 = d1 + 1
                    while self.is_holiday(d1):
                        d1 = d1 + 1
                    n -= 1
            else:
                while n < 0:
                    d1 = d1 - 1
                    while self.is_holiday(d1):
                        d1 = d1 - 1
                    n += 1
            return d1
        if unit == TimeUnit.Weeks:
            d1 = d + Period(n, unit)
            return self.adjust(d1, convention)
        # Months or Years
        d1 = d + Period(n, unit)
        if end_of_month:
            if convention == BusinessDayConvention.Unadjusted:
                if Date.is_end_of_month(d):
                    return Date.end_of_month(d1)
            elif self.is_end_of_month(d):
                return self.end_of_month(d1)
        return self.adjust(d1, convention)

    def advance_period(
        self,
        d: Date,
        period: Period,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
        end_of_month: bool = False,
    ) -> Date:
        return self.advance(d, period.length, period.units, convention, end_of_month)

    # --- business_days_between --------------------------------------------

    def _days_between_impl(self, frm: Date, to: Date, include_first: bool, include_last: bool) -> int:
        """Mirror C++ anonymous-namespace ``daysBetweenImpl``. Requires ``frm < to``."""
        res = int(include_last and self.is_business_day(to))
        d = frm if include_first else frm + 1
        while d < to:
            if self.is_business_day(d):
                res += 1
            d = d + 1
        return res

    def business_days_between(
        self,
        frm: Date,
        to: Date,
        include_first: bool = True,
        include_last: bool = False,
    ) -> int:
        if frm < to:
            return self._days_between_impl(frm, to, include_first, include_last)
        if frm > to:
            return -self._days_between_impl(to, frm, include_last, include_first)
        return int(include_first and include_last and self.is_business_day(frm))

    # --- list builders -----------------------------------------------------

    def holiday_list(self, frm: Date, to: Date, include_weekends: bool = False) -> tuple[Date, ...]:
        out: list[Date] = []
        d = frm
        while d <= to:
            if self.is_holiday(d) and (include_weekends or not self.is_weekend(d.weekday())):
                out.append(d)
            d = d + 1
        return tuple(out)

    def business_day_list(self, frm: Date, to: Date) -> tuple[Date, ...]:
        out: list[Date] = []
        d = frm
        while d <= to:
            if self.is_business_day(d):
                out.append(d)
            d = d + 1
        return tuple(out)

    # --- equality + repr ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calendar):
            return NotImplemented
        return self.name() == other.name()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.name())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name()!r})"

    def __str__(self) -> str:
        return self.name()


# --- Western / Orthodox shared bases ---------------------------------------


class WesternCalendar(Calendar):
    """Abstract base for Western-style calendars (Sat+Sun weekend, Easter Monday)."""

    def _is_weekend(self, w: Weekday) -> bool:
        return w in (Weekday.Saturday, Weekday.Sunday)

    @staticmethod
    def easter_monday(year: int) -> int:
        """Day-of-year offset of Easter Monday in ``year`` (Western)."""
        return _WESTERN_EASTER_MONDAY[year - _EASTER_BASE_YEAR]


class OrthodoxCalendar(Calendar):
    """Abstract base for Orthodox-style calendars (Sat+Sun weekend, Orthodox Easter Monday)."""

    def _is_weekend(self, w: Weekday) -> bool:
        return w in (Weekday.Saturday, Weekday.Sunday)

    @staticmethod
    def easter_monday(year: int) -> int:
        return _ORTHODOX_EASTER_MONDAY[year - _EASTER_BASE_YEAR]


_EASTER_BASE_YEAR: Final[int] = 1901

# C++ parity: ql/time/calendar.cpp ``Calendar::WesternImpl::easterMonday``
# table, 1901..2199 (299 entries). Each entry is day-of-year of Easter Monday.
_WESTERN_EASTER_MONDAY: Final[tuple[int, ...]] = (
    98,
    90,
    103,
    95,
    114,
    106,
    91,
    111,
    102,
    87,
    107,
    99,
    83,
    103,
    95,
    115,
    99,
    91,
    111,
    96,
    87,
    107,
    92,
    112,
    103,
    95,
    108,
    100,
    91,
    111,
    96,
    88,
    107,
    92,
    112,
    104,
    88,
    108,
    100,
    85,
    104,
    96,
    116,
    101,
    92,
    112,
    97,
    89,
    108,
    100,
    85,
    105,
    96,
    109,
    101,
    93,
    112,
    97,
    89,
    109,
    93,
    113,
    105,
    90,
    109,
    101,
    86,
    106,
    97,
    89,
    102,
    94,
    113,
    105,
    90,
    110,
    101,
    86,
    106,
    98,
    110,
    102,
    94,
    114,
    98,
    90,
    110,
    95,
    86,
    106,
    91,
    111,
    102,
    94,
    107,
    99,
    90,
    103,
    95,
    115,
    106,
    91,
    111,
    103,
    87,
    107,
    99,
    84,
    103,
    95,
    115,
    100,
    91,
    111,
    96,
    88,
    107,
    92,
    112,
    104,
    95,
    108,
    100,
    92,
    111,
    96,
    88,
    108,
    92,
    112,
    104,
    89,
    108,
    100,
    85,
    105,
    96,
    116,
    101,
    93,
    112,
    97,
    89,
    109,
    100,
    85,
    105,
    97,
    109,
    101,
    93,
    113,
    97,
    89,
    109,
    94,
    113,
    105,
    90,
    110,
    101,
    86,
    106,
    98,
    89,
    102,
    94,
    114,
    105,
    90,
    110,
    102,
    86,
    106,
    98,
    111,
    102,
    94,
    114,
    99,
    90,
    110,
    95,
    87,
    106,
    91,
    111,
    103,
    94,
    107,
    99,
    91,
    103,
    95,
    115,
    107,
    91,
    111,
    103,
    88,
    108,
    100,
    85,
    105,
    96,
    109,
    101,
    93,
    112,
    97,
    89,
    109,
    93,
    113,
    105,
    90,
    109,
    101,
    86,
    106,
    97,
    89,
    102,
    94,
    113,
    105,
    90,
    110,
    101,
    86,
    106,
    98,
    110,
    102,
    94,
    114,
    98,
    90,
    110,
    95,
    86,
    106,
    91,
    111,
    102,
    94,
    107,
    99,
    90,
    103,
    95,
    115,
    106,
    91,
    111,
    103,
    87,
    107,
    99,
    84,
    103,
    95,
    115,
    100,
    91,
    111,
    96,
    88,
    107,
    92,
    112,
    104,
    95,
    108,
    100,
    92,
    111,
    96,
    88,
    108,
    92,
    112,
    104,
    89,
    108,
    100,
    85,
    105,
    96,
    116,
    101,
    93,
    112,
    97,
    89,
    109,
    100,
    85,
    105,
)

# C++ parity: ql/time/calendar.cpp ``Calendar::OrthodoxImpl::easterMonday``
# table, 1901..2199.
_ORTHODOX_EASTER_MONDAY: Final[tuple[int, ...]] = (
    105,
    118,
    110,
    102,
    121,
    106,
    126,
    118,
    102,
    122,
    114,
    99,
    118,
    110,
    95,
    115,
    106,
    126,
    111,
    103,
    122,
    107,
    99,
    119,
    110,
    123,
    115,
    107,
    126,
    111,
    103,
    123,
    107,
    99,
    119,
    104,
    123,
    115,
    100,
    120,
    111,
    96,
    116,
    108,
    127,
    112,
    104,
    124,
    115,
    100,
    120,
    112,
    96,
    116,
    108,
    128,
    112,
    104,
    124,
    109,
    100,
    120,
    105,
    125,
    116,
    101,
    121,
    113,
    104,
    117,
    109,
    101,
    120,
    105,
    125,
    117,
    101,
    121,
    113,
    98,
    117,
    109,
    129,
    114,
    105,
    125,
    110,
    102,
    121,
    106,
    98,
    118,
    109,
    122,
    114,
    106,
    118,
    110,
    102,
    122,
    106,
    126,
    118,
    103,
    122,
    114,
    99,
    119,
    110,
    95,
    115,
    107,
    126,
    111,
    103,
    123,
    107,
    99,
    119,
    111,
    123,
    115,
    107,
    127,
    111,
    103,
    123,
    108,
    99,
    119,
    104,
    124,
    115,
    100,
    120,
    112,
    96,
    116,
    108,
    128,
    112,
    104,
    124,
    116,
    100,
    120,
    112,
    97,
    116,
    108,
    128,
    113,
    104,
    124,
    109,
    101,
    120,
    105,
    125,
    117,
    101,
    121,
    113,
    105,
    117,
    109,
    101,
    121,
    105,
    125,
    110,
    102,
    121,
    113,
    98,
    118,
    109,
    129,
    114,
    106,
    125,
    110,
    102,
    122,
    106,
    98,
    118,
    110,
    122,
    114,
    99,
    119,
    110,
    102,
    115,
    107,
    126,
    118,
    103,
    123,
    115,
    100,
    120,
    112,
    96,
    116,
    108,
    128,
    112,
    104,
    124,
    109,
    100,
    120,
    105,
    125,
    116,
    108,
    121,
    113,
    104,
    124,
    109,
    101,
    120,
    105,
    125,
    117,
    101,
    121,
    113,
    98,
    117,
    109,
    129,
    114,
    105,
    125,
    110,
    102,
    121,
    113,
    98,
    118,
    109,
    129,
    114,
    106,
    125,
    110,
    102,
    122,
    106,
    126,
    118,
    103,
    122,
    114,
    99,
    119,
    110,
    102,
    115,
    107,
    126,
    111,
    103,
    123,
    114,
    99,
    119,
    111,
    130,
    115,
    107,
    127,
    111,
    103,
    123,
    108,
    99,
    119,
    104,
    124,
    115,
    100,
    120,
    112,
    103,
    116,
    108,
    128,
    119,
    104,
    124,
    116,
    100,
    120,
    112,
)

# Sanity: the C++ tables are length-299 (1901..2199 inclusive).
assert len(_WESTERN_EASTER_MONDAY) == 299, "Western Easter table length mismatch"
assert len(_ORTHODOX_EASTER_MONDAY) == 299, "Orthodox Easter table length mismatch"


# Re-export `Month` so calendar implementations don't all need to import it
# from time.month directly when extending this module.
__all__ = ["Calendar", "Month", "OrthodoxCalendar", "WesternCalendar"]
