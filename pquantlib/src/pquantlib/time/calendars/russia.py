"""Russia calendars — Settlement + MOEX (Moscow Exchange).

# C++ parity: ql/time/calendars/russia.hpp + .cpp (v1.42.1).

Both impls inherit ``Calendar::OrthodoxImpl`` in C++ (Sat+Sun weekend +
Orthodox Easter table) — Python mirrors this by deriving each impl from
``OrthodoxCalendar``. The default ``Russia()`` (no args) selects the
Settlement market, which is the variant referenced by
``time/calendars/all`` -> "russia" (name: "Russian settlement").
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.time.calendar import Calendar, OrthodoxCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class RussiaMarket(IntEnum):
    Settlement = 0  # generic settlement calendar
    MOEX = 1  # Moscow Exchange calendar


def _is_extra_holiday_settlement_impl(day: int, month: Month, year: int) -> bool:
    # C++ parity: ``isExtraHolidaySettlementImpl`` anonymous namespace.
    if year == 2017:
        if month == Month.February:
            return day == 24
        if month == Month.May:
            return day == 8
        if month == Month.November:
            return day == 6
        return False
    if year == 2018:
        if month == Month.March:
            return day == 9
        if month == Month.April:
            return day == 30
        if month == Month.May:
            return day == 2
        if month == Month.June:
            return day == 11
        if month == Month.December:
            return day == 31
        return False
    if year == 2019:
        if month == Month.May:
            return day in (2, 3, 10)
        return False
    if year == 2020:
        if month == Month.March:
            return day in (30, 31)
        if month == Month.April:
            return day in (1, 2, 3)
        if month == Month.May:
            return day in (4, 5)
        return False
    return False


def _is_working_weekend(day: int, month: Month, year: int) -> bool:
    # C++ parity: ``isWorkingWeekend`` anonymous namespace.
    if year == 2012:
        if month == Month.March:
            return day == 11
        if month == Month.April:
            return day == 28
        if month == Month.May:
            return day in (5, 12)
        if month == Month.June:
            return day == 9
        return False
    if year == 2016:
        if month == Month.February:
            return day == 20
        return False
    if year == 2018:
        if month == Month.April:
            return day == 28
        if month == Month.June:
            return day == 9
        if month == Month.December:
            return day == 29
        return False
    return False


def _is_extra_holiday_exchange_impl(day: int, month: Month, year: int) -> bool:
    # C++ parity: ``isExtraHolidayExchangeImpl`` anonymous namespace.
    if year == 2012:
        if month == Month.January:
            return day == 2
        if month == Month.March:
            return day == 9
        if month == Month.April:
            return day == 30
        if month == Month.June:
            return day == 11
        return False
    if year == 2013:
        if month == Month.January:
            return day in (1, 2, 3, 4, 7)
        return False
    if year == 2014:
        if month == Month.January:
            return day in (1, 2, 3, 7)
        return False
    if year == 2015:
        if month == Month.January:
            return day in (1, 2, 7)
        return False
    if year == 2016:
        if month == Month.January:
            return day in (1, 7, 8)
        if month == Month.May:
            return day in (2, 3)
        if month == Month.June:
            return day == 13
        if month == Month.December:
            return day == 30
        return False
    if year == 2017:
        if month == Month.January:
            return day == 2
        if month == Month.May:
            return day == 8
        return False
    if year == 2018:
        if month == Month.January:
            return day in (1, 2, 8)
        if month == Month.December:
            return day == 31
        return False
    if year == 2019:
        if month == Month.January:
            return day in (1, 2, 7)
        if month == Month.December:
            return day == 31
        return False
    if year == 2020:
        if month == Month.January:
            return day in (1, 2, 7)
        if month == Month.February:
            return day == 24
        if month == Month.June:
            return day == 24
        if month == Month.July:
            return day == 1
        return False
    return False


class _RussiaSettlementCalendar(OrthodoxCalendar):
    """Russian settlement calendar (Orthodox base)."""

    def name(self) -> str:
        return "Russian settlement"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        y = d.year()

        if (
            self._is_weekend(w)
            # New Year's holidays
            or (y <= 2005 and day <= 2 and m == Month.January)
            or (y >= 2005 and day <= 5 and m == Month.January)
            # in 2012, the 6th was also a holiday
            or (y == 2012 and day == 6 and m == Month.January)
            # Christmas (possibly moved to Monday)
            or ((day == 7 or (day in {8, 9} and w == Weekday.Monday)) and m == Month.January)
            # Defender of the Fatherland Day (possibly moved to Monday)
            or ((day == 23 or (day in {24, 25} and w == Weekday.Monday)) and m == Month.February)
            # International Women's Day (possibly moved to Monday)
            or ((day == 8 or (day in {9, 10} and w == Weekday.Monday)) and m == Month.March)
            # Labour Day (possibly moved to Monday)
            or ((day == 1 or (day in {2, 3} and w == Weekday.Monday)) and m == Month.May)
            # Victory Day (possibly moved to Monday)
            or ((day == 9 or (day in {10, 11} and w == Weekday.Monday)) and m == Month.May)
            # Russia Day (possibly moved to Monday)
            or ((day == 12 or (day in {13, 14} and w == Weekday.Monday)) and m == Month.June)
            # Unity Day (possibly moved to Monday)
            or ((day == 4 or (day in {5, 6} and w == Weekday.Monday)) and m == Month.November)
        ):
            return False

        return not _is_extra_holiday_settlement_impl(day, m, y)


class _RussiaExchangeCalendar(OrthodoxCalendar):
    """Moscow Exchange (MOEX) calendar (Orthodox base)."""

    def name(self) -> str:
        return "Moscow exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        y = d.year()

        # MOEX brand only exists from 2012 onwards
        if y < 2012:
            raise LibraryException(f"MOEX calendar for the year {y} does not exist.")

        if _is_working_weekend(day, m, y):
            return True

        if (
            self._is_weekend(w)
            # Defender of the Fatherland Day
            or (day == 23 and m == Month.February)
            # International Women's Day (possibly moved to Monday)
            or ((day == 8 or (day in {9, 10} and w == Weekday.Monday)) and m == Month.March)
            # Labour Day
            or (day == 1 and m == Month.May)
            # Victory Day (possibly moved to Monday)
            or ((day == 9 or (day in {10, 11} and w == Weekday.Monday)) and m == Month.May)
            # Russia Day
            or (day == 12 and m == Month.June)
            # Unity Day (possibly moved to Monday)
            or ((day == 4 or (day in {5, 6} and w == Weekday.Monday)) and m == Month.November)
            # New Years Eve
            or (day == 31 and m == Month.December)
        ):
            return False

        return not _is_extra_holiday_exchange_impl(day, m, y)


class Russia(Calendar):
    """Russia calendar dispatching on market (Settlement / MOEX)."""

    def __init__(self, market: RussiaMarket = RussiaMarket.Settlement) -> None:
        super().__init__()
        if market == RussiaMarket.Settlement:
            self._impl: Calendar = _RussiaSettlementCalendar()
        elif market == RussiaMarket.MOEX:
            self._impl = _RussiaExchangeCalendar()
        else:
            qassert.fail("unknown market")
        self._market: RussiaMarket = market

    def name(self) -> str:
        return self._impl.name()

    def _is_weekend(self, w: Weekday) -> bool:
        return self._impl._is_weekend(w)

    def _is_business_day(self, d: Date) -> bool:
        return self._impl._is_business_day(d)
