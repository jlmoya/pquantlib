"""Canadian calendar — Settlement / TSX (Toronto Stock Exchange) markets.

# C++ parity: ql/time/calendars/canada.hpp + canada.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Canada(WesternCalendar):
    """Canadian calendar (Settlement default, or TSX exchange)."""

    class Market(IntEnum):
        Settlement = 0  # generic settlement calendar
        TSX = 1  # Toronto stock exchange calendar

    def __init__(self, market: Market = Market.Settlement) -> None:
        super().__init__()
        self._market = market

    def name(self) -> str:
        if self._market == Canada.Market.Settlement:
            return "Canada"
        if self._market == Canada.Market.TSX:
            return "TSX"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == Canada.Market.Settlement:
            return self._is_business_day_settlement(d)
        if self._market == Canada.Market.TSX:
            return self._is_business_day_tsx(d)
        qassert.fail("unknown market")

    def _is_business_day_settlement(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            # New Year's Day (possibly moved to Monday)
            ((day == 1 or (day in (2, 3) and w == Weekday.Monday)) and m == Month.January)
            # Family Day (third Monday in February, since 2008)
            or (15 <= day <= 21 and w == Weekday.Monday and m == Month.February and y >= 2008)
            # Good Friday
            or (dd == em - 3)
            # Victoria Day: the Monday on or preceding 24 May
            or (17 < day <= 24 and w == Weekday.Monday and m == Month.May)
            # July 1st, possibly moved to Monday (Canada Day)
            or ((day == 1 or (day in (2, 3) and w == Weekday.Monday)) and m == Month.July)
            # first Monday of August (Provincial Holiday)
            or (day <= 7 and w == Weekday.Monday and m == Month.August)
            # first Monday of September (Labor Day)
            or (day <= 7 and w == Weekday.Monday and m == Month.September)
            # September 30th, possibly moved to Monday
            # (National Day for Truth and Reconciliation, since 2021)
            or (
                (
                    (day == 30 and m == Month.September)
                    or (day <= 2 and m == Month.October and w == Weekday.Monday)
                )
                and y >= 2021
            )
            # second Monday of October (Thanksgiving Day)
            or (7 < day <= 14 and w == Weekday.Monday and m == Month.October)
            # November 11th (possibly moved to Monday)
            or ((day == 11 or (day in (12, 13) and w == Weekday.Monday)) and m == Month.November)
            # Christmas (possibly moved to Monday or Tuesday)
            or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # Boxing Day (possibly moved to Monday or Tuesday)
            or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
        )

    def _is_business_day_tsx(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            # New Year's Day (possibly moved to Monday)
            ((day == 1 or (day in (2, 3) and w == Weekday.Monday)) and m == Month.January)
            # Family Day (third Monday in February, since 2008)
            or (15 <= day <= 21 and w == Weekday.Monday and m == Month.February and y >= 2008)
            # Good Friday
            or (dd == em - 3)
            # Victoria Day: the Monday on or preceding 24 May
            or (17 < day <= 24 and w == Weekday.Monday and m == Month.May)
            # July 1st, possibly moved to Monday (Canada Day)
            or ((day == 1 or (day in (2, 3) and w == Weekday.Monday)) and m == Month.July)
            # first Monday of August (Provincial Holiday)
            or (day <= 7 and w == Weekday.Monday and m == Month.August)
            # first Monday of September (Labor Day)
            or (day <= 7 and w == Weekday.Monday and m == Month.September)
            # second Monday of October (Thanksgiving Day)
            or (7 < day <= 14 and w == Weekday.Monday and m == Month.October)
            # Christmas (possibly moved to Monday or Tuesday)
            or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # Boxing Day (possibly moved to Monday or Tuesday)
            or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
        )
