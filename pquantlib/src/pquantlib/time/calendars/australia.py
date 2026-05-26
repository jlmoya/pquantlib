"""Australian calendar — Settlement / ASX markets.

# C++ parity: ql/time/calendars/australia.hpp + australia.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Australia(WesternCalendar):
    """Australian calendar (Settlement default, or ASX exchange)."""

    class Market(IntEnum):
        Settlement = 0  # generic settlement calendar
        ASX = 1  # Australia ASX calendar

    def __init__(self, market: Market = Market.Settlement) -> None:
        super().__init__()
        self._market = market

    def name(self) -> str:
        if self._market == Australia.Market.Settlement:
            return "Australia settlement"
        if self._market == Australia.Market.ASX:
            return "Australia exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == Australia.Market.Settlement:
            return self._is_business_day_settlement(d)
        if self._market == Australia.Market.ASX:
            return self._is_business_day_asx(d)
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
            # Australia Day, January 26th (possibly moved to Monday)
            or ((day == 26 or (day in (27, 28) and w == Weekday.Monday)) and m == Month.January)
            # Good Friday
            or (dd == em - 3)
            # Easter Monday
            or (dd == em)
            # ANZAC Day, April 25th
            or (day == 25 and m == Month.April)
            # Queen's Birthday, second Monday in June
            or (7 < day <= 14 and w == Weekday.Monday and m == Month.June)
            # Bank Holiday, first Monday in August
            or (day <= 7 and w == Weekday.Monday and m == Month.August)
            # Labour Day, first Monday in October
            or (day <= 7 and w == Weekday.Monday and m == Month.October)
            # Christmas, December 25th (possibly Monday or Tuesday)
            or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # Boxing Day, December 26th (possibly Monday or Tuesday)
            or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # National Day of Mourning for Her Majesty, September 22 (only 2022)
            or (day == 22 and m == Month.September and y == 2022)
        )

    def _is_business_day_asx(self, d: Date) -> bool:
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
            # Australia Day, January 26th (possibly moved to Monday)
            or ((day == 26 or (day in (27, 28) and w == Weekday.Monday)) and m == Month.January)
            # Good Friday
            or (dd == em - 3)
            # Easter Monday
            or (dd == em)
            # ANZAC Day, April 25th
            or (day == 25 and m == Month.April)
            # Queen's Birthday, second Monday in June
            or (7 < day <= 14 and w == Weekday.Monday and m == Month.June)
            # Christmas, December 25th (possibly Monday or Tuesday)
            or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # Boxing Day, December 26th (possibly Monday or Tuesday)
            or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # National Day of Mourning for Her Majesty, September 22 (only 2022)
            or (day == 22 and m == Month.September and y == 2022)
        )
