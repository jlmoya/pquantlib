"""Iceland — Icelandic calendars (Iceland stock exchange).

# C++ parity: ql/time/calendars/iceland.hpp + iceland.cpp (v1.42.1).

Holidays for the Iceland stock exchange:
- Saturdays
- Sundays
- New Year's Day, January 1st
- Holy Thursday
- Good Friday
- Easter Monday
- First day of Summer (third or fourth Thursday in April)
- Labour Day, May 1st
- Ascension Thursday
- Pentecost Monday
- Independence Day, June 17th
- Commerce Day, first Monday in August
- Christmas, December 25th
- Boxing Day, December 26th
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Market(IntEnum):
    ICEX = 0  # Iceland stock exchange


class Iceland(WesternCalendar):
    def __init__(self, market: Market = Market.ICEX) -> None:
        super().__init__()
        self._market: Market = market

    def name(self) -> str:
        return "Iceland stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year's Day
            or (dom == 1 and m == Month.January)
            # Holy Thursday
            or (dy == em - 4)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # First day of Summer
            or (19 <= dom <= 25 and w == Weekday.Thursday and m == Month.April)
            # Ascension Thursday
            or (dy == em + 38)
            # Pentecost Monday
            or (dy == em + 49)
            # Labour Day
            or (dom == 1 and m == Month.May)
            # Independence Day
            or (dom == 17 and m == Month.June)
            # Commerce Day
            or (dom <= 7 and w == Weekday.Monday and m == Month.August)
            # Christmas
            or (dom == 25 and m == Month.December)
            # Boxing Day
            or (dom == 26 and m == Month.December)
        )
