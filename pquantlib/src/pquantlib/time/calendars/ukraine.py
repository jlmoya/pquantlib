"""Ukraine — Ukrainian calendars (Ukrainian stock exchange).

# C++ parity: ql/time/calendars/ukraine.hpp + ukraine.cpp (v1.42.1).

# C++ parity: Ukraine extends ``Calendar::OrthodoxImpl`` (orthodox Easter)
# rather than ``WesternImpl``. The Python port subclasses
# ``OrthodoxCalendar`` to inherit the correct Easter Monday table.

Holidays for the Ukrainian stock exchange:
- Saturdays
- Sundays
- New Year's Day, January 1st (possibly moved to Monday)
- Orthodox Christmas, January 7th (possibly moved to Monday)
- International Women's Day, March 8th (possibly moved to Monday)
- Easter Monday (Orthodox)
- Holy Trinity Day, 50 days after Easter
- International Workers' Solidarity Days, May 1st and 2nd
- Victory Day, May 9th
- Constitution Day, June 28th
- Independence Day, August 24th
- Defender's Day, October 14th (since 2015)
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import OrthodoxCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Market(IntEnum):
    USE = 0  # Ukrainian stock exchange


class Ukraine(OrthodoxCalendar):
    def __init__(self, market: Market = Market.USE) -> None:
        super().__init__()
        self._market: Market = market

    def name(self) -> str:
        return "Ukrainian stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = OrthodoxCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year's Day (possibly moved to Monday)
            or ((dom == 1 or (dom in {2, 3} and w == Weekday.Monday)) and m == Month.January)
            # Orthodox Christmas
            or ((dom == 7 or (dom in {8, 9} and w == Weekday.Monday)) and m == Month.January)
            # Women's Day
            or ((dom == 8 or (dom in {9, 10} and w == Weekday.Monday)) and m == Month.March)
            # Orthodox Easter Monday
            or (dy == em)
            # Holy Trinity Day
            or (dy == em + 49)
            # Workers' Solidarity Days
            or ((dom in {1, 2} or (dom == 3 and w == Weekday.Monday)) and m == Month.May)
            # Victory Day
            or ((dom == 9 or (dom in {10, 11} and w == Weekday.Monday)) and m == Month.May)
            # Constitution Day
            or (dom == 28 and m == Month.June)
            # Independence Day
            or (dom == 24 and m == Month.August)
            # Defender's Day (since 2015)
            or (dom == 14 and m == Month.October and y >= 2015)
        )
