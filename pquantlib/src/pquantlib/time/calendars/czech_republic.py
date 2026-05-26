"""CzechRepublic — Czech calendars (Prague stock exchange).

# C++ parity: ql/time/calendars/czechrepublic.hpp + czechrepublic.cpp (v1.42.1).

Holidays for the Prague stock exchange:
- Saturdays
- Sundays
- New Year's Day, January 1st
- Good Friday (since 2016)
- Easter Monday
- Labour Day, May 1st
- Liberation Day, May 8th
- SS. Cyril and Methodius, July 5th
- Jan Hus Day, July 6th
- Czech Statehood Day, September 28th
- Independence Day, October 28th
- Struggle for Freedom and Democracy Day, November 17th
- Christmas Eve, December 24th
- Christmas, December 25th
- St. Stephen, December 26th
- Unidentified stock-exchange closing days: 2004-01-02, 2004-12-31
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Market(IntEnum):
    PSE = 0  # Prague stock exchange


class CzechRepublic(WesternCalendar):
    def __init__(self, market: Market = Market.PSE) -> None:
        super().__init__()
        self._market: Market = market

    def name(self) -> str:
        return "Prague stock exchange"

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
            # Good Friday (since 2016)
            or (dy == em - 3 and y >= 2016)
            # Easter Monday
            or (dy == em)
            # Labour Day
            or (dom == 1 and m == Month.May)
            # Liberation Day
            or (dom == 8 and m == Month.May)
            # SS. Cyril and Methodius
            or (dom == 5 and m == Month.July)
            # Jan Hus Day
            or (dom == 6 and m == Month.July)
            # Czech Statehood Day
            or (dom == 28 and m == Month.September)
            # Independence Day
            or (dom == 28 and m == Month.October)
            # Struggle for Freedom and Democracy Day
            or (dom == 17 and m == Month.November)
            # Christmas Eve
            or (dom == 24 and m == Month.December)
            # Christmas
            or (dom == 25 and m == Month.December)
            # St. Stephen
            or (dom == 26 and m == Month.December)
            # unidentified closing days for stock exchange
            or (dom == 2 and m == Month.January and y == 2004)
            or (dom == 31 and m == Month.December and y == 2004)
        )
