"""Switzerland — Swiss calendar.

# C++ parity: ql/time/calendars/switzerland.hpp + switzerland.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- New Year's Day, January 1st
- Berchtoldstag, January 2nd
- Good Friday
- Easter Monday
- Ascension Day
- Whit Monday
- Labour Day, May 1st
- National Day, August 1st
- Christmas, December 25th
- St. Stephen's Day, December 26th
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Switzerland(WesternCalendar):
    def name(self) -> str:
        return "Switzerland"

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
            # Berchtoldstag
            or (dom == 2 and m == Month.January)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # Ascension Day
            or (dy == em + 38)
            # Whit Monday
            or (dy == em + 49)
            # Labour Day
            or (dom == 1 and m == Month.May)
            # National Day
            or (dom == 1 and m == Month.August)
            # Christmas
            or (dom == 25 and m == Month.December)
            # St. Stephen's Day
            or (dom == 26 and m == Month.December)
        )
