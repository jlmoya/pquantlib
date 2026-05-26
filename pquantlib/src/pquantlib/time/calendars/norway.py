"""Norway — Norwegian calendar.

# C++ parity: ql/time/calendars/norway.hpp + norway.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- Holy Thursday
- Good Friday
- Easter Monday
- Ascension
- Whit(Pentecost) Monday
- New Year's Day, January 1st
- May Day, May 1st
- National Independence Day, May 17th
- Christmas Eve, December 24th (since 2002)
- Christmas, December 25th
- Boxing Day, December 26th
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Norway(WesternCalendar):
    def name(self) -> str:
        return "Norway"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Holy Thursday
            or (dy == em - 4)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # Ascension Thursday
            or (dy == em + 38)
            # Whit Monday
            or (dy == em + 49)
            # New Year's Day
            or (dom == 1 and m == Month.January)
            # May Day
            or (dom == 1 and m == Month.May)
            # National Independence Day
            or (dom == 17 and m == Month.May)
            # Christmas Eve
            or (dom == 24 and m == Month.December and y >= 2002)
            # Christmas
            or (dom == 25 and m == Month.December)
            # Boxing Day
            or (dom == 26 and m == Month.December)
        )
