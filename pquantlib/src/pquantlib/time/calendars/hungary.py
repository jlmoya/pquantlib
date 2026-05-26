"""Hungary — Hungarian calendar.

# C++ parity: ql/time/calendars/hungary.hpp + hungary.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- Good Friday (since 2017)
- Easter Monday
- Whit(Pentecost) Monday
- New Year's Day, January 1st
- National Day, March 15th
- Labour Day, May 1st
- Constitution Day, August 20th
- Republic Day, October 23rd
- All Saints Day, November 1st
- Christmas, December 25th
- 2nd Day of Christmas, December 26th
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Hungary(WesternCalendar):
    def name(self) -> str:
        return "Hungary"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Good Friday (since 2017)
            or (dy == em - 3 and y >= 2017)
            # Easter Monday
            or (dy == em)
            # Whit Monday
            or (dy == em + 49)
            # New Year's Day
            or (dom == 1 and m == Month.January)
            # National Day
            or (dom == 15 and m == Month.March)
            # Labour Day
            or (dom == 1 and m == Month.May)
            # Constitution Day
            or (dom == 20 and m == Month.August)
            # Republic Day
            or (dom == 23 and m == Month.October)
            # All Saints Day
            or (dom == 1 and m == Month.November)
            # Christmas
            or (dom == 25 and m == Month.December)
            # 2nd Day of Christmas
            or (dom == 26 and m == Month.December)
        )
