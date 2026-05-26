"""Finland — Finnish calendar.

# C++ parity: ql/time/calendars/finland.hpp + finland.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- New Year's Day, January 1st
- Epiphany, January 6th
- Good Friday
- Easter Monday
- Ascension Thursday
- Labour Day, May 1st
- Midsummer Eve (Friday between June 19-25)
- Independence Day, December 6th
- Christmas Eve, December 24th
- Christmas, December 25th
- Boxing Day, December 26th
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Finland(WesternCalendar):
    def name(self) -> str:
        return "Finland"

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
            # Epiphany
            or (dom == 6 and m == Month.January)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # Ascension Thursday
            or (dy == em + 38)
            # Labour Day
            or (dom == 1 and m == Month.May)
            # Midsummer Eve (Friday between June 19-25)
            or (w == Weekday.Friday and 19 <= dom <= 25 and m == Month.June)
            # Independence Day
            or (dom == 6 and m == Month.December)
            # Christmas Eve
            or (dom == 24 and m == Month.December)
            # Christmas
            or (dom == 25 and m == Month.December)
            # Boxing Day
            or (dom == 26 and m == Month.December)
        )
