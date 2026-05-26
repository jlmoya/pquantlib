"""Sweden — Swedish calendar.

# C++ parity: ql/time/calendars/sweden.hpp + sweden.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- New Year's Day, January 1st
- Epiphany, January 6th
- Good Friday
- Easter Monday
- Ascension
- Whit(Pentecost) Monday (until 2004)
- May Day, May 1st
- National Day, June 6th (since 2005)
- Midsummer Eve (Friday between June 19-25)
- Christmas Eve, December 24th
- Christmas Day, December 25th
- Boxing Day, December 26th
- New Year's Eve, December 31st
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Sweden(WesternCalendar):
    def name(self) -> str:
        return "Sweden"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # Ascension Thursday
            or (dy == em + 38)
            # Whit Monday (till 2004)
            or (dy == em + 49 and y < 2005)
            # New Year's Day
            or (dom == 1 and m == Month.January)
            # Epiphany
            or (dom == 6 and m == Month.January)
            # May Day
            or (dom == 1 and m == Month.May)
            # National Day (only a holiday since 2005)
            or (dom == 6 and m == Month.June and y >= 2005)
            # Midsummer Eve (Friday between June 19-25)
            or (w == Weekday.Friday and 19 <= dom <= 25 and m == Month.June)
            # Christmas Eve
            or (dom == 24 and m == Month.December)
            # Christmas Day
            or (dom == 25 and m == Month.December)
            # Boxing Day
            or (dom == 26 and m == Month.December)
            # New Year's Eve
            or (dom == 31 and m == Month.December)
        )
