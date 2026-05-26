"""Denmark — Danish calendar.

# C++ parity: ql/time/calendars/denmark.hpp + denmark.cpp (v1.42.1).

Holidays:
- Saturdays
- Sundays
- Maunday Thursday
- Good Friday
- Easter Monday
- General Prayer Day, 25 days after Easter Monday (up until 2023)
- Ascension
- Day after Ascension (from 2009)
- Whit (Pentecost) Monday
- New Year's Day, January 1st
- Constitution Day, June 5th
- Christmas Eve, December 24th
- Christmas, December 25th
- Boxing Day, December 26th
- New Year's Eve, December 31st
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Denmark(WesternCalendar):
    def name(self) -> str:
        return "Denmark"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dom = d.day_of_month()
        dy = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Maunday Thursday
            or (dy == em - 4)
            # Good Friday
            or (dy == em - 3)
            # Easter Monday
            or (dy == em)
            # General Prayer Day
            or (dy == em + 25 and y <= 2023)
            # Ascension
            or (dy == em + 38)
            # Day after Ascension
            or (dy == em + 39 and y >= 2009)
            # Whit Monday
            or (dy == em + 49)
            # New Year's Day
            or (dom == 1 and m == Month.January)
            # Constitution Day, June 5th
            or (dom == 5 and m == Month.June)
            # Christmas Eve
            or (dom == 24 and m == Month.December)
            # Christmas
            or (dom == 25 and m == Month.December)
            # Boxing Day
            or (dom == 26 and m == Month.December)
            # New Year's Eve
            or (dom == 31 and m == Month.December)
        )
