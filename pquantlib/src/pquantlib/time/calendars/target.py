"""TARGET — Trans-European Automated Real-time Gross Settlement Express Transfer.

# C++ parity: ql/time/calendars/target.hpp + target.cpp (v1.42.1).

Holidays (see http://www.ecb.int):
- Saturdays
- Sundays
- New Year's Day, January 1st
- Good Friday (since 2000)
- Easter Monday (since 2000)
- Labour Day, May 1st (since 2000)
- Christmas, December 25th
- Day of Goodwill, December 26th (since 2000)
- December 31st (1998, 1999, and 2001)
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class TARGET(WesternCalendar):
    def name(self) -> str:
        return "TARGET"

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
            # Good Friday
            or (dy == em - 3 and y >= 2000)
            # Easter Monday
            or (dy == em and y >= 2000)
            # Labour Day
            or (dom == 1 and m == Month.May and y >= 2000)
            # Christmas
            or (dom == 25 and m == Month.December)
            # Day of Goodwill
            or (dom == 26 and m == Month.December and y >= 2000)
            # December 31st, 1998, 1999, and 2001 only
            or (dom == 31 and m == Month.December and y in (1998, 1999, 2001))
        )
