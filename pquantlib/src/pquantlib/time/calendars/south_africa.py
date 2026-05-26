"""South African calendar.

# C++ parity: ql/time/calendars/southafrica.hpp + southafrica.cpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class SouthAfrica(WesternCalendar):
    """South African national calendar."""

    def name(self) -> str:
        return "South Africa"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            # New Year's Day (possibly moved to Monday)
            ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.January)
            # Good Friday
            or (dd == em - 3)
            # Family Day (Easter Monday)
            or (dd == em)
            # Human Rights Day, March 21st (possibly moved to Monday)
            or ((day == 21 or (day == 22 and w == Weekday.Monday)) and m == Month.March)
            # Freedom Day, April 27th (possibly moved to Monday)
            or ((day == 27 or (day == 28 and w == Weekday.Monday)) and m == Month.April)
            # Election Day, April 14th 2004
            or (day == 14 and m == Month.April and y == 2004)
            # Workers Day, May 1st (possibly moved to Monday)
            or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.May)
            # Youth Day, June 16th (possibly moved to Monday)
            or ((day == 16 or (day == 17 and w == Weekday.Monday)) and m == Month.June)
            # National Women's Day, August 9th (possibly moved to Monday)
            or ((day == 9 or (day == 10 and w == Weekday.Monday)) and m == Month.August)
            # Heritage Day, September 24th (possibly moved to Monday)
            or ((day == 24 or (day == 25 and w == Weekday.Monday)) and m == Month.September)
            # Day of Reconciliation, December 16th (possibly moved to Monday)
            or ((day == 16 or (day == 17 and w == Weekday.Monday)) and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Day of Goodwill (possibly moved to Monday)
            or ((day == 26 or (day == 27 and w == Weekday.Monday)) and m == Month.December)
            # one-shot: Election day 2009
            or (day == 22 and m == Month.April and y == 2009)
            # one-shot: Election day 2016
            or (day == 3 and m == Month.August and y == 2016)
            # one-shot: Election day 2021
            or (day == 1 and m == Month.November and y == 2021)
            # one-shot: In lieu of Christmas falling on Sunday in 2022
            or (day == 27 and m == Month.December and y == 2022)
            # one-shot: Special holiday for Rugby World Cup 2023 win
            or (day == 15 and m == Month.December and y == 2023)
            # one-shot: Election day 2024
            or (day == 29 and m == Month.May and y == 2024)
        )
