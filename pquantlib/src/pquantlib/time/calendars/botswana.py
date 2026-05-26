"""Botswana calendar.

# C++ parity: ql/time/calendars/botswana.hpp + botswana.cpp (v1.42.1).
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Botswana(WesternCalendar):
    """Botswana national calendar."""

    def name(self) -> str:
        return "Botswana"

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
            # New Year's Day (possibly moved to Monday or Tuesday)
            (
                (day == 1 or (day == 2 and w == Weekday.Monday) or (day == 3 and w == Weekday.Tuesday))
                and m == Month.January
            )
            # Good Friday
            or (dd == em - 3)
            # Easter Monday
            or (dd == em)
            # Labour Day, May 1st (possibly moved to Monday)
            or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.May)
            # Ascension
            or (dd == em + 38)
            # Sir Seretse Khama Day, July 1st (possibly moved to Monday)
            or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.July)
            # Presidents' Day (third Monday of July)
            or (15 <= day <= 21 and w == Weekday.Monday and m == Month.July)
            # Independence Day, September 30th (possibly moved to Monday)
            or (
                (day == 30 and m == Month.September)
                or (day == 1 and w == Weekday.Monday and m == Month.October)
            )
            # Botswana Day, October 1st (possibly moved to Monday or Tuesday)
            or (
                (day == 1 or (day == 2 and w == Weekday.Monday) or (day == 3 and w == Weekday.Tuesday))
                and m == Month.October
            )
            # Christmas
            or (day == 25 and m == Month.December)
            # Boxing Day (possibly moved to Monday)
            or ((day == 26 or (day == 27 and w == Weekday.Monday)) and m == Month.December)
        )
