"""Turkey (Istanbul Stock Exchange) calendar.

# C++ parity: ql/time/calendars/turkey.hpp + .cpp (v1.42.1).

C++ uses a fresh ``Calendar::Impl`` (not ``WesternImpl``), but the
weekend is the standard Saturday + Sunday — so we still subclass
``WesternCalendar`` for the weekend logic. The body is a year-by-year
table of Kurban/Ramadan holidays plus a small set of national fixed
holidays.

To keep the C++ control flow easy to recognise we keep one branch per
year, but factor the year-table into ``_is_local_holiday`` so the main
``_is_business_day`` stays small.
"""

from __future__ import annotations

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _is_local_holiday(day: int, m: Month, y: int) -> bool:
    # C++ parity: ``turkey.cpp`` per-year Kurban/Ramadan tables.
    if y == 2004:
        return (m == Month.February and day <= 4) or (m == Month.November and 14 <= day <= 16)
    if y == 2005:
        return (m == Month.January and 19 <= day <= 21) or (m == Month.November and 2 <= day <= 5)
    if y == 2006:
        return (
            (m == Month.January and 10 <= day <= 13)
            or (m == Month.October and 23 <= day <= 25)
            or (m == Month.December and day == 31)
        )
    if y == 2007:
        return (
            (m == Month.January and day <= 3)
            or (m == Month.October and 12 <= day <= 14)
            or (m == Month.December and 20 <= day <= 23)
        )
    if y == 2008:
        return (
            (m == Month.September and day == 30)
            or (m == Month.October and day <= 2)
            or (m == Month.December and 8 <= day <= 11)
        )
    if y == 2009:
        return (m == Month.September and 20 <= day <= 22) or (m == Month.November and 27 <= day <= 30)
    if y == 2010:
        return (m == Month.September and 9 <= day <= 11) or (m == Month.November and 16 <= day <= 19)
    if y == 2011:
        return (m == Month.October and day == 1) or (m == Month.November and 9 <= day <= 13)
    if y == 2012:
        return (m == Month.August and 18 <= day <= 21) or (m == Month.October and 24 <= day <= 28)
    if y == 2013:
        return (
            (m == Month.August and 7 <= day <= 10)
            or (m == Month.October and 14 <= day <= 18)
            or (m == Month.October and day == 28)
        )
    if y == 2014:
        return (
            (m == Month.July and 27 <= day <= 30)
            or (m == Month.October and 4 <= day <= 7)
            or (m == Month.October and day == 29)
        )
    if y == 2015:
        return (m == Month.July and 17 <= day <= 19) or (m == Month.October and 24 <= day <= 27)
    if y == 2016:
        return (m == Month.July and 5 <= day <= 7) or (m == Month.September and 12 <= day <= 15)
    if y == 2017:
        return (m == Month.June and 25 <= day <= 27) or (m == Month.September and 1 <= day <= 4)
    if y == 2018:
        return (m == Month.June and 15 <= day <= 17) or (m == Month.August and 21 <= day <= 24)
    if y == 2019:
        return (m == Month.June and 4 <= day <= 6) or (m == Month.August and 11 <= day <= 14)
    if y == 2020:
        return (
            (m == Month.May and 24 <= day <= 26)
            or (m == Month.July and day == 31)
            or (m == Month.August and 1 <= day <= 3)
        )
    if y == 2021:
        return (m == Month.May and 13 <= day <= 15) or (m == Month.July and 20 <= day <= 23)
    if y == 2022:
        return (m == Month.May and 2 <= day <= 4) or (m == Month.July and 9 <= day <= 12)
    if y == 2023:
        # July 1 is also a holiday but falls on a Saturday (already flagged)
        return (m == Month.April and 21 <= day <= 23) or (m == Month.June and 28 <= day <= 30)
    if y == 2024:
        # Note: Holidays >= 2024 are not yet officially announced and need validation.
        return (m == Month.April and 10 <= day <= 12) or (m == Month.June and 17 <= day <= 19)
    if y == 2025:
        return (
            (m == Month.March and day == 31)
            or (m == Month.April and 1 <= day <= 2)
            or (m == Month.June and 6 <= day <= 9)
        )
    if y == 2026:
        return (m == Month.March and 20 <= day <= 22) or (m == Month.May and 26 <= day <= 29)
    if y == 2027:
        return (m == Month.March and 10 <= day <= 12) or (m == Month.May and 16 <= day <= 19)
    if y == 2028:
        return (m == Month.February and 27 <= day <= 29) or (m == Month.May and 4 <= day <= 7)
    if y == 2029:
        return (m == Month.February and 15 <= day <= 17) or (m == Month.April and 23 <= day <= 26)
    if y == 2030:
        return (m == Month.February and 5 <= day <= 7) or (m == Month.April and 13 <= day <= 16)
    if y == 2031:
        return (m == Month.January and 25 <= day <= 27) or (m == Month.April and 2 <= day <= 5)
    if y == 2032:
        return (m == Month.January and 14 <= day <= 16) or (m == Month.March and 21 <= day <= 24)
    if y == 2033:
        return (
            (m == Month.January and 3 <= day <= 5)
            or (m == Month.December and day == 23)
            or (m == Month.March and 11 <= day <= 14)
        )
    if y == 2034:
        return (
            (m == Month.December and 12 <= day <= 14)
            or (m == Month.February and day == 28)
            or (m == Month.March and 1 <= day <= 3)
        )
    return False


class Turkey(WesternCalendar):
    """Istanbul Stock Exchange / Turkish public holidays."""

    def name(self) -> str:
        return "Turkey"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        y = d.year()

        if (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # 23 nisan / National Holiday
            or (day == 23 and m == Month.April)
            # 1 may/ National Holiday
            or (day == 1 and m == Month.May)
            # 19 may/ National Holiday
            or (day == 19 and m == Month.May)
            # 15 july / National Holiday (since 2017)
            or (day == 15 and m == Month.July and y >= 2017)
            # 30 aug/ National Holiday
            or (day == 30 and m == Month.August)
            # 29 ekim  National Holiday
            or (day == 29 and m == Month.October)
        ):
            return False

        return not _is_local_holiday(day, m, y)
