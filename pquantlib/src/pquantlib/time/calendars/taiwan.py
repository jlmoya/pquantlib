"""Taiwan (Taiwan Stock Exchange — TSEC) calendar.

# C++ parity: ql/time/calendars/taiwan.hpp + .cpp (v1.42.1).

Sat+Sun weekend. The C++ ``TsecImpl`` derives from the generic
``Calendar::Impl`` (not ``WesternImpl``), but it still uses the
Sat+Sun weekend rule, so we subclass ``WesternCalendar`` for the
weekend logic. Holiday tables for Chinese Lunar New Year, Tomb
Sweeping Day, Dragon Boat Festival and Moon Festival are encoded
year-by-year matching the C++ source — factored into
``_is_per_year_holiday`` so the main method stays small.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class TaiwanMarket(IntEnum):
    TSEC = 0  # Taiwan stock exchange


def _is_per_year_holiday(day: int, m: Month, y: int) -> bool:
    # C++ parity: ``taiwan.cpp`` per-year Lunar/Tomb-Sweeping/Dragon/Moon tables.
    if y == 2002:
        # Dragon Boat Festival and Moon Festival fall on Saturday
        return (9 <= day <= 17 and m == Month.February) or (day == 5 and m == Month.April)
    if y == 2003:
        # Tomb Sweeping Day falls on Saturday
        return (
            ((day >= 31 and m == Month.January) or (day <= 5 and m == Month.February))
            or (day == 4 and m == Month.June)
            or (day == 11 and m == Month.September)
        )
    if y == 2004:
        # Tomb Sweeping Day falls on Sunday
        return (
            (21 <= day <= 26 and m == Month.January)
            or (day == 22 and m == Month.June)
            or (day == 28 and m == Month.September)
        )
    if y == 2005:
        return (
            (6 <= day <= 13 and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 2 and m == Month.May)
        )
    if y == 2006:
        return (
            ((day >= 28 and m == Month.January) or (day <= 5 and m == Month.February))
            or (day == 5 and m == Month.April)
            or (day == 31 and m == Month.May)
            or (day == 6 and m == Month.October)
        )
    if y == 2007:
        return (
            (17 <= day <= 25 and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 6 and m == Month.April)
            or (day == 18 and m == Month.June)
            or (day == 19 and m == Month.June)
            or (day == 24 and m == Month.September)
            or (day == 25 and m == Month.September)
        )
    if y == 2008:
        return (4 <= day <= 11 and m == Month.February) or (day == 4 and m == Month.April)
    if y == 2009:
        return (
            (day == 2 and m == Month.January)
            or (day >= 24 and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day in {28, 29} and m == Month.May)
            or (day == 3 and m == Month.October)
        )
    if y == 2010:
        return (
            (13 <= day <= 21 and m == Month.January)
            or (day == 5 and m == Month.April)
            or (day == 16 and m == Month.May)
            or (day == 22 and m == Month.September)
        )
    if y == 2011:
        return (
            (2 <= day <= 7 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 2 and m == Month.May)
            or (day == 6 and m == Month.June)
            or (day == 12 and m == Month.September)
        )
    if y == 2012:
        return (
            (23 <= day <= 27 and m == Month.January)
            or (day == 27 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 1 and m == Month.May)
            or (day == 23 and m == Month.June)
            or (day == 30 and m == Month.September)
            or (day == 31 and m == Month.December)
        )
    if y == 2013:
        return (
            (10 <= day <= 15 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 1 and m == Month.May)
            or (day == 12 and m == Month.June)
            or (19 <= day <= 20 and m == Month.September)
        )
    if y == 2014:
        return (
            (28 <= day <= 30 and m == Month.January)
            or ((day == 31 and m == Month.January) or (day <= 4 and m == Month.February))
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 2 and m == Month.June)
            or (day == 8 and m == Month.September)
        )
    if y == 2015:
        return (
            (day == 2 and m == Month.January)
            or (18 <= day <= 23 and m == Month.February)
            or (day == 27 and m == Month.February)
            or (day == 3 and m == Month.April)
            or (day == 6 and m == Month.April)
            or (day == 19 and m == Month.June)
            or (day == 28 and m == Month.September)
            or (day == 9 and m == Month.October)
        )
    if y == 2016:
        return (
            (8 <= day <= 12 and m == Month.February)
            or (day == 29 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 2 and m == Month.May)
            or (day == 9 and m == Month.June)
            or (day == 10 and m == Month.June)
            or (day == 15 and m == Month.September)
            or (day == 16 and m == Month.September)
        )
    if y == 2017:
        return (
            (day == 2 and m == Month.January)
            or ((day >= 27 and m == Month.January) or (day == 1 and m == Month.February))
            or (day == 27 and m == Month.February)
            or (day == 3 and m == Month.April)
            or (day == 4 and m == Month.April)
            or (day == 29 and m == Month.May)
            or (day == 30 and m == Month.May)
            or (day == 4 and m == Month.October)
            or (day == 9 and m == Month.October)
        )
    if y == 2018:
        return (
            (15 <= day <= 20 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 6 and m == Month.April)
            or (day == 18 and m == Month.June)
            or (day == 24 and m == Month.September)
            or (day == 31 and m == Month.December)
        )
    if y == 2019:
        return (
            (4 <= day <= 8 and m == Month.February)
            or (day == 1 and m == Month.March)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 7 and m == Month.June)
            or (day == 13 and m == Month.September)
            or (day == 11 and m == Month.October)
        )
    if y == 2020:
        return (
            (day == 23 and m == Month.January)
            or (24 <= day <= 29 and m == Month.January)
            or (day == 2 and m == Month.April)
            or (day == 3 and m == Month.April)
            or (day == 25 and m == Month.June)
            or (day == 26 and m == Month.June)
            or (day == 1 and m == Month.October)
            or (day == 2 and m == Month.October)
            or (day == 9 and m == Month.October)
        )
    if y == 2021:
        # Tomb Sweeping Day falls on Sunday
        return (
            (day == 10 and m == Month.February)
            or (11 <= day <= 16 and m == Month.February)
            or (day == 1 and m == Month.March)
            or (day == 2 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 30 and m == Month.April)
            or (day == 14 and m == Month.June)
            or (day == 20 and m == Month.September)
            or (day == 21 and m == Month.September)
            or (day == 11 and m == Month.October)
            or (day == 31 and m == Month.December)
        )
    if y == 2022:
        # Mid-Autumn Festival falls on Saturday
        return (
            ((day == 31 and m == Month.January) or (day <= 4 and m == Month.February))
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 2 and m == Month.May)
            or (day == 3 and m == Month.June)
            or (day == 9 and m == Month.September)
        )
    if y == 2023:
        return (
            (day == 2 and m == Month.January)
            or (day == 20 and m == Month.January)
            or (21 <= day <= 24 and m == Month.January)
            or (25 <= day <= 27 and m == Month.January)
            or (day == 27 and m == Month.February)
            or (day == 3 and m == Month.April)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 22 and m == Month.June)
            or (day == 23 and m == Month.June)
            or (day == 29 and m == Month.September)
            or (day == 9 and m == Month.October)
        )
    if y == 2024:
        return (
            (day == 8 and m == Month.February)
            or (9 <= day <= 12 and m == Month.February)
            or (13 <= day <= 14 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.April)
            or (day == 10 and m == Month.June)
            or (day == 17 and m == Month.September)
        )
    if y == 2025:
        # Dragon Boat Festival falls on Saturday
        return (
            (23 <= day <= 24 and m == Month.January)
            or (27 <= day <= 31 and m == Month.January)
            or (day == 3 and m == Month.April)
            or (day == 4 and m == Month.April)
            or (day == 30 and m == Month.May)
            or (day == 6 and m == Month.October)
        )
    if y == 2026:
        return (
            (12 <= day <= 13 and m == Month.February)
            or (16 <= day <= 20 and m == Month.February)
            or (day == 27 and m == Month.February)
            or (day == 3 and m == Month.April)
            or (day == 6 and m == Month.April)
            or (day == 19 and m == Month.June)
            or (day == 25 and m == Month.September)
            or (day == 9 and m == Month.October)
        )
    return False


class Taiwan(WesternCalendar):
    """Taiwan Stock Exchange calendar."""

    def __init__(self, market: TaiwanMarket = TaiwanMarket.TSEC) -> None:
        super().__init__()
        # C++ silently accepts any value (constructor ignores ``market``);
        # we reject unknown values explicitly for diagnostic clarity.
        qassert.require(market == TaiwanMarket.TSEC, "unknown market")
        self._market: TaiwanMarket = market

    def name(self) -> str:
        return "Taiwan stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        y = d.year()

        if (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Peace Memorial Day
            or (day == 28 and m == Month.February)
            # Labor Day
            or (day == 1 and m == Month.May)
            # Double Tenth
            or (day == 10 and m == Month.October)
        ):
            return False

        return not _is_per_year_holiday(day, m, y)
