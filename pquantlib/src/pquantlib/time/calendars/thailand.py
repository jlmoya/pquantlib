"""Thailand calendar — Thailand stock exchange (SET).

# C++ parity: ql/time/calendars/thailand.hpp + thailand.cpp (v1.42.1).

The C++ class has no Market enum (single SET market); the Python port
exposes a single-value ``Market.SET`` enum for symmetry with the rest of
the calendar family.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Thailand(WesternCalendar):
    """Thailand calendar (SET stock exchange)."""

    class Market(IntEnum):
        SET = 0  # Thailand stock exchange

    def __init__(self, market: Market = Market.SET) -> None:
        super().__init__()
        qassert.require(market == Thailand.Market.SET, "unknown market")
        self._market = market

    def name(self) -> str:
        return "Thailand stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        m = d.month()
        y = d.year()

        if _is_fixed_set_holiday(day, w, m, y):
            return False
        return not _is_year_specific_set_holiday(day, m, y)


def _is_fixed_set_holiday(day: int, w: Weekday, m: Month, y: int) -> bool:
    """Annually-recurring SET holidays (with substitution-Monday rules)."""
    return (
        # New Year's Day
        ((day == 1 or (day == 3 and w == Weekday.Monday)) and m == Month.January)
        # Chakri Memorial Day
        or ((day == 6 or ((day in (7, 8)) and w == Weekday.Monday)) and m == Month.April)
        # Songkran Festival (cancelled 2020 due to COVID)
        or ((day in (13, 14, 15)) and m == Month.April and y != 2020)
        # Substitution Songkran Festival (cancelled 2020)
        or (day == 16 and w in (Weekday.Monday, Weekday.Tuesday) and m == Month.April and y != 2020)
        # Labor Day
        or ((day == 1 or ((day in (2, 3)) and w == Weekday.Monday)) and m == Month.May)
        # Coronation Day (since 2019)
        or ((day == 4 or ((day in (5, 6)) and w == Weekday.Monday)) and m == Month.May and y >= 2019)
        # H.M. Queen Suthida's Birthday (since 2019)
        or ((day == 3 or ((day in (4, 5)) and w == Weekday.Monday)) and m == Month.June and y >= 2019)
        # H.M. King Maha Vajiralongkorn's Birthday (since 2017)
        or ((day == 28 or ((day in (29, 30)) and w == Weekday.Monday)) and m == Month.July and y >= 2017)
        # H.M. Queen Sirikit's Birthday / Mother's Day
        or ((day == 12 or ((day in (13, 14)) and w == Weekday.Monday)) and m == Month.August)
        # H.M. King Bhumibol Adulyadej Memorial Day (since 2017)
        or ((day == 13 or ((day in (14, 15)) and w == Weekday.Monday)) and m == Month.October and y >= 2017)
        # Chulalongkorn Day (moved in 2021)
        or ((day == 23 or ((day in (24, 25)) and w == Weekday.Monday)) and m == Month.October and y != 2021)
        # H.M. King Bhumibol Adulyadej's Birthday / National Day / Father's Day
        or ((day == 5 or ((day in (6, 7)) and w == Weekday.Monday)) and m == Month.December)
        # Constitution Day
        or ((day == 10 or ((day in (11, 12)) and w == Weekday.Monday)) and m == Month.December)
        # New Year's Eve (moved in 2024)
        or (
            (day == 31 and m == Month.December)
            or (day == 2 and w == Weekday.Monday and m == Month.January and y != 2024)
        )
    )


def _is_year_specific_set_holiday(day: int, m: Month, y: int) -> bool:
    """Year-specific SET holidays (Buddhist / Coronation / substitutions)."""
    if y == 2000:
        return (
            (day == 21 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 17 and m == Month.May)
            or (day == 17 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2001:
        return (
            (day == 8 and m == Month.February)
            or (day == 7 and m == Month.May)
            or (day == 8 and m == Month.May)
            or (day == 6 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2005:
        return (
            (day == 23 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 23 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 22 and m == Month.July)
            or (day == 24 and m == Month.October)
        )
    if y == 2006:
        return (
            (day == 13 and m == Month.February)
            or (day == 19 and m == Month.April)
            or (day == 5 and m == Month.May)
            or (day == 12 and m == Month.May)
            or (day == 12 and m == Month.June)
            or (day == 13 and m == Month.June)
            or (day == 11 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2007:
        return (
            (day == 5 and m == Month.March)
            or (day == 7 and m == Month.May)
            or (day == 31 and m == Month.May)
            or (day == 30 and m == Month.July)
            or (day == 23 and m == Month.October)
            or (day == 24 and m == Month.December)
        )
    if y == 2008:
        return (
            (day == 21 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 19 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 17 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2009:
        return (
            (day == 2 and m == Month.January)
            or (day == 9 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 8 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 6 and m == Month.July)
            or (day == 7 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2010:
        return (
            (day == 1 and m == Month.March)
            or (day == 5 and m == Month.May)
            or (day == 20 and m == Month.May)
            or (day == 21 and m == Month.May)
            or (day == 28 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 26 and m == Month.July)
            or (day == 13 and m == Month.August)
            or (day == 25 and m == Month.October)
        )
    if y == 2011:
        return (
            (day == 18 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 16 and m == Month.May)
            or (day == 17 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 15 and m == Month.July)
            or (day == 24 and m == Month.October)
        )
    if y == 2012:
        return (
            (day == 3 and m == Month.January)
            or (day == 7 and m == Month.March)
            or (day == 9 and m == Month.April)
            or (day == 7 and m == Month.May)
            or (day == 4 and m == Month.June)
            or (day == 2 and m == Month.August)
            or (day == 23 and m == Month.October)
        )
    if y == 2013:
        return (
            (day == 25 and m == Month.February)
            or (day == 6 and m == Month.May)
            or (day == 24 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 22 and m == Month.July)
            or (day == 23 and m == Month.October)
            or (day == 30 and m == Month.December)
        )
    if y == 2014:
        return (
            (day == 14 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 13 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 11 and m == Month.July)
            or (day == 11 and m == Month.August)
            or (day == 23 and m == Month.October)
        )
    if y == 2015:
        return (
            (day == 2 and m == Month.January)
            or (day == 4 and m == Month.March)
            or (day == 4 and m == Month.May)
            or (day == 5 and m == Month.May)
            or (day == 1 and m == Month.June)
            or (day == 1 and m == Month.July)
            or (day == 30 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2016:
        return (
            (day == 22 and m == Month.February)
            or (day == 5 and m == Month.May)
            or (day == 6 and m == Month.May)
            or (day == 20 and m == Month.May)
            or (day == 1 and m == Month.July)
            or (day == 18 and m == Month.July)
            or (day == 19 and m == Month.July)
            or (day == 24 and m == Month.October)
        )
    if y == 2017:
        return (
            (day == 13 and m == Month.February)
            or (day == 10 and m == Month.May)
            or (day == 10 and m == Month.July)
            or (day == 23 and m == Month.October)
            or (day == 26 and m == Month.October)
        )
    if y == 2018:
        return (
            (day == 1 and m == Month.March)
            or (day == 29 and m == Month.May)
            or (day == 27 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2019:
        return (
            (day == 19 and m == Month.February)
            or (day == 6 and m == Month.May)
            or (day == 20 and m == Month.May)
            or (day == 16 and m == Month.July)
        )
    if y == 2020:
        return (
            (day == 10 and m == Month.February)
            or (day == 6 and m == Month.May)
            or (day == 6 and m == Month.July)
            or (day == 27 and m == Month.July)
            or (day == 4 and m == Month.September)
            or (day == 7 and m == Month.September)
            or (day == 11 and m == Month.December)
        )
    if y == 2021:
        return (
            (day == 12 and m == Month.February)
            or (day == 26 and m == Month.February)
            or (day == 26 and m == Month.May)
            or (day == 26 and m == Month.July)
            or (day == 24 and m == Month.September)
            or (day == 22 and m == Month.October)
        )
    if y == 2022:
        return (
            (day == 16 and m == Month.February)
            or (day == 16 and m == Month.May)
            or (day == 13 and m == Month.July)
            or (day == 29 and m == Month.July)
            or (day == 14 and m == Month.October)
            or (day == 24 and m == Month.October)
        )
    if y == 2023:
        return (
            (day == 6 and m == Month.March)
            or (day == 5 and m == Month.May)
            or (day == 5 and m == Month.June)
            or (day == 1 and m == Month.August)
            or (day == 23 and m == Month.October)
            or (day == 29 and m == Month.December)
        )
    if y == 2024:
        return (
            (day == 26 and m == Month.February)
            or (day == 8 and m == Month.April)
            or (day == 12 and m == Month.April)
            or (day == 6 and m == Month.May)
            or (day == 22 and m == Month.May)
            or (day == 22 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    if y == 2025:
        return (
            (day == 12 and m == Month.February)
            or (day == 7 and m == Month.April)
            or (day == 5 and m == Month.May)
            or (day == 12 and m == Month.May)
            or (day == 10 and m == Month.July)
            or (day == 23 and m == Month.October)
        )
    return False
