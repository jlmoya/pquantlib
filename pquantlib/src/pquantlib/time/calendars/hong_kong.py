"""Hong Kong calendar — HKEx stock exchange.

# C++ parity: ql/time/calendars/hongkong.hpp + hongkong.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class HongKong(WesternCalendar):
    """Hong Kong calendar (HKEx stock exchange)."""

    class Market(IntEnum):
        HKEx = 0  # Hong Kong stock exchange

    def __init__(self, market: Market = Market.HKEx) -> None:
        super().__init__()
        qassert.require(market == HongKong.Market.HKEx, "unknown market")
        self._market = market

    def name(self) -> str:
        return "Hong Kong stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)

        if _is_fixed_hkex_holiday(day, w, dd, em, m):
            return False
        return not _is_year_specific_hkex_holiday(day, m, y)


def _is_fixed_hkex_holiday(day: int, w: Weekday, dd: int, em: int, m: Month) -> bool:
    """Annually-recurring HKEx public holidays."""
    return (
        # New Year's Day
        ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.January)
        # Good Friday
        or (dd == em - 3)
        # Easter Monday
        or (dd == em)
        # Labor Day
        or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.May)
        # SAR Establishment Day
        or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.July)
        # National Day
        or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.October)
        # Christmas Day
        or (day == 25 and m == Month.December)
        # Boxing Day
        or (day == 26 and m == Month.December)
    )


def _is_year_specific_hkex_holiday(day: int, m: Month, y: int) -> bool:
    """Year-specific HKEx holidays (Lunar / Ching Ming / Buddha / etc.) for 2004..2025."""
    if y == 2004:
        return (
            (day in (22, 23, 24) and m == Month.January)
            or (day == 5 and m == Month.April)
            or (day == 26 and m == Month.May)
            or (day == 22 and m == Month.June)
            or (day == 29 and m == Month.September)
            or (day == 22 and m == Month.October)
        )
    if y == 2005:
        return (
            (day in (9, 10, 11) and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 16 and m == Month.May)
            or (day == 11 and m == Month.June)
            or (day == 19 and m == Month.September)
            or (day == 11 and m == Month.October)
        )
    if y == 2006:
        return (
            (28 <= day <= 31 and m == Month.January)
            or (day == 5 and m == Month.April)
            or (day == 5 and m == Month.May)
            or (day == 31 and m == Month.May)
            or (day == 7 and m == Month.October)
            or (day == 30 and m == Month.October)
        )
    if y == 2007:
        return (
            (17 <= day <= 20 and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 24 and m == Month.May)
            or (day == 19 and m == Month.June)
            or (day == 26 and m == Month.September)
            or (day == 19 and m == Month.October)
        )
    if y == 2008:
        return (
            (7 <= day <= 9 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 12 and m == Month.May)
            or (day == 9 and m == Month.June)
            or (day == 15 and m == Month.September)
            or (day == 7 and m == Month.October)
        )
    if y == 2009:
        return (
            (26 <= day <= 28 and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day == 2 and m == Month.May)
            or (day == 28 and m == Month.May)
            or (day == 3 and m == Month.October)
            or (day == 26 and m == Month.October)
        )
    if y == 2010:
        return (
            (day in (15, 16) and m == Month.February)
            or (day == 6 and m == Month.April)
            or (day == 21 and m == Month.May)
            or (day == 16 and m == Month.June)
            or (day == 23 and m == Month.September)
        )
    if y == 2011:
        return (
            (day in (3, 4) and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 10 and m == Month.May)
            or (day == 6 and m == Month.June)
            or (day == 13 and m == Month.September)
            or (day == 5 and m == Month.October)
            or (day == 27 and m == Month.December)
        )
    if y == 2012:
        return (
            (23 <= day <= 25 and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day == 10 and m == Month.May)
            or (day == 1 and m == Month.October)
            or (day == 23 and m == Month.October)
        )
    if y == 2013:
        return (
            (11 <= day <= 13 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 17 and m == Month.May)
            or (day == 12 and m == Month.June)
            or (day == 20 and m == Month.September)
            or (day == 14 and m == Month.October)
        )
    if y == 2014:
        return (
            ((day == 31 and m == Month.January) or (day <= 3 and m == Month.February))
            or (day == 6 and m == Month.May)
            or (day == 2 and m == Month.June)
            or (day == 9 and m == Month.September)
            or (day == 2 and m == Month.October)
        )
    if y == 2015:
        return (
            (day in (19, 20) and m == Month.February)
            or (day == 7 and m == Month.April)
            or (day == 25 and m == Month.May)
            or (day == 20 and m == Month.June)
            or (day == 3 and m == Month.September)
            or (day == 28 and m == Month.September)
            or (day == 21 and m == Month.October)
        )
    if y == 2016:
        return (
            (8 <= day <= 10 and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 9 and m == Month.June)
            or (day == 16 and m == Month.September)
            or (day == 10 and m == Month.October)
            or (day == 27 and m == Month.December)
        )
    if y == 2017:
        return (
            (day in (30, 31) and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day == 3 and m == Month.May)
            or (day == 30 and m == Month.May)
            or (day == 5 and m == Month.October)
        )
    if y == 2018:
        return (
            (day in (16, 19) and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 22 and m == Month.May)
            or (day == 18 and m == Month.June)
            or (day == 25 and m == Month.September)
            or (day == 17 and m == Month.October)
        )
    if y == 2019:
        return (
            (5 <= day <= 7 and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 7 and m == Month.June)
            or (day == 7 and m == Month.October)
        )
    if y == 2020:
        return (
            (day in (27, 28) and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day == 30 and m == Month.April)
            or (day == 25 and m == Month.June)
            or (day == 2 and m == Month.October)
            or (day == 26 and m == Month.October)
        )
    if y == 2021:
        return (
            (day in (12, 15) and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 19 and m == Month.May)
            or (day == 14 and m == Month.June)
            or (day == 22 and m == Month.September)
            or (day == 14 and m == Month.October)
        )
    if y == 2022:
        return (
            (1 <= day <= 3 and m == Month.February)
            or (day == 5 and m == Month.April)
            or (day == 9 and m == Month.May)
            or (day == 3 and m == Month.June)
            or (day == 12 and m == Month.September)
            or (day == 4 and m == Month.October)
        )
    if y == 2023:
        return (
            (23 <= day <= 25 and m == Month.January)
            or (day == 5 and m == Month.April)
            or (day == 26 and m == Month.May)
            or (day == 22 and m == Month.June)
            or (day == 23 and m == Month.October)
        )
    if y == 2024:
        return (
            (day in (12, 13) and m == Month.February)
            or (day == 4 and m == Month.April)
            or (day == 15 and m == Month.May)
            or (day == 10 and m == Month.June)
            or (day == 18 and m == Month.September)
            or (day == 11 and m == Month.October)
        )
    if y == 2025:
        return (
            (29 <= day <= 31 and m == Month.January)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.May)
            or (day == 7 and m == Month.October)
            or (day == 29 and m == Month.October)
        )
    return False
