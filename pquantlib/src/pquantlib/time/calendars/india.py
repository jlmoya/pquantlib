"""Indian calendar — National Stock Exchange of India (NSE).

# C++ parity: ql/time/calendars/india.hpp + india.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class India(WesternCalendar):
    """Indian calendar (National Stock Exchange / NSE)."""

    class Market(IntEnum):
        NSE = 0  # National Stock Exchange

    def __init__(self, market: Market = Market.NSE) -> None:
        super().__init__()
        qassert.require(market == India.Market.NSE, "unknown market")
        self._market = market

    def name(self) -> str:
        return "National Stock Exchange of India"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)

        if _is_fixed_nse_holiday(day, dd, em, m):
            return False
        return not _is_year_specific_nse_holiday(day, m, y)


def _is_fixed_nse_holiday(day: int, dd: int, em: int, m: Month) -> bool:
    """Annually-recurring NSE public holidays."""
    return (
        # Republic Day
        (day == 26 and m == Month.January)
        # Good Friday
        or (dd == em - 3)
        # Ambedkar Jayanti
        or (day == 14 and m == Month.April)
        # May Day
        or (day == 1 and m == Month.May)
        # Independence Day
        or (day == 15 and m == Month.August)
        # Gandhi Jayanti
        or (day == 2 and m == Month.October)
        # Christmas
        or (day == 25 and m == Month.December)
    )


def _is_year_specific_nse_holiday(day: int, m: Month, y: int) -> bool:
    """Year-specific NSE holidays (Diwali / Holi / Eid / etc.) for years with C++ data."""
    if y == 2005:
        return (
            (day == 21 and m == Month.January)
            or (day == 7 and m == Month.September)
            or (day == 12 and m == Month.October)
            or (day == 1 and m == Month.November)
            or (day == 3 and m == Month.November)
            or (day == 15 and m == Month.November)
        )
    if y == 2006:
        return (
            (day == 11 and m == Month.January)
            or (day == 9 and m == Month.February)
            or (day == 15 and m == Month.March)
            or (day == 6 and m == Month.April)
            or (day == 11 and m == Month.April)
            or (day == 1 and m == Month.May)
            or (day == 24 and m == Month.October)
            or (day == 25 and m == Month.October)
        )
    if y == 2007:
        return (
            (day == 1 and m == Month.January)
            or (day == 30 and m == Month.January)
            or (day == 16 and m == Month.February)
            or (day == 27 and m == Month.March)
            or (day == 1 and m == Month.May)
            or (day == 2 and m == Month.May)
            or (day == 9 and m == Month.November)
            or (day == 21 and m == Month.December)
        )
    if y == 2008:
        return (
            (day == 6 and m == Month.March)
            or (day == 20 and m == Month.March)
            or (day == 18 and m == Month.April)
            or (day == 1 and m == Month.May)
            or (day == 19 and m == Month.May)
            or (day == 3 and m == Month.September)
            or (day == 2 and m == Month.October)
            or (day == 9 and m == Month.October)
            or (day == 28 and m == Month.October)
            or (day == 30 and m == Month.October)
            or (day == 13 and m == Month.November)
            or (day == 9 and m == Month.December)
        )
    if y == 2009:
        return (
            (day == 8 and m == Month.January)
            or (day == 23 and m == Month.February)
            or (day == 10 and m == Month.March)
            or (day == 11 and m == Month.March)
            or (day == 3 and m == Month.April)
            or (day == 7 and m == Month.April)
            or (day == 1 and m == Month.May)
            or (day == 21 and m == Month.September)
            or (day == 28 and m == Month.September)
            or (day == 19 and m == Month.October)
            or (day == 2 and m == Month.November)
            or (day == 28 and m == Month.December)
        )
    if y == 2010:
        return (
            (day == 1 and m == Month.January)
            or (day == 12 and m == Month.February)
            or (day == 1 and m == Month.March)
            or (day == 24 and m == Month.March)
            or (day == 10 and m == Month.September)
            or (day == 5 and m == Month.November)
            or (day == 17 and m == Month.November)
            or (day == 17 and m == Month.December)
        )
    if y == 2011:
        return (
            (day == 2 and m == Month.March)
            or (day == 12 and m == Month.April)
            or (day == 31 and m == Month.August)
            or (day == 1 and m == Month.September)
            or (day == 6 and m == Month.October)
            or (day == 26 and m == Month.October)
            or (day == 27 and m == Month.October)
            or (day == 7 and m == Month.November)
            or (day == 10 and m == Month.November)
            or (day == 6 and m == Month.December)
        )
    if y == 2012:
        return (
            (day == 20 and m == Month.February)
            or (day == 8 and m == Month.March)
            or (day == 5 and m == Month.April)
            or (day == 20 and m == Month.August)
            or (day == 19 and m == Month.September)
            or (day == 24 and m == Month.October)
            or (day == 14 and m == Month.November)
            or (day == 28 and m == Month.November)
        )
    if y == 2013:
        return (
            (day == 27 and m == Month.March)
            or (day == 19 and m == Month.April)
            or (day == 24 and m == Month.April)
            or (day == 9 and m == Month.August)
            or (day == 9 and m == Month.September)
            or (day == 16 and m == Month.October)
            or (day == 4 and m == Month.November)
            or (day == 14 and m == Month.November)
        )
    if y == 2014:
        return (
            (day == 27 and m == Month.February)
            or (day == 17 and m == Month.March)
            or (day == 8 and m == Month.April)
            or (day == 29 and m == Month.July)
            or (day == 29 and m == Month.August)
            or (day == 3 and m == Month.October)
            or (day == 6 and m == Month.October)
            or (day == 24 and m == Month.October)
            or (day == 4 and m == Month.November)
            or (day == 6 and m == Month.November)
        )
    if y == 2019:
        return (
            (day == 19 and m == Month.February)
            or (day == 4 and m == Month.March)
            or (day == 21 and m == Month.March)
            or (day == 1 and m == Month.April)
            or (day == 17 and m == Month.April)
            or (day == 29 and m == Month.April)
            or (day == 5 and m == Month.June)
            or (day == 12 and m == Month.August)
            or (day == 2 and m == Month.September)
            or (day == 10 and m == Month.September)
            or (day == 8 and m == Month.October)
            or (day == 21 and m == Month.October)
            or (day == 28 and m == Month.October)
            or (day == 12 and m == Month.November)
        )
    if y == 2020:
        return (
            (day == 19 and m == Month.February)
            or (day == 21 and m == Month.February)
            or (day == 10 and m == Month.March)
            or (day == 25 and m == Month.March)
            or (day == 1 and m == Month.April)
            or (day == 2 and m == Month.April)
            or (day == 6 and m == Month.April)
            or (day == 7 and m == Month.May)
            or (day == 25 and m == Month.May)
            or (day == 30 and m == Month.October)
            or (day == 16 and m == Month.November)
            or (day == 30 and m == Month.November)
        )
    if y == 2021:
        return (
            (day == 19 and m == Month.February)
            or (day == 11 and m == Month.March)
            or (day == 29 and m == Month.March)
            or (day == 13 and m == Month.April)
            or (day == 14 and m == Month.April)
            or (day == 21 and m == Month.April)
            or (day == 26 and m == Month.May)
            or (day == 21 and m == Month.July)
            or (day == 10 and m == Month.September)
            or (day == 15 and m == Month.October)
            or (day == 19 and m == Month.October)
            or (day == 5 and m == Month.November)
            or (day == 19 and m == Month.November)
        )
    if y == 2022:
        return (
            (day == 1 and m == Month.March)
            or (day == 18 and m == Month.March)
            or (day == 3 and m == Month.May)
            or (day == 16 and m == Month.May)
            or (day == 31 and m == Month.August)
            or (day == 5 and m == Month.October)
            or (day == 26 and m == Month.October)
            or (day == 8 and m == Month.November)
        )
    if y == 2023:
        return (
            (day == 7 and m == Month.March)
            or (day == 22 and m == Month.March)
            or (day == 30 and m == Month.March)
            or (day == 4 and m == Month.April)
            or (day == 5 and m == Month.May)
            or (day == 29 and m == Month.June)
            or (day == 16 and m == Month.August)
            or (day == 19 and m == Month.September)
            or (day == 29 and m == Month.September)
            or (day == 24 and m == Month.October)
            or (day == 14 and m == Month.November)
            or (day == 27 and m == Month.November)
        )
    if y == 2024:
        return (
            (day == 22 and m == Month.January)
            or (day == 19 and m == Month.February)
            or (day == 8 and m == Month.March)
            or (day == 25 and m == Month.March)
            or (day == 1 and m == Month.April)
            or (day == 9 and m == Month.April)
            or (day == 11 and m == Month.April)
            or (day == 17 and m == Month.April)
            or (day == 21 and m == Month.April)
            or (day == 20 and m == Month.May)
            or (day == 23 and m == Month.May)
            or (day == 17 and m == Month.June)
            or (day == 17 and m == Month.July)
            or (day == 16 and m == Month.September)
            or (day == 1 and m == Month.November)
            or (day == 15 and m == Month.November)
        )
    if y == 2025:
        return (
            (day == 19 and m == Month.February)
            or (day == 26 and m == Month.February)
            or (day == 14 and m == Month.March)
            or (day == 31 and m == Month.March)
            or (day == 10 and m == Month.April)
            or (day == 12 and m == Month.May)
            or (day == 5 and m == Month.September)
            or (day == 22 and m == Month.October)
            or (day == 5 and m == Month.November)
        )
    return False
