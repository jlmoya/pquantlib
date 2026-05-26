"""Indonesian calendars — Indonesia stock exchange (IDX / BEJ / JSX).

# C++ parity: ql/time/calendars/indonesia.hpp + indonesia.cpp (v1.42.1).

The three markets (BEJ, JSX, IDX) all delegate to the same ``BejImpl`` in
C++; the Python port matches by ignoring the market choice in
``_is_business_day`` (one branch only) but still validating the enum in
``__init__``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class Indonesia(WesternCalendar):
    """Indonesian calendar (Indonesia stock exchange)."""

    class Market(IntEnum):
        BEJ = 0  # Jakarta stock exchange (merged into IDX)
        JSX = 1  # Jakarta stock exchange (merged into IDX)
        IDX = 2  # Indonesia stock exchange

    def __init__(self, market: Market = Market.IDX) -> None:
        super().__init__()
        qassert.require(
            market in (Indonesia.Market.BEJ, Indonesia.Market.JSX, Indonesia.Market.IDX),
            "unknown market",
        )
        self._market = market

    def name(self) -> str:
        return "Jakarta stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)

        if _is_fixed_idx_holiday(day, dd, em, m):
            return False
        return not _is_year_specific_idx_holiday(day, m, y)


def _is_fixed_idx_holiday(day: int, dd: int, em: int, m: Month) -> bool:
    """Annually-recurring Indonesian public holidays."""
    return (
        # New Year's Day
        (day == 1 and m == Month.January)
        # Good Friday
        or (dd == em - 3)
        # Ascension Thursday
        or (dd == em + 38)
        # Independence Day
        or (day == 17 and m == Month.August)
        # Christmas
        or (day == 25 and m == Month.December)
    )


def _is_year_specific_idx_holiday(day: int, m: Month, y: int) -> bool:
    """Year-specific Indonesian holidays for 2005..2014."""
    if y == 2005:
        return (
            (day == 21 and m == Month.January)
            or (day == 9 and m == Month.February)
            or (day == 10 and m == Month.February)
            or (day == 11 and m == Month.March)
            or (day == 22 and m == Month.April)
            or (day == 24 and m == Month.May)
            or (day == 2 and m == Month.September)
            or ((day in (3, 4)) and m == Month.November)
            or ((day in (2, 7, 8)) and m == Month.November)
            or (day == 26 and m == Month.December)
        )
    if y == 2006:
        return (
            (day == 10 and m == Month.January)
            or (day == 31 and m == Month.January)
            or (day == 30 and m == Month.March)
            or (day == 10 and m == Month.April)
            or (day == 21 and m == Month.August)
            or ((day in (24, 25)) and m == Month.October)
            or ((day in (23, 26, 27)) and m == Month.October)
        )
    if y == 2007:
        return (
            (day == 19 and m == Month.March)
            or (day == 1 and m == Month.June)
            or (day == 20 and m == Month.December)
            or (day == 18 and m == Month.May)
            or ((day in (12, 15, 16)) and m == Month.October)
            or ((day in (21, 24)) and m == Month.October)
        )
    if y == 2008:
        return (
            ((day in (10, 11)) and m == Month.January)
            or ((day in (7, 8)) and m == Month.February)
            or (day == 7 and m == Month.March)
            or (day == 20 and m == Month.March)
            or (day == 20 and m == Month.May)
            or (day == 30 and m == Month.July)
            or (day == 18 and m == Month.August)
            or (day == 30 and m == Month.September)
            or ((day in (1, 2, 3)) and m == Month.October)
            or (day == 8 and m == Month.December)
            or (day == 29 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    if y == 2009:
        return (
            (day == 2 and m == Month.January)
            or (day == 26 and m == Month.January)
            or (day == 9 and m == Month.March)
            or (day == 26 and m == Month.March)
            or (day == 9 and m == Month.April)
            or (day == 20 and m == Month.July)
            or (18 <= day <= 23 and m == Month.September)
            or (day == 27 and m == Month.November)
            or (day == 18 and m == Month.December)
            or (day == 24 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    if y == 2010:
        return (
            (day == 26 and m == Month.February)
            or (day == 16 and m == Month.March)
            or (day == 28 and m == Month.May)
            or (8 <= day <= 14 and m == Month.September)
            or (day == 17 and m == Month.November)
            or (day == 7 and m == Month.December)
            or (day == 24 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    if y == 2011:
        return (
            (day == 3 and m == Month.February)
            or (day == 15 and m == Month.February)
            or (day == 17 and m == Month.May)
            or (day == 29 and m == Month.June)
            or (day >= 29 and m == Month.August)
            or (day <= 2 and m == Month.September)
            or (day == 26 and m == Month.December)
        )
    if y == 2012:
        return (
            (day == 23 and m == Month.January)
            or (day == 23 and m == Month.March)
            or (20 <= day <= 22 and m == Month.August)
            or (day == 26 and m == Month.October)
            or (15 <= day <= 16 and m == Month.November)
            or (day == 24 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    if y == 2013:
        return (
            (day == 24 and m == Month.January)
            or (day == 12 and m == Month.March)
            or (day == 6 and m == Month.June)
            or (5 <= day <= 9 and m == Month.August)
            or (14 <= day <= 15 and m == Month.October)
            or (day == 5 and m == Month.November)
            or (day == 26 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    if y == 2014:
        return (
            (day == 14 and m == Month.January)
            or (day == 31 and m == Month.January)
            or (day == 31 and m == Month.March)
            or (day == 1 and m == Month.May)
            or (day == 15 and m == Month.May)
            or (day == 27 and m == Month.May)
            or (day == 29 and m == Month.May)
            or ((day >= 28 and m == Month.July) or (day == 1 and m == Month.August))
            or (day == 26 and m == Month.December)
            or (day == 31 and m == Month.December)
        )
    return False
