"""Singapore (SGX) calendar.

# C++ parity: ql/time/calendars/singapore.hpp + .cpp (v1.42.1).

Sat+Sun weekend (C++ ``WesternImpl``). The calendar contains the
standard SGX trading-holiday set plus year-by-year tables of
Chinese New Year / Hari Raya Haji / Vesak Poya Day / Deepavali /
Hari Raya Puasa, matching the C++ table verbatim.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class SingaporeMarket(IntEnum):
    SGX = 0  # Singapore exchange


def _is_lunar_or_minority_holiday(day: int, m: Month, y: int) -> bool:
    # C++ parity: the long list of Chinese New Year / Hari Raya Haji / Vesak / Deepavali /
    # Diwali / Hari Raya Puasa one-shot holidays from ``singapore.cpp``.
    # Chinese New Year
    if day in {22, 23} and m == Month.January and y == 2004:
        return True
    if day in {9, 10} and m == Month.February and y == 2005:
        return True
    if day in {30, 31} and m == Month.January and y == 2006:
        return True
    if day in {19, 20} and m == Month.February and y == 2007:
        return True
    if day in {7, 8} and m == Month.February and y == 2008:
        return True
    if day in {26, 27} and m == Month.January and y == 2009:
        return True
    if day in {15, 16} and m == Month.January and y == 2010:
        return True
    if day in {23, 24} and m == Month.January and y == 2012:
        return True
    if day in {11, 12} and m == Month.February and y == 2013:
        return True
    if day == 31 and m == Month.January and y == 2014:
        return True
    if day == 1 and m == Month.February and y == 2014:
        return True
    # Hari Raya Haji
    if day in {1, 2} and m == Month.February and y == 2004:
        return True
    if day == 21 and m == Month.January and y == 2005:
        return True
    if day == 10 and m == Month.January and y == 2006:
        return True
    if day == 2 and m == Month.January and y == 2007:
        return True
    if day == 20 and m == Month.December and y == 2007:
        return True
    if day == 8 and m == Month.December and y == 2008:
        return True
    if day == 27 and m == Month.November and y == 2009:
        return True
    if day == 17 and m == Month.November and y == 2010:
        return True
    if day == 26 and m == Month.October and y == 2012:
        return True
    if day == 15 and m == Month.October and y == 2013:
        return True
    if day == 6 and m == Month.October and y == 2014:
        return True
    # Vesak Poya Day
    if day == 2 and m == Month.June and y == 2004:
        return True
    if day == 22 and m == Month.May and y == 2005:
        return True
    if day == 12 and m == Month.May and y == 2006:
        return True
    if day == 31 and m == Month.May and y == 2007:
        return True
    if day == 18 and m == Month.May and y == 2008:
        return True
    if day == 9 and m == Month.May and y == 2009:
        return True
    if day == 28 and m == Month.May and y == 2010:
        return True
    if day == 5 and m == Month.May and y == 2012:
        return True
    if day == 24 and m == Month.May and y == 2013:
        return True
    if day == 13 and m == Month.May and y == 2014:
        return True
    # Deepavali
    if day == 11 and m == Month.November and y == 2004:
        return True
    if day == 8 and m == Month.November and y == 2007:
        return True
    if day == 28 and m == Month.October and y == 2008:
        return True
    if day == 16 and m == Month.November and y == 2009:
        return True
    if day == 5 and m == Month.November and y == 2010:
        return True
    if day == 13 and m == Month.November and y == 2012:
        return True
    if day == 2 and m == Month.November and y == 2013:
        return True
    if day == 23 and m == Month.October and y == 2014:
        return True
    # Diwali
    if day == 1 and m == Month.November and y == 2005:
        return True
    # Hari Raya Puasa
    if day in {14, 15} and m == Month.November and y == 2004:
        return True
    if day == 3 and m == Month.November and y == 2005:
        return True
    if day == 24 and m == Month.October and y == 2006:
        return True
    if day == 13 and m == Month.October and y == 2007:
        return True
    if day == 1 and m == Month.October and y == 2008:
        return True
    if day == 21 and m == Month.September and y == 2009:
        return True
    if day == 10 and m == Month.September and y == 2010:
        return True
    if day == 20 and m == Month.August and y == 2012:
        return True
    if day == 8 and m == Month.August and y == 2013:
        return True
    return day == 28 and m == Month.July and y == 2014


def _is_sgx_calendar_override(day: int, m: Month, y: int) -> bool:
    # C++ parity: ``singapore.cpp`` per-year SGX-published trading calendar overrides.
    if y == 2019:
        return (
            (day in {5, 6} and m == Month.February)
            or (day == 20 and m == Month.May)
            or (day == 5 and m == Month.June)
            or (day == 12 and m == Month.August)
            or (day == 28 and m == Month.October)
        )
    if y == 2020:
        return (
            (day == 27 and m == Month.January)
            or (day == 7 and m == Month.May)
            or (day == 25 and m == Month.May)
            or (day == 31 and m == Month.July)
            or (day == 14 and m == Month.November)
        )
    if y == 2021:
        return (
            (day == 12 and m == Month.February)
            or (day == 13 and m == Month.May)
            or (day == 26 and m == Month.May)
            or (day == 20 and m == Month.July)
            or (day == 4 and m == Month.November)
        )
    if y == 2022:
        return (
            (day in {1, 2} and m == Month.February)
            or (day == 2 and m == Month.May)
            or (day == 3 and m == Month.May)
            or (day == 16 and m == Month.May)
            or (day == 11 and m == Month.July)
            or (day == 24 and m == Month.October)
            or (day == 26 and m == Month.December)
        )
    if y == 2023:
        return (
            (day in {23, 24} and m == Month.January)
            or (day == 22 and m == Month.April)
            or (day == 2 and m == Month.June)
            or (day == 29 and m == Month.June)
            or (day == 1 and m == Month.September)
            or (day == 13 and m == Month.November)
        )
    if y == 2024:
        return (
            (day == 12 and m == Month.February)
            or (day == 10 and m == Month.April)
            or (day == 22 and m == Month.May)
            or (day == 17 and m == Month.June)
            or (day == 31 and m == Month.October)
        )
    if y == 2025:
        return (
            (day in {29, 30} and m == Month.January)
            or (day == 31 and m == Month.March)
            or (day == 12 and m == Month.May)
            or (day == 20 and m == Month.October)
        )
    if y == 2026:
        return (
            (day in {17, 18} and m == Month.February)
            or (day == 20 and m == Month.March)
            or (day == 27 and m == Month.May)
            or (day == 1 and m == Month.June)
            or (day == 9 and m == Month.November)
        )
    return False


class Singapore(WesternCalendar):
    """Singapore exchange calendar."""

    def __init__(self, market: SingaporeMarket = SingaporeMarket.SGX) -> None:
        super().__init__()
        # C++ silently accepts any value (the constructor ignores market) — but
        # for diagnostic clarity we reject unknown values explicitly.
        qassert.require(market == SingaporeMarket.SGX, "unknown market")
        self._market: SingaporeMarket = market

    def name(self) -> str:
        return "Singapore exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)

        if (
            self._is_weekend(w)
            # New Year's Day
            or ((day == 1 or (day == 2 and w == Weekday.Monday)) and m == Month.January)
            # Good Friday
            or dd == em - 3
            # Labor Day
            or (day == 1 and m == Month.May)
            # National Day
            or ((day == 9 or (day == 10 and w == Weekday.Monday)) and m == Month.August)
            # Christmas Day
            or (day == 25 and m == Month.December)
        ):
            return False

        if _is_lunar_or_minority_holiday(day, m, y):
            return False

        return not _is_sgx_calendar_override(day, m, y)
