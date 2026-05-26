"""Chilean calendar — Santiago Stock Exchange (SSE).

# C++ parity: ql/time/calendars/chile.hpp + chile.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday

# C++ parity: ql/time/calendars/chile.cpp anonymous-namespace
# ``aboriginalPeopleDay`` table, years 2021..2199 (179 entries).
# Day of the Winter Solstice (usually June 20 or 21) when the Day of
# Aboriginal People is observed.
_ABORIGINAL_PEOPLE_DAY: Final[tuple[int, ...]] = (
    21,
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,  # 2021-2029
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,
    21,
    21,  # 2030-2039
    20,
    20,
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,  # 2040-2049
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    21,  # 2050-2059
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,  # 2060-2069
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    20,  # 2070-2079
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,  # 2080-2089
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,  # 2090-2099
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    21,
    20,
    21,  # 2100-2109
    21,
    21,
    20,
    21,
    21,
    21,
    20,
    21,
    21,
    21,  # 2110-2119
    20,
    21,
    21,
    21,
    20,
    21,
    21,
    21,
    20,
    21,  # 2120-2129
    21,
    21,
    20,
    21,
    21,
    21,
    20,
    20,
    21,
    21,  # 2130-2139
    20,
    20,
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,  # 2140-2149
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,
    21,
    21,  # 2150-2159
    20,
    20,
    21,
    21,
    20,
    20,
    21,
    21,
    20,
    20,  # 2160-2169
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    21,  # 2170-2179
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,  # 2180-2189
    20,
    21,
    20,
    20,
    20,
    21,
    20,
    20,
    20,
    20,  # 2190-2199
)


def _is_aboriginal_people_day(day: int, m: Month, y: int) -> bool:
    """C++ parity: ql/time/calendars/chile.cpp ``isAboriginalPeopleDay``."""
    return m == Month.June and y >= 2021 and day == _ABORIGINAL_PEOPLE_DAY[y - 2021]


class Chile(WesternCalendar):
    """Chilean calendar (Santiago Stock Exchange / SSE)."""

    class Market(IntEnum):
        SSE = 0  # Santiago Stock Exchange

    def __init__(self, market: Market = Market.SSE) -> None:
        super().__init__()
        qassert.require(market == Chile.Market.SSE, "unknown market")
        self._market = market

    def name(self) -> str:
        return "Santiago Stock Exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        m = d.month()
        y = d.year()
        dd = d.day_of_year()
        em = WesternCalendar.easter_monday(y)
        return not (
            # New Year's Day
            (day == 1 and m == Month.January)
            or (day == 2 and m == Month.January and w == Weekday.Monday and y > 2016)
            # Papal visit in 2018
            or (day == 16 and m == Month.January and y == 2018)
            # Good Friday
            or (dd == em - 3)
            # Easter Saturday
            or (dd == em - 2)
            # Census Day in 2017
            or (day == 19 and m == Month.April and y == 2017)
            # Labour Day
            or (day == 1 and m == Month.May)
            # Navy Day
            or (day == 21 and m == Month.May)
            # Day of Aboriginal People
            or _is_aboriginal_people_day(day, m, y)
            # St. Peter and St. Paul
            or (26 <= day <= 29 and m == Month.June and w == Weekday.Monday)
            or (day == 2 and m == Month.July and w == Weekday.Monday)
            # Our Lady of Mount Carmel
            or (day == 16 and m == Month.July)
            # Assumption Day
            or (day == 15 and m == Month.August)
            # Independence Day
            or (day == 16 and m == Month.September and y == 2022)
            or (
                day == 17
                and m == Month.September
                and ((w == Weekday.Monday and y >= 2007) or (w == Weekday.Friday and y > 2016))
            )
            or (day == 18 and m == Month.September)
            # Army Day
            or (day == 19 and m == Month.September)
            or (day == 20 and m == Month.September and w == Weekday.Friday and y >= 2007)
            # Discovery of Two Worlds
            or (9 <= day <= 12 and m == Month.October and w == Weekday.Monday)
            or (day == 15 and m == Month.October and w == Weekday.Monday)
            # Reformation Day
            or (
                (
                    (day == 27 and m == Month.October and w == Weekday.Friday)
                    or (day == 31 and m == Month.October and w not in (Weekday.Tuesday, Weekday.Wednesday))
                    or (day == 2 and m == Month.November and w == Weekday.Friday)
                )
                and y >= 2008
            )
            # All Saints' Day
            or (day == 1 and m == Month.November)
            # Immaculate Conception
            or (day == 8 and m == Month.December)
            # Christmas Day
            or (day == 25 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )
