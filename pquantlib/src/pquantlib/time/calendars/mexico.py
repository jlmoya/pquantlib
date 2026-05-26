"""Mexico — Mexican Stock Exchange (BMV) calendar.

# C++ parity: ql/time/calendars/mexico.hpp + mexico.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class MexicoMarket(IntEnum):
    """Mexico market sub-enum.

    # C++ parity: ``Mexico::Market`` in ql/time/calendars/mexico.hpp.
    """

    BMV = 0


class Mexico(WesternCalendar):
    """Mexican calendar — Bolsa Mexicana de Valores (BMV)."""

    def __init__(self, market: MexicoMarket = MexicoMarket.BMV) -> None:
        super().__init__()
        self._market: MexicoMarket = market

    def name(self) -> str:
        if self._market == MexicoMarket.BMV:
            return "Mexican stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == MexicoMarket.BMV:
            return self._bmv_is_business_day(d)
        qassert.fail("unknown market")

    def _bmv_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Mexico::BmvImpl::isBusinessDay`` in mexico.cpp.
        w = date.weekday()
        day = date.day_of_month()
        dd = date.day_of_year()
        m = date.month()
        y = date.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Constitution Day
            or (y <= 2005 and day == 5 and m == Month.February)
            or (y >= 2006 and day <= 7 and w == Weekday.Monday and m == Month.February)
            # Birthday of Benito Juarez
            or (y <= 2005 and day == 21 and m == Month.March)
            or (y >= 2006 and 15 <= day <= 21 and w == Weekday.Monday and m == Month.March)
            # Holy Thursday
            or dd == em - 4
            # Good Friday
            or dd == em - 3
            # Labour Day
            or (day == 1 and m == Month.May)
            # National Day
            or (day == 16 and m == Month.September)
            # Inauguration Day (every sixth year starting 2024)
            or (day == 1 and m == Month.October and y >= 2024 and (y - 2024) % 6 == 0)
            # All Souls Day
            or (day == 2 and m == Month.November)
            # Revolution Day
            or (y <= 2005 and day == 20 and m == Month.November)
            or (y >= 2006 and 15 <= day <= 21 and w == Weekday.Monday and m == Month.November)
            # Our Lady of Guadalupe
            or (day == 12 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
        )
