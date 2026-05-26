"""Slovakia — Bratislava Stock Exchange (BSSE) calendar.

# C++ parity: ql/time/calendars/slovakia.hpp + slovakia.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class SlovakiaMarket(IntEnum):
    """Slovakia market sub-enum.

    # C++ parity: ``Slovakia::Market`` in ql/time/calendars/slovakia.hpp.
    """

    BSSE = 0


class Slovakia(WesternCalendar):
    """Slovak calendar — Bratislava Stock Exchange (BSSE)."""

    def __init__(self, market: SlovakiaMarket = SlovakiaMarket.BSSE) -> None:
        super().__init__()
        self._market: SlovakiaMarket = market

    def name(self) -> str:
        if self._market == SlovakiaMarket.BSSE:
            return "Bratislava stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == SlovakiaMarket.BSSE:
            return self._bsse_is_business_day(d)
        qassert.fail("unknown market")

    def _bsse_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Slovakia::BsseImpl::isBusinessDay`` in slovakia.cpp.
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
            # Epiphany
            or (day == 6 and m == Month.January)
            # Good Friday
            or dd == em - 3
            # Easter Monday
            or dd == em
            # May Day
            or (day == 1 and m == Month.May)
            # Liberation of the Republic
            or (day == 8 and m == Month.May)
            # SS. Cyril and Methodius
            or (day == 5 and m == Month.July)
            # Slovak National Uprising
            or (day == 29 and m == Month.August)
            # Constitution of the Slovak Republic
            or (day == 1 and m == Month.September)
            # Our Lady of the Seven Sorrows
            or (day == 15 and m == Month.September)
            # All Saints Day
            or (day == 1 and m == Month.November)
            # Freedom and Democracy of the Slovak Republic
            or (day == 17 and m == Month.November)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # unidentified closing days for stock exchange
            or (24 <= day <= 31 and m == Month.December and y == 2004)
            or (24 <= day <= 31 and m == Month.December and y == 2005)
        )
