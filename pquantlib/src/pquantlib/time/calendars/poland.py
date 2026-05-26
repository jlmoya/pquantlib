"""Poland — Settlement + Warsaw Stock Exchange (WSE) calendars.

# C++ parity: ql/time/calendars/poland.hpp + poland.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class PolandMarket(IntEnum):
    """Poland market sub-enum.

    # C++ parity: ``Poland::Market`` in ql/time/calendars/poland.hpp.
    """

    Settlement = 0
    WSE = 1


class Poland(WesternCalendar):
    """Polish calendars (Settlement default, WSE — Warsaw Stock Exchange)."""

    def __init__(self, market: PolandMarket = PolandMarket.Settlement) -> None:
        super().__init__()
        self._market: PolandMarket = market

    def name(self) -> str:
        if self._market == PolandMarket.Settlement:
            return "Poland Settlement"
        if self._market == PolandMarket.WSE:
            return "Warsaw stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == PolandMarket.Settlement:
            return self._settlement_is_business_day(d)
        if self._market == PolandMarket.WSE:
            return self._wse_is_business_day(d)
        qassert.fail("unknown market")

    def _settlement_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Poland::SettlementImpl::isBusinessDay`` in poland.cpp.
        w = date.weekday()
        day = date.day_of_month()
        dd = date.day_of_year()
        m = date.month()
        y = date.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Easter Monday
            or dd == em
            # Corpus Christi
            or dd == em + 59
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Epiphany (since 2011)
            or (day == 6 and m == Month.January and y >= 2011)
            # May Day
            or (day == 1 and m == Month.May)
            # Constitution Day
            or (day == 3 and m == Month.May)
            # Assumption of the Blessed Virgin Mary
            or (day == 15 and m == Month.August)
            # All Saints Day
            or (day == 1 and m == Month.November)
            # Independence Day
            or (day == 11 and m == Month.November)
            # Christmas
            or (day == 25 and m == Month.December)
            # 2nd Day of Christmas
            or (day == 26 and m == Month.December)
        )

    def _wse_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Poland::WseImpl::isBusinessDay`` in poland.cpp.
        # Additional WSE-only holidays: Dec 24 & Dec 31, then delegate.
        day = date.day_of_month()
        m = date.month()
        if (day == 24 and m == Month.December) or (day == 31 and m == Month.December):
            return False
        return self._settlement_is_business_day(date)
