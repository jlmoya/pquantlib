"""Italy — Settlement + Milan Stock Exchange calendars.

# C++ parity: ql/time/calendars/italy.hpp + italy.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class ItalyMarket(IntEnum):
    """Italy market sub-enum.

    # C++ parity: ``Italy::Market`` in ql/time/calendars/italy.hpp.
    """

    Settlement = 0
    Exchange = 1


class Italy(WesternCalendar):
    """Italian calendars (Settlement default, Milan Stock Exchange)."""

    def __init__(self, market: ItalyMarket = ItalyMarket.Settlement) -> None:
        super().__init__()
        self._market: ItalyMarket = market

    def name(self) -> str:
        if self._market == ItalyMarket.Settlement:
            return "Italian settlement"
        if self._market == ItalyMarket.Exchange:
            return "Milan stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == ItalyMarket.Settlement:
            return self._settlement_is_business_day(d)
        if self._market == ItalyMarket.Exchange:
            return self._exchange_is_business_day(d)
        qassert.fail("unknown market")

    def _settlement_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Italy::SettlementImpl::isBusinessDay`` in italy.cpp.
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
            # Easter Monday
            or dd == em
            # Liberation Day
            or (day == 25 and m == Month.April)
            # Labour Day
            or (day == 1 and m == Month.May)
            # Republic Day
            or (day == 2 and m == Month.June and y >= 2000)
            # Assumption
            or (day == 15 and m == Month.August)
            # All Saints' Day
            or (day == 1 and m == Month.November)
            # Immaculate Conception
            or (day == 8 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # December 31st, 1999 only
            or (day == 31 and m == Month.December and y == 1999)
        )

    def _exchange_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Italy::ExchangeImpl::isBusinessDay`` in italy.cpp.
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
            # Good Friday
            or dd == em - 3
            # Easter Monday
            or dd == em
            # Labour Day
            or (day == 1 and m == Month.May)
            # Assumption
            or (day == 15 and m == Month.August)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )
