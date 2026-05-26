"""Austria — Settlement + Vienna Stock Exchange calendars.

# C++ parity: ql/time/calendars/austria.hpp + austria.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class AustriaMarket(IntEnum):
    """Austria market sub-enum.

    # C++ parity: ``Austria::Market`` in ql/time/calendars/austria.hpp.
    """

    Settlement = 0
    Exchange = 1


class Austria(WesternCalendar):
    """Austrian calendars (Settlement default, Vienna Stock Exchange)."""

    def __init__(self, market: AustriaMarket = AustriaMarket.Settlement) -> None:
        super().__init__()
        self._market: AustriaMarket = market

    def name(self) -> str:
        if self._market == AustriaMarket.Settlement:
            return "Austrian settlement"
        if self._market == AustriaMarket.Exchange:
            return "Vienna stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == AustriaMarket.Settlement:
            return self._settlement_is_business_day(d)
        if self._market == AustriaMarket.Exchange:
            return self._exchange_is_business_day(d)
        qassert.fail("unknown market")

    def _settlement_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Austria::SettlementImpl::isBusinessDay`` in austria.cpp.
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
            # Ascension Thursday
            or dd == em + 38
            # Whit Monday
            or dd == em + 49
            # Corpus Christi
            or dd == em + 59
            # Labour Day
            or (day == 1 and m == Month.May)
            # Assumption
            or (day == 15 and m == Month.August)
            # National Holiday since 1967
            or (day == 26 and m == Month.October and y >= 1967)
            # National Holiday 1919-1934
            or (day == 12 and m == Month.November and 1919 <= y <= 1934)
            # All Saints' Day
            or (day == 1 and m == Month.November)
            # Immaculate Conception
            or (day == 8 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
        )

    def _exchange_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Austria::ExchangeImpl::isBusinessDay`` in austria.cpp.
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
            # Whit Monday
            or dd == em + 49
            # Labour Day
            or (day == 1 and m == Month.May)
            # National Holiday since 1967
            or (day == 26 and m == Month.October and y >= 1967)
            # National Holiday 1919-1934
            or (day == 12 and m == Month.November and 1919 <= y <= 1934)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # Exchange Holiday
            or (day == 31 and m == Month.December)
        )
