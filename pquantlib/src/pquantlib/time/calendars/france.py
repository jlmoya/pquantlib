"""France — Settlement + Paris Stock Exchange calendars.

# C++ parity: ql/time/calendars/france.hpp + france.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class FranceMarket(IntEnum):
    """France market sub-enum.

    # C++ parity: ``France::Market`` in ql/time/calendars/france.hpp.
    """

    Settlement = 0
    Exchange = 1


class France(WesternCalendar):
    """French calendars (Settlement default, Paris Stock Exchange)."""

    def __init__(self, market: FranceMarket = FranceMarket.Settlement) -> None:
        super().__init__()
        self._market: FranceMarket = market

    def name(self) -> str:
        if self._market == FranceMarket.Settlement:
            return "French settlement"
        if self._market == FranceMarket.Exchange:
            return "Paris stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == FranceMarket.Settlement:
            return self._settlement_is_business_day(d)
        if self._market == FranceMarket.Exchange:
            return self._exchange_is_business_day(d)
        qassert.fail("unknown market")

    def _settlement_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``France::SettlementImpl::isBusinessDay`` in france.cpp.
        w = date.weekday()
        day = date.day_of_month()
        dd = date.day_of_year()
        m = date.month()
        y = date.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Jour de l'An
            or (day == 1 and m == Month.January)
            # Lundi de Paques
            or dd == em
            # Fete du Travail
            or (day == 1 and m == Month.May)
            # Victoire 1945
            or (day == 8 and m == Month.May)
            # Ascension
            or (day == 10 and m == Month.May)
            # Pentecote
            or (day == 21 and m == Month.May)
            # Fete nationale
            or (day == 14 and m == Month.July)
            # Assomption
            or (day == 15 and m == Month.August)
            # Toussaint
            or (day == 1 and m == Month.November)
            # Armistice 1918
            or (day == 11 and m == Month.November)
            # Noel
            or (day == 25 and m == Month.December)
        )

    def _exchange_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``France::ExchangeImpl::isBusinessDay`` in france.cpp.
        w = date.weekday()
        day = date.day_of_month()
        dd = date.day_of_year()
        m = date.month()
        y = date.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # Jour de l'An
            or (day == 1 and m == Month.January)
            # Good Friday
            or dd == em - 3
            # Easter Monday
            or dd == em
            # Labor Day
            or (day == 1 and m == Month.May)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas Day
            or (day == 25 and m == Month.December)
            # Boxing Day
            or (day == 26 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )
