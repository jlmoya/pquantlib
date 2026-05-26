"""Germany — Settlement + Frankfurt/Xetra/Eurex/Euwax exchange calendars.

# C++ parity: ql/time/calendars/germany.hpp + germany.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class GermanyMarket(IntEnum):
    """Germany market sub-enum.

    # C++ parity: ``Germany::Market`` in ql/time/calendars/germany.hpp.
    """

    Settlement = 0
    FrankfurtStockExchange = 1
    Xetra = 2
    Eurex = 3
    Euwax = 4


class Germany(WesternCalendar):
    """German calendars (Settlement, Frankfurt SE default, Xetra, Eurex, Euwax)."""

    def __init__(
        self,
        market: GermanyMarket = GermanyMarket.FrankfurtStockExchange,
    ) -> None:
        super().__init__()
        self._market: GermanyMarket = market

    def name(self) -> str:
        if self._market == GermanyMarket.Settlement:
            return "German settlement"
        if self._market == GermanyMarket.FrankfurtStockExchange:
            return "Frankfurt stock exchange"
        if self._market == GermanyMarket.Xetra:
            return "Xetra"
        if self._market == GermanyMarket.Eurex:
            return "Eurex"
        if self._market == GermanyMarket.Euwax:
            return "Euwax"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == GermanyMarket.Settlement:
            return self._settlement_is_business_day(d)
        if self._market == GermanyMarket.FrankfurtStockExchange:
            return self._frankfurt_is_business_day(d)
        if self._market == GermanyMarket.Xetra:
            return self._xetra_is_business_day(d)
        if self._market == GermanyMarket.Eurex:
            return self._eurex_is_business_day(d)
        if self._market == GermanyMarket.Euwax:
            return self._euwax_is_business_day(d)
        qassert.fail("unknown market")

    def _settlement_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Germany::SettlementImpl::isBusinessDay`` in germany.cpp.
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
            # Ascension Thursday
            or dd == em + 38
            # Whit Monday
            or dd == em + 49
            # Corpus Christi
            or dd == em + 59
            # Labour Day
            or (day == 1 and m == Month.May)
            # National Day
            or (day == 3 and m == Month.October)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Boxing Day
            or (day == 26 and m == Month.December)
        )

    def _frankfurt_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Germany::FrankfurtStockExchangeImpl::isBusinessDay``.
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
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Christmas Day (Boxing Day)
            or (day == 26 and m == Month.December)
        )

    def _xetra_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Germany::XetraImpl::isBusinessDay``.
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
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Christmas Day (Boxing Day)
            or (day == 26 and m == Month.December)
        )

    def _eurex_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Germany::EurexImpl::isBusinessDay``.
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
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Christmas Day (Boxing Day)
            or (day == 26 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )

    def _euwax_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Germany::EuwaxImpl::isBusinessDay``.
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
            # Whit Monday
            or dd == em + 49
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Christmas Day (Boxing Day)
            or (day == 26 and m == Month.December)
        )
