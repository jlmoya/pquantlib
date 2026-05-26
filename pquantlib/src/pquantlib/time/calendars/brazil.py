"""Brazilian calendar — Settlement / BOVESPA Exchange markets.

# C++ parity: ql/time/calendars/brazil.hpp + brazil.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Brazil(WesternCalendar):
    """Brazilian calendar (Settlement default, or BOVESPA exchange)."""

    class Market(IntEnum):
        Settlement = 0  # generic settlement calendar
        Exchange = 1  # BOVESPA calendar

    def __init__(self, market: Market = Market.Settlement) -> None:
        super().__init__()
        self._market = market

    def name(self) -> str:
        if self._market == Brazil.Market.Settlement:
            return "Brazil"
        if self._market == Brazil.Market.Exchange:
            return "BOVESPA"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == Brazil.Market.Settlement:
            return self._is_business_day_settlement(d)
        if self._market == Brazil.Market.Exchange:
            return self._is_business_day_exchange(d)
        qassert.fail("unknown market")

    def _is_business_day_settlement(self, d: Date) -> bool:
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
            # Tiradentes Day
            or (day == 21 and m == Month.April)
            # Labor Day
            or (day == 1 and m == Month.May)
            # Independence Day
            or (day == 7 and m == Month.September)
            # Nossa Sra. Aparecida Day
            or (day == 12 and m == Month.October)
            # All Souls Day
            or (day == 2 and m == Month.November)
            # Republic Day
            or (day == 15 and m == Month.November)
            # Black Awareness Day (since 2024)
            or (day == 20 and m == Month.November and y >= 2024)
            # Christmas
            or (day == 25 and m == Month.December)
            # Passion of Christ
            or (dd == em - 3)
            # Carnival
            or (dd in (em - 49, em - 48))
            # Corpus Christi
            or (dd == em + 59)
        )

    def _is_business_day_exchange(self, d: Date) -> bool:
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
            # Sao Paulo City Day (up to 2021 included)
            or (day == 25 and m == Month.January and y < 2022)
            # Tiradentes Day
            or (day == 21 and m == Month.April)
            # Labor Day
            or (day == 1 and m == Month.May)
            # Revolution Day (up to 2021 included)
            or (day == 9 and m == Month.July and y < 2022)
            # Independence Day
            or (day == 7 and m == Month.September)
            # Nossa Sra. Aparecida Day
            or (day == 12 and m == Month.October)
            # All Souls Day
            or (day == 2 and m == Month.November)
            # Republic Day
            or (day == 15 and m == Month.November)
            # Black Consciousness Day (since 2007, except 2022 and 2023)
            or (day == 20 and m == Month.November and y >= 2007 and y not in (2022, 2023))
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # Passion of Christ
            or (dd == em - 3)
            # Carnival
            or (dd in (em - 49, em - 48))
            # Corpus Christi
            or (dd == em + 59)
            # last business day of the year
            or (m == Month.December and (day == 31 or (day >= 29 and w == Weekday.Friday)))
        )
