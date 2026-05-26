"""Argentinian calendars — Buenos Aires stock exchange (Merval).

# C++ parity: ql/time/calendars/argentina.hpp + argentina.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class Argentina(WesternCalendar):
    """Argentinian calendar (Buenos Aires stock exchange / Merval)."""

    class Market(IntEnum):
        Merval = 0  # Buenos Aires stock exchange calendar

    def __init__(self, market: Market = Market.Merval) -> None:
        super().__init__()
        qassert.require(market == Argentina.Market.Merval, "unknown market")
        self._market = market

    def name(self) -> str:
        return "Buenos Aires stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            # New Year's Day
            (day == 1 and m == Month.January)
            # Holy Thursday
            or (dd == em - 4)
            # Good Friday
            or (dd == em - 3)
            # Labour Day
            or (day == 1 and m == Month.May)
            # May Revolution
            or (day == 25 and m == Month.May)
            # Death of General Manuel Belgrano (third Monday in June)
            or (15 <= day <= 21 and w == Weekday.Monday and m == Month.June)
            # Independence Day
            or (day == 9 and m == Month.July)
            # Death of General Jose de San Martin (third Monday in August)
            or (15 <= day <= 21 and w == Weekday.Monday and m == Month.August)
            # Columbus Day
            or (day in (10, 11, 12, 15, 16) and w == Weekday.Monday and m == Month.October)
            # Immaculate Conception
            or (day == 8 and m == Month.December)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # New Year's Eve
            or ((day == 31 or (day == 30 and w == Weekday.Friday)) and m == Month.December)
        )
