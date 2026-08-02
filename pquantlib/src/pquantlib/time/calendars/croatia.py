"""Zagreb stock exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/croatia.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class CroatiaMarket(IntEnum):
    """Croatia market sub-enum."""

    ZSE = 0


class Croatia(WesternCalendar):
    """Zagreb stock exchange calendar."""

    def __init__(self, market: CroatiaMarket = CroatiaMarket.ZSE) -> None:
        super().__init__()
        self._market: CroatiaMarket = market

    def name(self) -> str:
        return "Zagreb stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Croatia::ZseImpl::isBusinessDay`` in croatia.cpp.
        w = d.weekday()
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Epiphany
            or (day == 6 and m == Month.January)
            # Good Friday
            or (dd == em - 3 and y >= 2016)
            # Easter Monday
            or dd == em
            # Labour Day
            or (day == 1 and m == Month.May)
            # National Day
            or (day == 30 and m == Month.May)
            # Corpus Christi
            or dd == em + 59
            # Anti-Fascist Struggle Day
            or (day == 22 and m == Month.June)
            # Victory and Homeland Thanksgiving Day
            or (day == 5 and m == Month.August)
            # Assumption of Mary
            or (day == 15 and m == Month.August)
            # Remembrance Day (Vukovar and Skabrnja)
            or (day == 18 and m == Month.November)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )
