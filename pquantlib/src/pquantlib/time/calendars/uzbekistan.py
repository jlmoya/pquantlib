"""Uzbekistan Stock Exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/uzbekistan.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``. Uses the tabulated
moon-sighting Islamic holidays (see ``islamic_holidays``).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.calendars.islamic_holidays import is_eid_al_adha, is_eid_al_fitr
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class UzbekistanMarket(IntEnum):
    """Uzbekistan market sub-enum."""

    UZSE = 0


class Uzbekistan(WesternCalendar):
    """Uzbekistan Stock Exchange calendar."""

    def __init__(self, market: UzbekistanMarket = UzbekistanMarket.UZSE) -> None:
        super().__init__()
        self._market: UzbekistanMarket = market

    def name(self) -> str:
        return "Uzbekistan Stock Exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Uzbekistan::Impl2::isBusinessDay`` in uzbekistan.cpp.
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        return not (
            self._is_weekend(w)
            or is_eid_al_fitr(d)
            or is_eid_al_adha(d)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # International Women's Day
            or (day == 8 and m == Month.March)
            # Navruz (Persian New Year)
            or (day == 21 and m == Month.March)
            # Day of Remembrance and Honors
            or (day == 9 and m == Month.May)
            # Independence Day
            or (day == 1 and m == Month.September)
            # Teachers Day
            or (day == 1 and m == Month.October)
            # Constitution Day
            or (day == 8 and m == Month.December)
        )
