"""Belgrade stock exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/serbia.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class SerbiaMarket(IntEnum):
    """Serbia market sub-enum."""

    BSE = 0


class Serbia(WesternCalendar):
    """Belgrade stock exchange calendar."""

    def __init__(self, market: SerbiaMarket = SerbiaMarket.BSE) -> None:
        super().__init__()
        self._market: SerbiaMarket = market

    def name(self) -> str:
        return "Belgrade stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Serbia::BseImpl::isBusinessDay`` in serbia.cpp.
        w = d.weekday()
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year
            or (day == 1 and m == Month.January)
            # New Year Holiday
            or (day == 2 and m == Month.January)
            # Serbian Orthodox Christmas
            or (day == 7 and m == Month.January)
            # Statehood Day
            or (day == 15 and m == Month.February)
            # Statehood Day (2nd)
            or (day == 16 and m == Month.February)
            # Statehood Day observed (when 15+16 Feb both fall at a weekend)
            or (
                day == 17
                and m == Month.February
                and self._is_weekend(Date.from_ymd(15, Month.February, y).weekday())
                and self._is_weekend(Date.from_ymd(16, Month.February, y).weekday())
            )
            # Good Friday
            or (dd == em - 3 and y >= 2016)
            # Easter Monday
            or dd == em
            # Labour Day
            or (day == 1 and m == Month.May)
            # Armistice Day in World War I
            or (day == 11 and m == Month.November)
            # Trading system maintenance
            or (day == 31 and m == Month.December)
        )
