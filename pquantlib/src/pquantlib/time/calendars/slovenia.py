"""Ljubljana stock exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/slovenia.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class SloveniaMarket(IntEnum):
    """Slovenia market sub-enum."""

    LSE = 0


class Slovenia(WesternCalendar):
    """Ljubljana stock exchange calendar."""

    def __init__(self, market: SloveniaMarket = SloveniaMarket.LSE) -> None:
        super().__init__()
        self._market: SloveniaMarket = market

    def name(self) -> str:
        return "Ljubljana stock exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Slovenia::LseImpl::isBusinessDay`` in slovenia.cpp.
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
            # New Year's Holiday
            or (day == 2 and m == Month.January)
            # Good Friday
            or dd == em - 3
            # Easter Monday
            or dd == em
            # May Day
            or (day == 1 and m == Month.May)
            # May Day Holiday
            or (day == 2 and m == Month.May)
            # Statehood Day
            or (day == 25 and m == Month.June)
            # Assumption of Mary
            or (day == 15 and m == Month.August)
            # Reformation Day
            or (day == 31 and m == Month.October)
            # Christmas Eve
            or (day == 24 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # St. Stephen
            or (day == 26 and m == Month.December)
            # New Year's Eve
            or (day == 31 and m == Month.December)
        )
