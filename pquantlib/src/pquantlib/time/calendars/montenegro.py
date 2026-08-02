"""Montenegro Stock Exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/montenegro.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class MontenegroMarket(IntEnum):
    """Montenegro market sub-enum."""

    MNSE = 0


class Montenegro(WesternCalendar):
    """Montenegro Stock Exchange calendar."""

    def __init__(self, market: MontenegroMarket = MontenegroMarket.MNSE) -> None:
        super().__init__()
        self._market: MontenegroMarket = market

    def name(self) -> str:
        return "Montenegro Stock Exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Montenegro::MnseImpl::isBusinessDay`` in montenegro.cpp.
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        return not (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # New Year Holiday
            or (day == 2 and m == Month.January)
            # Labour Day
            or (day == 1 and m == Month.May)
            # Labour Day Holiday
            or (day == 2 and m == Month.May)
            # Independence Day
            or (day == 21 and m == Month.May)
            # Independence Day Holiday
            or (day == 22 and m == Month.May)
            # Statehood Day
            or (day == 13 and m == Month.July)
            # Statehood Day Holiday
            or (day == 14 and m == Month.July)
            # Njegos Day
            or (day == 13 and m == Month.November)
            # Njegos Day Holiday
            or (day == 14 and m == Month.November)
        )
