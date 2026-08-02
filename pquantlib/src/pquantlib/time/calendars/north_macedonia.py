"""Macedonian Stock Exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/northmacedonia.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``. Uses the tabulated
moon-sighting Islamic holidays (see ``islamic_holidays``).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import OrthodoxCalendar
from pquantlib.time.calendars.islamic_holidays import is_eid_al_adha, is_eid_al_fitr
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class NorthMacedoniaMarket(IntEnum):
    """NorthMacedonia market sub-enum."""

    MSE = 0


class NorthMacedonia(OrthodoxCalendar):
    """Macedonian Stock Exchange calendar."""

    def __init__(self, market: NorthMacedoniaMarket = NorthMacedoniaMarket.MSE) -> None:
        super().__init__()
        self._market: NorthMacedoniaMarket = market

    def name(self) -> str:
        return "Macedonian Stock Exchange"

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``NorthMacedonia::MseImpl::isBusinessDay`` in northmacedonia.cpp.
        w = d.weekday()
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        # C++ declares MseImpl : public Calendar::OrthodoxImpl — North Macedonia is
        # an Orthodox country, so Easter Monday uses the Orthodox computation.
        em = OrthodoxCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            or is_eid_al_fitr(d)
            or is_eid_al_adha(d)
            # New Year
            or (day == 1 and m == Month.January)
            # Orthodox Christmas
            or (day == 7 and m == Month.January)
            # Easter Monday
            or (dd == em)
            # Labour Day
            or (day == 1 and m == Month.May)
            # Saints Cyril and Methodius Day
            or (day == 24 and m == Month.May)
            # Republic Day
            or (day == 2 and m == Month.August)
            # Independence Day
            or (day == 8 and m == Month.September)
            # Day of the People's Uprising
            or (day == 11 and m == Month.October)
            # Day of the Macedonian Revolutionary Struggle
            or (day == 23 and m == Month.October)
            # Saint Clement of Ohrid Day
            or (day == 8 and m == Month.December)
        )
