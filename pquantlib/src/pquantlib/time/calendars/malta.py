"""Malta Stock Exchange calendar — new in C++ QuantLib v1.43.

Faithful port of ``ql/time/calendars/malta.{hpp,cpp}`` from QuantLib
v1.43 @ ``6b57206e04598f092efee66e3b367efc84771995``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class MaltaMarket(IntEnum):
    """Malta market sub-enum."""

    MSE = 0


class Malta(WesternCalendar):
    """Malta Stock Exchange calendar."""

    def __init__(self, market: MaltaMarket = MaltaMarket.MSE) -> None:
        super().__init__()
        self._market: MaltaMarket = market

    def name(self) -> str:
        return "Malta Stock Exchange"

    def _is_weekend(self, w: Weekday) -> bool:
        # C++ ``Malta::MseImpl::isWeekend`` overrides the Western
        # Saturday+Sunday weekend with Friday+Saturday, even though the impl
        # derives from ``Calendar::WesternImpl``. Almost certainly an upstream
        # quirk -- Malta observes a Saturday/Sunday weekend -- but v1.43
        # defines it this way and C++ is the ground truth, so mirror it.
        return w in (Weekday.Friday, Weekday.Saturday)

    def _is_business_day(self, d: Date) -> bool:
        # C++ parity: ``Malta::MseImpl::isBusinessDay`` in malta.cpp.
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
            # St. Paul's Shipwreck
            or (day == 10 and m == Month.February)
            # St. Joseph's Day
            or (day == 19 and m == Month.March)
            # Freedom Day
            or (day == 31 and m == Month.March)
            # Good Friday
            or dd == em - 3
            # Easter Monday (exchange holiday)
            or dd == em
            # Labour Day
            or (day == 1 and m == Month.May)
            # Imnarja (Sts Peter & Paul)
            or (day == 29 and m == Month.June)
            # Assumption of Mary
            or (day == 15 and m == Month.August)
            # Our Lady of Victories
            or (day == 8 and m == Month.September)
            # Independence Day
            or (day == 21 and m == Month.September)
            # Immaculate Conception
            or (day == 8 and m == Month.December)
            # Republic Day
            or (day == 13 and m == Month.December)
            # Christmas Vigil
            or (day == 24 and m == Month.December)
            # Christmas Day
            or (day == 25 and m == Month.December)
            # Boxing Day
            or (day == 26 and m == Month.December)
            # New Year's Eve (non-trading)
            or (day == 31 and m == Month.December)
        )
