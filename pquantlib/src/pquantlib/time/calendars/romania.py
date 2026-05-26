"""Romania — Public + Bucharest Stock Exchange (BVB) calendars.

# C++ parity: ql/time/calendars/romania.hpp + romania.cpp (v1.42.1).

Romania is an Orthodox-Easter calendar (Easter Monday from the Orthodox
table), so this calendar extends ``OrthodoxCalendar`` rather than
``WesternCalendar``. This matches the C++ source where ``PublicImpl``
derives from ``Calendar::OrthodoxImpl``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import OrthodoxCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


class RomaniaMarket(IntEnum):
    """Romania market sub-enum.

    # C++ parity: ``Romania::Market`` in ql/time/calendars/romania.hpp.
    """

    Public = 0
    BVB = 1


class Romania(OrthodoxCalendar):
    """Romanian calendars (Public, BVB default — Bucharest Stock Exchange)."""

    def __init__(self, market: RomaniaMarket = RomaniaMarket.BVB) -> None:
        super().__init__()
        self._market: RomaniaMarket = market

    def name(self) -> str:
        if self._market == RomaniaMarket.Public:
            return "Romania"
        if self._market == RomaniaMarket.BVB:
            return "Bucharest stock exchange"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if self._market == RomaniaMarket.Public:
            return self._public_is_business_day(d)
        if self._market == RomaniaMarket.BVB:
            return self._bvb_is_business_day(d)
        qassert.fail("unknown market")

    def _public_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Romania::PublicImpl::isBusinessDay`` in romania.cpp.
        w = date.weekday()
        day = date.day_of_month()
        dd = date.day_of_year()
        m = date.month()
        y = date.year()
        em = OrthodoxCalendar.easter_monday(y)
        return not (
            self._is_weekend(w)
            # New Year's Day
            or (day == 1 and m == Month.January)
            # Day after New Year's Day
            or (day == 2 and m == Month.January)
            # Unification Day
            or (day == 24 and m == Month.January)
            # Orthodox Easter Monday
            or dd == em
            # Labour Day
            or (day == 1 and m == Month.May)
            # Pentecost
            or dd == em + 49
            # Children's Day (since 2017)
            or (day == 1 and m == Month.June and y >= 2017)
            # St Marys Day
            or (day == 15 and m == Month.August)
            # Feast of St Andrew
            or (day == 30 and m == Month.November)
            # National Day
            or (day == 1 and m == Month.December)
            # Christmas
            or (day == 25 and m == Month.December)
            # 2nd Day of Christmas
            or (day == 26 and m == Month.December)
        )

    def _bvb_is_business_day(self, date: Date) -> bool:
        # C++ parity: ``Romania::BVBImpl::isBusinessDay`` in romania.cpp.
        if not self._public_is_business_day(date):
            return False
        day = date.day_of_month()
        m = date.month()
        y = date.year()
        # one-off closing days
        return not (
            (day == 24 and m == Month.December and y == 2014)
            or (day == 31 and m == Month.December and y == 2014)
        )
