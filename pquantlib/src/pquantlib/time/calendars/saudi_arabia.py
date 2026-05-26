"""Saudi Arabia (Tadawul) calendar.

# C++ parity: ql/time/calendars/saudiarabia.hpp + .cpp (v1.42.1).

Saudi Arabia uses a Friday+Saturday weekend (NOT Sat+Sun), so the
calendar inherits ``Calendar`` directly rather than ``WesternCalendar``
and provides its own ``_is_weekend``. The ``_is_business_day`` rule
additionally consults a date-dependent "true weekend" check because the
Saudi weekend changed from (Thursday, Friday) to (Friday, Saturday) on
29-June-2013 — see C++ anonymous-namespace ``isTrueWeekend``.

Eid Al-Adha / Eid Al-Fitr dates are hard-coded tables taken straight
from the C++ source.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from pquantlib import qassert
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class SaudiArabiaMarket(IntEnum):
    Tadawul = 0  # Tadawul financial market


# C++ parity: ql/time/calendars/saudiarabia.cpp ``isEidAlAdha`` table.
_EID_AL_ADHA: Final[tuple[Date, ...]] = (
    Date.from_ymd(7, Month.April, 1998),
    Date.from_ymd(27, Month.March, 1999),
    Date.from_ymd(16, Month.March, 2000),
    Date.from_ymd(5, Month.March, 2001),
    Date.from_ymd(23, Month.February, 2002),
    Date.from_ymd(12, Month.February, 2003),
    Date.from_ymd(1, Month.February, 2004),
    Date.from_ymd(21, Month.January, 2005),
    Date.from_ymd(10, Month.January, 2006),
    Date.from_ymd(31, Month.December, 2006),
    Date.from_ymd(20, Month.December, 2007),
    Date.from_ymd(8, Month.December, 2008),
    Date.from_ymd(27, Month.November, 2009),
    Date.from_ymd(16, Month.November, 2010),
    Date.from_ymd(6, Month.November, 2011),
    Date.from_ymd(26, Month.October, 2012),
    Date.from_ymd(15, Month.October, 2013),
    Date.from_ymd(4, Month.October, 2014),
    Date.from_ymd(24, Month.September, 2015),
    Date.from_ymd(11, Month.September, 2016),
    Date.from_ymd(1, Month.September, 2017),
    Date.from_ymd(23, Month.August, 2018),
    Date.from_ymd(12, Month.August, 2019),
    Date.from_ymd(31, Month.July, 2020),
    Date.from_ymd(20, Month.July, 2021),
    Date.from_ymd(10, Month.July, 2022),
)


# C++ parity: ql/time/calendars/saudiarabia.cpp ``isEidAlFitr`` table.
_EID_AL_FITR: Final[tuple[Date, ...]] = (
    Date.from_ymd(16, Month.December, 2001),
    Date.from_ymd(5, Month.December, 2002),
    Date.from_ymd(25, Month.November, 2003),
    Date.from_ymd(13, Month.November, 2004),
    Date.from_ymd(3, Month.November, 2005),
    Date.from_ymd(23, Month.October, 2006),
    Date.from_ymd(12, Month.October, 2007),
    Date.from_ymd(30, Month.September, 2008),
    Date.from_ymd(20, Month.September, 2009),
    Date.from_ymd(10, Month.September, 2010),
    Date.from_ymd(30, Month.August, 2011),
    Date.from_ymd(19, Month.August, 2012),
    Date.from_ymd(8, Month.August, 2013),
    Date.from_ymd(28, Month.July, 2014),
    Date.from_ymd(17, Month.July, 2015),
    Date.from_ymd(6, Month.July, 2016),
    Date.from_ymd(25, Month.June, 2017),
    Date.from_ymd(15, Month.June, 2018),
    Date.from_ymd(4, Month.June, 2019),
    Date.from_ymd(24, Month.May, 2020),
    Date.from_ymd(13, Month.May, 2021),
    Date.from_ymd(2, Month.May, 2022),
    Date.from_ymd(21, Month.April, 2023),
    Date.from_ymd(10, Month.April, 2024),
    Date.from_ymd(30, Month.March, 2025),
    Date.from_ymd(20, Month.March, 2026),
    Date.from_ymd(9, Month.March, 2027),
    Date.from_ymd(26, Month.February, 2028),
    Date.from_ymd(14, Month.February, 2029),
)

# Saudi weekend changed from (Thursday, Friday) to (Friday, Saturday) on 29-June-2013.
_WEEKEND_CHANGE_DATE: Final[Date] = Date.from_ymd(29, Month.June, 2013)


def _is_true_weekend(d: Date) -> bool:
    w = d.weekday()
    if d < _WEEKEND_CHANGE_DATE:
        return w in (Weekday.Thursday, Weekday.Friday)
    return w in (Weekday.Friday, Weekday.Saturday)


def _is_eid_window(d: Date, anchors: tuple[Date, ...]) -> bool:
    # C++ parity: ``any_of(p) { d >= p - 1 && d <= p + 4 }``.
    return any(p - 1 <= d <= p + 4 for p in anchors)


class SaudiArabia(Calendar):
    """Saudi Arabian Tadawul calendar."""

    def __init__(self, market: SaudiArabiaMarket = SaudiArabiaMarket.Tadawul) -> None:
        super().__init__()
        # C++ ``QL_FAIL("unknown market")`` for any other value.
        qassert.require(market == SaudiArabiaMarket.Tadawul, "unknown market")
        self._market: SaudiArabiaMarket = market

    def name(self) -> str:
        return "Tadawul"

    def _is_weekend(self, w: Weekday) -> bool:
        # C++ ``TadawulImpl::isWeekend`` (static rule — Friday + Saturday).
        return w in (Weekday.Friday, Weekday.Saturday)

    def _is_business_day(self, d: Date) -> bool:
        day = d.day_of_month()
        m = d.month()
        y = d.year()
        return not (
            _is_true_weekend(d)
            or _is_eid_window(d, _EID_AL_ADHA)
            or _is_eid_window(d, _EID_AL_FITR)
            # National Day
            or (day == 23 and m == Month.September)
            # other one-shot holidays
            or (day == 26 and m == Month.February and y == 2011)
            or (day == 19 and m == Month.March and y == 2011)
        )
