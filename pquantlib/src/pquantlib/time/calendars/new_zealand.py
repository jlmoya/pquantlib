"""New Zealand calendar — Wellington / Auckland markets.

# C++ parity: ql/time/calendars/newzealand.hpp + newzealand.cpp (v1.42.1).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


def _is_common_business_day(d: Date) -> bool:
    """C++ parity: ql/time/calendars/newzealand.cpp ``CommonImpl::isBusinessDay``.

    Returns ``True`` if ``d`` is a business day under the common (national) rules.
    The market-specific subclasses (Wellington / Auckland) layer additional
    Anniversary Day rules on top.
    """
    w = d.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    day = d.day_of_month()
    dd = d.day_of_year()
    m = d.month()
    y = d.year()
    em = WesternCalendar.easter_monday(y)
    return not (
        # New Year's Day (possibly moved to Monday or Tuesday)
        ((day == 1 or (day == 3 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.January)
        # Day after New Year's Day (possibly moved to Monday or Tuesday)
        or ((day == 2 or (day == 4 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.January)
        # Waitangi Day, February 6th (possibly moved to Monday since 2013)
        or (day == 6 and m == Month.February)
        or (day in (7, 8) and w == Weekday.Monday and m == Month.February and y > 2013)
        # Good Friday
        or (dd == em - 3)
        # Easter Monday
        or (dd == em)
        # ANZAC Day, April 25th (possibly moved to Monday since 2013)
        or (day == 25 and m == Month.April)
        or (day in (26, 27) and w == Weekday.Monday and m == Month.April and y > 2013)
        # Queen's Birthday, first Monday in June
        or (day <= 7 and w == Weekday.Monday and m == Month.June)
        # Labour Day, fourth Monday in October
        or (22 <= day <= 28 and w == Weekday.Monday and m == Month.October)
        # Christmas, December 25th (possibly Monday or Tuesday)
        or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
        # Boxing Day, December 26th (possibly Monday or Tuesday)
        or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
        # Matariki holidays (NZ government release, 2022-2052)
        or (day == 20 and m == Month.June and y == 2025)
        or (day == 21 and m == Month.June and y in (2030, 2052))
        or (day == 24 and m == Month.June and y in (2022, 2033, 2044))
        or (day == 25 and m == Month.June and y in (2027, 2038, 2049))
        or (day == 28 and m == Month.June and y == 2024)
        or (day == 29 and m == Month.June and y in (2035, 2046))
        or (day == 30 and m == Month.June and y == 2051)
        or (day == 2 and m == Month.July and y == 2032)
        or (day == 3 and m == Month.July and y in (2043, 2048))
        or (day == 6 and m == Month.July and y in (2029, 2040))
        or (day == 7 and m == Month.July and y in (2034, 2045))
        or (day == 10 and m == Month.July and y in (2026, 2037))
        or (day == 11 and m == Month.July and y in (2031, 2042))
        or (day == 14 and m == Month.July and y in (2023, 2028))
        or (day == 15 and m == Month.July and y in (2039, 2050))
        or (day == 18 and m == Month.July and y == 2036)
        or (day == 19 and m == Month.July and y in (2041, 2047))
        # Queen Elizabeth's funeral, September 26 2022
        or (day == 26 and m == Month.September and y == 2022)
    )


class NewZealand(WesternCalendar):
    """New Zealand calendar (Wellington default, or Auckland)."""

    class Market(IntEnum):
        Wellington = 0
        Auckland = 1

    def __init__(self, market: Market = Market.Wellington) -> None:
        super().__init__()
        self._market = market

    def name(self) -> str:
        if self._market == NewZealand.Market.Wellington:
            return "New Zealand (Wellington)"
        if self._market == NewZealand.Market.Auckland:
            return "New Zealand (Auckland)"
        qassert.fail("unknown market")

    def _is_business_day(self, d: Date) -> bool:
        if not _is_common_business_day(d):
            return False
        w = d.weekday()
        day = d.day_of_month()
        m = d.month()
        if self._market == NewZealand.Market.Wellington:
            # Anniversary Day, Monday nearest January 22nd
            return not (19 <= day <= 25 and w == Weekday.Monday and m == Month.January)
        if self._market == NewZealand.Market.Auckland:
            # Anniversary Day, Monday nearest January 29th
            return not (
                (day >= 26 and w == Weekday.Monday and m == Month.January)
                or (day == 1 and w == Weekday.Monday and m == Month.February)
            )
        qassert.fail("unknown market")
