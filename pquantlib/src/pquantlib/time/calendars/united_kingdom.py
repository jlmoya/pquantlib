"""United Kingdom calendars — Settlement / Exchange / Metals markets.

# C++ parity: ql/time/calendars/unitedkingdom.hpp + unitedkingdom.cpp (v1.42.1).

The three C++ Impl classes (``SettlementImpl``, ``ExchangeImpl``,
``MetalsImpl``) all share identical rule logic — the Python port collapses
them into a single ``_is_business_day`` driven by ``self._market`` (only
``name()`` differs between markets).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class UnitedKingdom(WesternCalendar):
    """UK calendars (Settlement, London Stock Exchange, London Metals)."""

    class Market(IntEnum):
        Settlement = 0  # generic settlement calendar
        Exchange = 1  # London stock-exchange calendar
        Metals = 2  # London metals-exchange calendar

    def __init__(self, market: Market = Market.Settlement) -> None:
        super().__init__()
        qassert.require(
            market
            in (
                UnitedKingdom.Market.Settlement,
                UnitedKingdom.Market.Exchange,
                UnitedKingdom.Market.Metals,
            ),
            "unknown market",
        )
        self._market = market

    def name(self) -> str:
        if self._market == UnitedKingdom.Market.Exchange:
            return "London stock exchange"
        if self._market == UnitedKingdom.Market.Metals:
            return "London metals exchange"
        return "UK settlement"

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        if self._is_weekend(w):
            return False
        day = d.day_of_month()
        dd = d.day_of_year()
        m = d.month()
        y = d.year()
        em = WesternCalendar.easter_monday(y)

        return not (
            # New Year's Day (possibly moved to Monday)
            ((day == 1 or ((day in (2, 3)) and w == Weekday.Monday)) and m == Month.January)
            # Good Friday
            or (dd == em - 3)
            # Easter Monday
            or (dd == em)
            or _is_bank_holiday(day, w, m, y)
            # Christmas (possibly moved to Monday or Tuesday)
            or ((day == 25 or (day == 27 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # Boxing Day (possibly moved to Monday or Tuesday)
            or ((day == 26 or (day == 28 and w in (Weekday.Monday, Weekday.Tuesday))) and m == Month.December)
            # December 31st, 1999 only
            or (day == 31 and m == Month.December and y == 1999)
        )


def _is_bank_holiday(d: int, w: Weekday, m: Month, y: int) -> bool:
    """Common bank-holiday rules used by all three UK markets.

    Mirrors C++ ``unitedkingdom.cpp`` anonymous-namespace ``isBankHoliday``.
    """
    return (
        # First Monday of May (Early May Bank Holiday); moved to May 8th
        # in 1995 and 2020 for V.E. day.
        (d <= 7 and w == Weekday.Monday and m == Month.May and y not in {1995, 2020})
        or (d == 8 and m == Month.May and (y in {1995, 2020}))
        # Last Monday of May (Spring Bank Holiday); moved in 2002, 2012, 2022
        # for the Golden, Diamond, and Platinum Jubilee with an additional holiday.
        or (d >= 25 and w == Weekday.Monday and m == Month.May and y not in {2002, 2012, 2022})
        or ((d in {3, 4}) and m == Month.June and y == 2002)
        or ((d in {4, 5}) and m == Month.June and y == 2012)
        or ((d in {2, 3}) and m == Month.June and y == 2022)
        # Last Monday of August (Summer Bank Holiday)
        or (d >= 25 and w == Weekday.Monday and m == Month.August)
        # April 29th, 2011 only (Royal Wedding)
        or (d == 29 and m == Month.April and y == 2011)
        # September 19th, 2022 only (Queen's Funeral)
        or (d == 19 and m == Month.September and y == 2022)
        # May 8th, 2023 only (King Charles III Coronation)
        or (d == 8 and m == Month.May and y == 2023)
    )
