"""South Korean calendars — Settlement and KRX (Korea exchange).

# C++ parity: ql/time/calendars/southkorea.hpp + southkorea.cpp (v1.42.1).

The C++ class has two markets:
- ``Settlement``: public holidays only.
- ``KRX`` (default): public holidays + year-end closing + occasional KRX-only days.

The KRX impl inherits from Settlement impl in C++; the Python port dispatches
on ``self._market`` inside ``_is_business_day``. The shared logic lives in
``_settlement_is_business_day``. Because pyright reports the original 60-line
``or``-chain as too complex to analyse, the rule list is split into a sequence
of narrowly-scoped predicate helpers.

Note: although weekends are Sat+Sun and there is no Easter rule in C++, we
inherit from ``WesternCalendar`` for the weekend definition and ``_is_weekend``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class SouthKorea(WesternCalendar):
    """South Korean calendar (Settlement / KRX markets)."""

    class Market(IntEnum):
        Settlement = 0  # Public holidays
        KRX = 1  # Korea exchange

    def __init__(self, market: Market = Market.KRX) -> None:
        super().__init__()
        qassert.require(
            market in (SouthKorea.Market.Settlement, SouthKorea.Market.KRX),
            "unknown market",
        )
        self._market = market

    def name(self) -> str:
        if self._market == SouthKorea.Market.Settlement:
            return "South-Korean settlement"
        return "South-Korea exchange"

    def _is_business_day(self, d: Date) -> bool:
        # Public holidays first.
        if not _settlement_is_business_day(d):
            return False
        if self._market == SouthKorea.Market.Settlement:
            return True
        # KRX-only closures.
        return _krx_extra_business_day(d)


# --- shared rule helpers -------------------------------------------------


def _is_fixed_holiday(d: int, m: Month, y: int, w: Weekday) -> bool:
    """Fixed-date Korean public holidays (with substitution-Monday rules)."""
    return (
        # New Year's Day
        (d == 1 and m == Month.January)
        # Independence Day
        or (d == 1 and m == Month.March)
        or (w == Weekday.Monday and (d in {2, 3}) and m == Month.March and y > 2021)
        # Arbour Day (until 2005)
        or (d == 5 and m == Month.April and y <= 2005)
        # Labour Day
        or (d == 1 and m == Month.May)
        # Children's Day
        or (d == 5 and m == Month.May)
        or (w == Weekday.Monday and (d in {6, 7}) and m == Month.May and y > 2013)
        # Memorial Day
        or (d == 6 and m == Month.June)
        # Constitution Day (until 2007)
        or (d == 17 and m == Month.July and y <= 2007)
        # Liberation Day
        or (d == 15 and m == Month.August)
        or (w == Weekday.Monday and (d in {16, 17}) and m == Month.August and y > 2020)
        # National Foundation Day
        or (d == 3 and m == Month.October)
        or (w == Weekday.Monday and (d in {4, 5}) and m == Month.October and y > 2020)
        # Christmas Day
        or (d == 25 and m == Month.December)
        or (w == Weekday.Monday and (d in {26, 27}) and m == Month.December and y > 2022)
    )


def _is_lunar_new_year(d: int, m: Month, y: int) -> bool:
    """Lunar New Year, 2004..2050 (per C++ ``southkorea.cpp``)."""
    if y == 2004:
        return d in (21, 22, 23) and m == Month.January
    if y == 2005:
        return d in (8, 9, 10) and m == Month.February
    if y == 2006:
        return d in (28, 29, 30) and m == Month.January
    if y == 2007:
        return d == 19 and m == Month.February
    if y == 2008:
        return d in (6, 7, 8) and m == Month.February
    if y == 2009:
        return d in (25, 26, 27) and m == Month.January
    if y == 2010:
        return d in (13, 14, 15) and m == Month.February
    if y == 2011:
        return d in (2, 3, 4) and m == Month.February
    if y == 2012:
        return d in (23, 24) and m == Month.January
    if y == 2013:
        return d == 11 and m == Month.February
    if y == 2014:
        return d in (30, 31) and m == Month.January
    if y == 2015:
        return d in (18, 19, 20) and m == Month.February
    if y == 2016:
        return 7 <= d <= 10 and m == Month.February
    if y == 2017:
        return 27 <= d <= 30 and m == Month.January
    if y == 2018:
        return d in (15, 16, 17) and m == Month.February
    if y == 2019:
        return d in (4, 5, 6) and m == Month.February
    if y == 2020:
        return 24 <= d <= 27 and m == Month.January
    if y == 2021:
        return d in (11, 12, 13) and m == Month.February
    if y == 2022:
        return (d == 31 and m == Month.January) or ((d in {1, 2}) and m == Month.February)
    if y == 2023:
        return d in (23, 24) and m == Month.January
    if y == 2024:
        return 9 <= d <= 12 and m == Month.February
    if y == 2025:
        return d in (28, 29, 30) and m == Month.January
    if y == 2026:
        return d in (16, 17, 18) and m == Month.February
    if y == 2027:
        return d in (8, 9) and m == Month.February
    if y == 2028:
        return d in (26, 27, 28) and m == Month.January
    if y == 2029:
        return d in (12, 13, 14) and m == Month.February
    if y == 2030:
        return d in (4, 5) and m == Month.February
    if y == 2031:
        return d in (22, 23, 24) and m == Month.January
    if y == 2032:
        return d in (10, 11, 12) and m == Month.February
    if y == 2033:
        return (d == 31 and m == Month.January) or ((d in {1, 2}) and m == Month.February)
    if y == 2034:
        return d in (20, 21) and m == Month.February
    if y == 2035:
        return d in (7, 8, 9) and m == Month.February
    if y == 2036:
        return d in (28, 29, 30) and m == Month.January
    if y == 2037:
        return d in (16, 17) and m == Month.February
    if y == 2038:
        return d in (3, 4, 5) and m == Month.February
    if y == 2039:
        return d in (24, 25, 26) and m == Month.January
    if y == 2040:
        return d in (13, 14) and m == Month.February
    if y == 2041:
        return (d == 31 and m == Month.January) or ((d in {1, 2}) and m == Month.February)
    if y == 2042:
        return d in (21, 22, 23) and m == Month.January
    if y == 2043:
        return d in (9, 10, 11) and m == Month.February
    if y == 2044:
        return (d in (29, 30, 31) and m == Month.January) or (d == 1 and m == Month.February)
    if y == 2045:
        return d in (16, 17, 18) and m == Month.February
    if y == 2046:
        return d in (5, 6, 7) and m == Month.February
    if y == 2047:
        return 25 <= d <= 28 and m == Month.January
    if y == 2048:
        return d in (13, 14, 15) and m == Month.February
    if y == 2049:
        return d in (1, 2, 3) and m == Month.February
    if y == 2050:
        return d in (24, 25) and m == Month.January
    return False


def _is_election_day(d: int, m: Month, y: int) -> bool:
    """Election days, 2004..2024."""
    if y == 2004:
        return d == 15 and m == Month.April
    if y == 2006:
        return d == 31 and m == Month.May
    if y == 2007:
        return d == 19 and m == Month.December
    if y == 2008:
        return d == 9 and m == Month.April
    if y == 2010:
        return d == 2 and m == Month.June
    if y == 2012:
        return (d == 11 and m == Month.April) or (d == 19 and m == Month.December)
    if y == 2014:
        return d == 4 and m == Month.June
    if y == 2016:
        return d == 13 and m == Month.April
    if y == 2017:
        return d == 9 and m == Month.May
    if y == 2018:
        return d == 13 and m == Month.June
    if y == 2020:
        return d == 15 and m == Month.April
    if y == 2022:
        return (d == 9 and m == Month.March) or (d == 1 and m == Month.June)
    if y == 2024:
        return d == 10 and m == Month.April
    return False


_BUDDHA_BIRTHDAYS: dict[int, tuple[int, Month]] = {
    2004: (26, Month.May),
    2005: (15, Month.May),
    2006: (5, Month.May),
    2007: (24, Month.May),
    2008: (12, Month.May),
    2009: (2, Month.May),
    2010: (21, Month.May),
    2011: (10, Month.May),
    2012: (28, Month.May),
    2013: (17, Month.May),
    2014: (6, Month.May),
    2015: (25, Month.May),
    2016: (14, Month.May),
    2017: (3, Month.May),
    2018: (22, Month.May),
    2019: (12, Month.May),
    2020: (30, Month.April),
    2021: (19, Month.May),
    2022: (8, Month.May),
    2023: (29, Month.May),
    2024: (15, Month.May),
    2025: (6, Month.May),
    2026: (25, Month.May),
    2027: (13, Month.May),
    2028: (2, Month.May),
    2029: (21, Month.May),
    2030: (9, Month.May),
    2031: (28, Month.May),
    2032: (17, Month.May),
    2033: (6, Month.May),
    2034: (25, Month.May),
    2035: (15, Month.May),
    2036: (6, Month.May),
    2037: (22, Month.May),
    2038: (11, Month.May),
    2039: (2, Month.May),
    2040: (18, Month.May),
    2041: (7, Month.May),
    2042: (26, Month.May),
    2043: (18, Month.May),
    2044: (6, Month.May),
    2045: (24, Month.May),
    2046: (14, Month.May),
    2047: (2, Month.May),
    2048: (20, Month.May),
    2049: (10, Month.May),
    2050: (30, Month.May),
}


def _is_buddhas_birthday(d: int, m: Month, y: int) -> bool:
    entry = _BUDDHA_BIRTHDAYS.get(y)
    if entry is None:
        return False
    return d == entry[0] and m == entry[1]


def _is_special_holiday(d: int, m: Month, y: int) -> bool:
    """Special / temporary holidays for specific years."""
    return (
        # 70 years from Independence Day
        (d == 14 and m == Month.August and y == 2015)
        # Special temporary holidays
        or (d == 17 and m == Month.August and y == 2020)
        or (d == 2 and m == Month.October and y == 2023)
        or (d == 1 and m == Month.October and y == 2024)
    )


def _is_harvest_moon_day(d: int, m: Month, y: int) -> bool:
    """Harvest Moon Day (Chuseok), 2004..2050."""
    if y == 2004:
        return d in (27, 28, 29) and m == Month.September
    if y == 2005:
        return d in (17, 18, 19) and m == Month.September
    if y == 2006:
        return d in (5, 6, 7) and m == Month.October
    if y == 2007:
        return d in (24, 25, 26) and m == Month.September
    if y == 2008:
        return d in (13, 14, 15) and m == Month.September
    if y == 2009:
        return d in (2, 3, 4) and m == Month.October
    if y == 2010:
        return d in (21, 22, 23) and m == Month.September
    if y == 2011:
        return d in (12, 13) and m == Month.September
    if y == 2012:
        return d == 1 and m == Month.October
    if y == 2013:
        return d in (18, 19, 20) and m == Month.September
    if y == 2014:
        return d in (8, 9, 10) and m == Month.September
    if y == 2015:
        return d in (28, 29) and m == Month.September
    if y == 2016:
        return d in (14, 15, 16) and m == Month.September
    if y == 2017:
        return 3 <= d <= 6 and m == Month.October
    if y == 2018:
        return 23 <= d <= 26 and m == Month.September
    if y == 2019:
        return d in (12, 13, 14) and m == Month.September
    if y == 2020:
        return (d == 30 and m == Month.September) or ((d in {1, 2}) and m == Month.October)
    if y == 2021:
        return d in (20, 21, 22) and m == Month.September
    if y == 2022:
        # C++ has both d in (9,10,11) and 9<=d<=12 in September 2022 — union = 9..12.
        return 9 <= d <= 12 and m == Month.September
    if y == 2023:
        return d in (28, 29, 30) and m == Month.September
    if y == 2024:
        return d in (16, 17, 18) and m == Month.September
    if y == 2025:
        return d in (6, 7, 8) and m == Month.October
    if y == 2026:
        return d in (24, 25, 26) and m == Month.September
    if y == 2027:
        return d in (14, 15, 16) and m == Month.September
    if y == 2028:
        return 2 <= d <= 5 and m == Month.October
    if y == 2029:
        return 21 <= d <= 24 and m == Month.September
    if y == 2030:
        return d in (11, 12, 13) and m == Month.September
    if y == 2031:
        return (d == 30 and m == Month.September) or ((d in {1, 2}) and m == Month.October)
    if y == 2032:
        return d in (20, 21) and m == Month.September
    if y == 2033:
        return d in (7, 8, 9) and m == Month.September
    if y == 2034:
        return d in (26, 27, 28) and m == Month.September
    if y == 2035:
        return d in (17, 18) and m == Month.September
    if y == 2036:
        return 3 <= d <= 7 and m == Month.October
    if y == 2037:
        return d in (23, 24, 25) and m == Month.September
    if y == 2038:
        return d in (13, 14, 15) and m == Month.September
    if y == 2039:
        return d in (3, 4, 5) and m == Month.October
    if y == 2040:
        return d in (20, 21, 22) and m == Month.September
    if y == 2041:
        return d in (9, 10, 11) and m == Month.September
    if y == 2042:
        return d in (29, 30) and m == Month.September
    if y == 2043:
        return d in (16, 17, 18) and m == Month.September
    if y == 2044:
        return d in (4, 5, 6) and m == Month.October
    if y == 2045:
        return d in (25, 26, 27) and m == Month.September
    if y == 2046:
        return 14 <= d <= 17 and m == Month.September
    if y == 2047:
        return d in (4, 5, 7) and m == Month.October
    if y == 2048:
        return d in (21, 22, 23) and m == Month.September
    if y == 2049:
        return 10 <= d <= 13 and m == Month.September
    if y == 2050:
        return ((d in {29, 30}) and m == Month.September) or (d == 1 and m == Month.October)
    return False


def _is_hangul_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    """Hangul Proclamation Day (October 9), since 2013, with substitution since 2020."""
    return (d == 9 and m == Month.October and y >= 2013) or (
        w == Weekday.Monday and (d in {10, 11}) and m == Month.October and y > 2020
    )


def _settlement_is_business_day(date: Date) -> bool:
    """Mirrors C++ ``SouthKorea::SettlementImpl::isBusinessDay``."""
    w = date.weekday()
    d = date.day_of_month()
    m = date.month()
    y = date.year()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    return not (
        _is_fixed_holiday(d, m, y, w)
        or _is_lunar_new_year(d, m, y)
        or _is_election_day(d, m, y)
        or _is_buddhas_birthday(d, m, y)
        or _is_special_holiday(d, m, y)
        or _is_harvest_moon_day(d, m, y)
        or _is_hangul_day(d, m, y, w)
    )


def _krx_extra_business_day(date: Date) -> bool:
    """Return True iff this KRX-additional rule does not declare a holiday.

    Mirrors the post-Settlement checks in ``SouthKorea::KrxImpl::isBusinessDay``.
    """
    d = date.day_of_month()
    w = date.weekday()
    m = date.month()
    y = date.year()

    # Year-end closing
    if (((d in {29, 30}) and w == Weekday.Friday) or d == 31) and m == Month.December:
        return False
    # Occasional closing days (KRX day)
    return not ((d == 6 and m == Month.May and y == 2016) or (d == 2 and m == Month.October and y == 2017))
