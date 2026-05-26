"""United States calendars — seven distinct markets.

# C++ parity: ql/time/calendars/unitedstates.hpp + unitedstates.cpp (v1.42.1).

Markets supported (matching C++ ``UnitedStates::Market``):
- ``Settlement`` — generic settlement calendar.
- ``NYSE`` — New York stock exchange.
- ``GovernmentBond`` — US government bond market.
- ``NERC`` — North American Energy Reliability Council off-peak days.
- ``LiborImpact`` — Libor impact calendar (since 2015).
- ``FederalReserve`` — Federal Reserve Bankwire System.
- ``SOFR`` — SOFR fixing calendar.

The C++ class has no default constructor (``explicit UnitedStates(Market)``).
The Python port defaults to ``Market.Settlement`` so that ``UnitedStates()``
matches the probe's "us settlement" reference.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.time.calendar import WesternCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.weekday import Weekday


class UnitedStates(WesternCalendar):
    """US calendars — multiple market variants."""

    class Market(IntEnum):
        Settlement = 0  # generic settlement calendar
        NYSE = 1  # New York stock exchange
        GovernmentBond = 2  # government bond market
        NERC = 3  # NERC off-peak days
        LiborImpact = 4  # Libor impact calendar
        FederalReserve = 5  # Federal Reserve Bankwire System
        SOFR = 6  # SOFR fixing calendar

    _VALID_MARKETS = (
        Market.Settlement,
        Market.NYSE,
        Market.GovernmentBond,
        Market.NERC,
        Market.LiborImpact,
        Market.FederalReserve,
        Market.SOFR,
    )

    def __init__(self, market: Market = Market.Settlement) -> None:
        super().__init__()
        qassert.require(market in UnitedStates._VALID_MARKETS, "unknown market")
        self._market = market

    def name(self) -> str:
        m = self._market
        if m == UnitedStates.Market.NYSE:
            return "New York stock exchange"
        if m == UnitedStates.Market.GovernmentBond:
            return "US government bond market"
        if m == UnitedStates.Market.NERC:
            return "North American Energy Reliability Council"
        if m == UnitedStates.Market.LiborImpact:
            return "US with Libor impact"
        if m == UnitedStates.Market.FederalReserve:
            return "Federal Reserve Bankwire System"
        if m == UnitedStates.Market.SOFR:
            return "SOFR fixing calendar"
        return "US settlement"

    def _is_business_day(self, d: Date) -> bool:
        m = self._market
        if m == UnitedStates.Market.NYSE:
            return _nyse_is_business_day(d)
        if m == UnitedStates.Market.GovernmentBond:
            return _government_bond_is_business_day(d)
        if m == UnitedStates.Market.NERC:
            return _nerc_is_business_day(d)
        if m == UnitedStates.Market.LiborImpact:
            return _libor_impact_is_business_day(d)
        if m == UnitedStates.Market.FederalReserve:
            return _federal_reserve_is_business_day(d)
        if m == UnitedStates.Market.SOFR:
            return _sofr_is_business_day(d)
        return _settlement_is_business_day(d)


# --- shared rule helpers (mirror anonymous-namespace functions in C++) ----


def _is_washington_birthday(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y >= 1971:
        # Third Monday in February
        return 15 <= d <= 21 and w == Weekday.Monday and m == Month.February
    # February 22nd, possibly adjusted.
    return (
        d == 22 or (d == 23 and w == Weekday.Monday) or (d == 21 and w == Weekday.Friday)
    ) and m == Month.February


def _is_memorial_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y >= 1971:
        return d >= 25 and w == Weekday.Monday and m == Month.May
    return (
        d == 30 or (d == 31 and w == Weekday.Monday) or (d == 29 and w == Weekday.Friday)
    ) and m == Month.May


def _is_labor_day(d: int, m: Month, _y: int, w: Weekday) -> bool:
    return d <= 7 and w == Weekday.Monday and m == Month.September


def _is_columbus_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    return 8 <= d <= 14 and w == Weekday.Monday and m == Month.October and y >= 1971


def _is_veterans_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y <= 1970 or y >= 1978:
        return (
            d == 11 or (d == 12 and w == Weekday.Monday) or (d == 10 and w == Weekday.Friday)
        ) and m == Month.November
    return 22 <= d <= 28 and w == Weekday.Monday and m == Month.October


def _is_veterans_day_no_saturday(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y <= 1970 or y >= 1978:
        return (d == 11 or (d == 12 and w == Weekday.Monday)) and m == Month.November
    return 22 <= d <= 28 and w == Weekday.Monday and m == Month.October


def _is_juneteenth(d: int, m: Month, y: int, w: Weekday, move_to_friday: bool = True) -> bool:
    # Declared 2021, observed by exchanges since 2022.
    return (
        (d == 19 or (d == 20 and w == Weekday.Monday) or (d == 18 and w == Weekday.Friday and move_to_friday))
        and m == Month.June
        and y >= 2022
    )


# --- per-market business-day predicates -----------------------------------


def _settlement_is_business_day(date: Date) -> bool:
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    m = date.month()
    y = date.year()
    return not (
        # New Year's Day (Monday if Sunday)
        ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
        # (or to Friday if on Saturday)
        or (d == 31 and w == Weekday.Friday and m == Month.December)
        # Martin Luther King's birthday (third Monday in January, since 1983)
        or (15 <= d <= 21 and w == Weekday.Monday and m == Month.January and y >= 1983)
        or _is_washington_birthday(d, m, y, w)
        or _is_memorial_day(d, m, y, w)
        or _is_juneteenth(d, m, y, w)
        # Independence Day (Monday if Sunday or Friday if Saturday)
        or (
            (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
            and m == Month.July
        )
        or _is_labor_day(d, m, y, w)
        or _is_columbus_day(d, m, y, w)
        or _is_veterans_day(d, m, y, w)
        # Thanksgiving Day (fourth Thursday in November)
        or (22 <= d <= 28 and w == Weekday.Thursday and m == Month.November)
        # Christmas (Monday if Sunday or Friday if Saturday)
        or (
            (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
            and m == Month.December
        )
    )


def _libor_impact_is_business_day(date: Date) -> bool:
    # Since 2015 Independence Day only impacts Libor if it falls on a weekday.
    w = date.weekday()
    d = date.day_of_month()
    m = date.month()
    y = date.year()
    if (
        ((d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
        and m == Month.July
        and y >= 2015
    ):
        return True
    return _settlement_is_business_day(date)


def _nyse_is_business_day(date: Date) -> bool:
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    dd = date.day_of_year()
    m = date.month()
    y = date.year()
    em = WesternCalendar.easter_monday(y)

    if (
        # New Year's Day (Monday if Sunday)
        ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
        or _is_washington_birthday(d, m, y, w)
        # Good Friday
        or (dd == em - 3)
        or _is_memorial_day(d, m, y, w)
        or _is_juneteenth(d, m, y, w)
        # Independence Day (Monday if Sunday or Friday if Saturday)
        or (
            (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
            and m == Month.July
        )
        or _is_labor_day(d, m, y, w)
        # Thanksgiving Day (fourth Thursday in November)
        or (22 <= d <= 28 and w == Weekday.Thursday and m == Month.November)
        # Christmas (Monday if Sunday or Friday if Saturday)
        or (
            (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
            and m == Month.December
        )
    ):
        return False

    # Martin Luther King's birthday (NYSE since 1998)
    if y >= 1998 and (15 <= d <= 21) and w == Weekday.Monday and m == Month.January:
        return False

    # Presidential election days (until 1980)
    if (y <= 1968 or (y <= 1980 and y % 4 == 0)) and m == Month.November and d <= 7 and w == Weekday.Tuesday:
        return False

    return not _is_nyse_special_closing(d, dd, w, m, y)


def _is_nyse_special_closing(d: int, dd: int, w: Weekday, m: Month, y: int) -> bool:
    """Mirrors the special-closings block inside ``NyseImpl::isBusinessDay``."""
    return (
        # President Carter's Funeral
        (y == 2025 and m == Month.January and d == 9)
        # President Bush's Funeral
        or (y == 2018 and m == Month.December and d == 5)
        # Hurricane Sandy
        or (y == 2012 and m == Month.October and (d in {29, 30}))
        # President Ford's funeral
        or (y == 2007 and m == Month.January and d == 2)
        # President Reagan's funeral
        or (y == 2004 and m == Month.June and d == 11)
        # September 11-14, 2001
        or (y == 2001 and m == Month.September and 11 <= d <= 14)
        # President Nixon's funeral
        or (y == 1994 and m == Month.April and d == 27)
        # Hurricane Gloria
        or (y == 1985 and m == Month.September and d == 27)
        # 1977 Blackout
        or (y == 1977 and m == Month.July and d == 14)
        # Funeral of former President Lyndon B. Johnson
        or (y == 1973 and m == Month.January and d == 25)
        # Funeral of former President Harry S. Truman
        or (y == 1972 and m == Month.December and d == 28)
        # National Day of Participation for the lunar exploration
        or (y == 1969 and m == Month.July and d == 21)
        # Funeral of former President Eisenhower
        or (y == 1969 and m == Month.March and d == 31)
        # Closed all day - heavy snow
        or (y == 1969 and m == Month.February and d == 10)
        # Day after Independence Day
        or (y == 1968 and m == Month.July and d == 5)
        # June 12 - Dec 31, 1968: four-day weeks (Paperwork Crisis)
        or (y == 1968 and dd >= 163 and w == Weekday.Wednesday)
        # Day of mourning for Martin Luther King Jr.
        or (y == 1968 and m == Month.April and d == 9)
        # Funeral of President Kennedy
        or (y == 1963 and m == Month.November and d == 25)
        # Day before Decoration Day
        or (y == 1961 and m == Month.May and d == 29)
        # Day after Christmas
        or (y == 1958 and m == Month.December and d == 26)
        # Christmas Eve (1954, 1956, 1965)
        or ((y in {1954, 1956, 1965}) and m == Month.December and d == 24)
    )


def _government_bond_is_business_day(date: Date) -> bool:
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    dd = date.day_of_year()
    m = date.month()
    y = date.year()
    em = WesternCalendar.easter_monday(y)

    return not (
        # New Year's Day (Monday if Sunday)
        ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
        # Martin Luther King's birthday (third Monday in January, since 1983)
        or (15 <= d <= 21 and w == Weekday.Monday and m == Month.January and y >= 1983)
        or _is_washington_birthday(d, m, y, w)
        # Good Friday. Since 1996 it's an early close (not a full market close)
        # when it coincides with the NFP release date — the first Friday of the
        # April month. So skip Good Friday if y >= 1996 and d <= 7.
        or (dd == em - 3 and (y < 1996 or d > 7))
        or _is_memorial_day(d, m, y, w)
        or _is_juneteenth(d, m, y, w)
        # Independence Day (Monday if Sunday or Friday if Saturday)
        or (
            (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
            and m == Month.July
        )
        or _is_labor_day(d, m, y, w)
        or _is_columbus_day(d, m, y, w)
        # Veterans' Day (no Saturday adjustment for bond market)
        or _is_veterans_day_no_saturday(d, m, y, w)
        # Thanksgiving Day (fourth Thursday in November)
        or (22 <= d <= 28 and w == Weekday.Thursday and m == Month.November)
        # Christmas (Monday if Sunday or Friday if Saturday)
        or (
            (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
            and m == Month.December
        )
        # Special closings (bond market subset)
        or (y == 2018 and m == Month.December and d == 5)
        or (y == 2012 and m == Month.October and d == 30)
        or (y == 2004 and m == Month.June and d == 11)
    )


def _sofr_is_business_day(date: Date) -> bool:
    # SOFR never fixes on Good Friday (so far through 2023; extrapolated).
    dy = date.day_of_year()
    y = date.year()
    if dy == (WesternCalendar.easter_monday(y) - 3):
        return False
    return _government_bond_is_business_day(date)


def _nerc_is_business_day(date: Date) -> bool:
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    m = date.month()
    y = date.year()
    return not (
        # New Year's Day (Monday if Sunday)
        ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
        or _is_memorial_day(d, m, y, w)
        # Independence Day (Monday if Sunday)
        or ((d == 4 or (d == 5 and w == Weekday.Monday)) and m == Month.July)
        or _is_labor_day(d, m, y, w)
        # Thanksgiving Day (fourth Thursday in November)
        or (22 <= d <= 28 and w == Weekday.Thursday and m == Month.November)
        # Christmas (Monday if Sunday)
        or ((d == 25 or (d == 26 and w == Weekday.Monday)) and m == Month.December)
    )


def _federal_reserve_is_business_day(date: Date) -> bool:
    w = date.weekday()
    if w in (Weekday.Saturday, Weekday.Sunday):
        return False
    d = date.day_of_month()
    m = date.month()
    y = date.year()
    return not (
        # New Year's Day (Monday if Sunday)
        ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
        # MLK Day (third Monday in January, since 1983)
        or (15 <= d <= 21 and w == Weekday.Monday and m == Month.January and y >= 1983)
        or _is_washington_birthday(d, m, y, w)
        or _is_memorial_day(d, m, y, w)
        # Juneteenth — Federal Reserve uses no-friday-move variant.
        or _is_juneteenth(d, m, y, w, False)
        # Independence Day (Monday if Sunday)
        or ((d == 4 or (d == 5 and w == Weekday.Monday)) and m == Month.July)
        or _is_labor_day(d, m, y, w)
        or _is_columbus_day(d, m, y, w)
        or _is_veterans_day_no_saturday(d, m, y, w)
        # Thanksgiving Day (fourth Thursday in November)
        or (22 <= d <= 28 and w == Weekday.Thursday and m == Month.November)
        # Christmas (Monday if Sunday)
        or ((d == 25 or (d == 26 and w == Weekday.Monday)) and m == Month.December)
    )
