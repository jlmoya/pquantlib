"""Time machinery: calendars, day-count conventions, schedule generation."""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConv
from pquantlib.daycounters.thirty_360 import Convention as T360Conv
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.calendars.united_states import UnitedStates
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

CALENDARS = {"TARGET (EUR)": TARGET, "United States": UnitedStates, "United Kingdom": UnitedKingdom}

_MONTHS = [
    Month.January,
    Month.February,
    Month.March,
    Month.April,
    Month.May,
    Month.June,
    Month.July,
    Month.August,
    Month.September,
    Month.October,
    Month.November,
    Month.December,
]


def _date(y: int, m: int, d: int) -> Date:
    return Date.from_ymd(d, _MONTHS[m - 1], y)


@dataclass(frozen=True, slots=True)
class CalendarFacts:
    iso: str
    weekday: str
    is_business_day: bool
    is_holiday: bool
    following: str
    modified_following: str
    preceding: str


def calendar_facts(y: int, m: int, d: int, calendar_name: str) -> CalendarFacts:
    cal = CALENDARS[calendar_name]()
    dt = _date(y, m, d)
    return CalendarFacts(
        iso=str(dt),
        weekday=dt.weekday().name if hasattr(dt.weekday(), "name") else str(dt.weekday()),
        is_business_day=cal.is_business_day(dt),
        is_holiday=cal.is_holiday(dt),
        following=str(cal.adjust(dt, BusinessDayConvention.Following)),
        modified_following=str(cal.adjust(dt, BusinessDayConvention.ModifiedFollowing)),
        preceding=str(cal.adjust(dt, BusinessDayConvention.Preceding)),
    )


def day_count_table(y: int, m: int, d: int, months_ahead: int) -> list[tuple[str, int, float]]:
    """Each convention's (day count, year fraction) over the same interval."""
    start = _date(y, m, d)
    end = start + Period(months_ahead, TimeUnit.Months)
    counters = [
        ("Actual/360", Actual360()),
        ("Actual/365 Fixed", Actual365Fixed()),
        ("Actual/Actual (ISDA)", ActualActual(AAConv.ISDA)),
        ("30/360 (Bond Basis)", Thirty360(T360Conv.BondBasis)),
        ("30E/360 (European)", Thirty360(T360Conv.European)),
    ]
    return [(name, dc.day_count(start, end), dc.year_fraction(start, end)) for name, dc in counters]


def schedule_table(
    y: int, m: int, d: int, tenor_months: int, total_years: int, calendar_name: str
) -> list[str]:
    """Generate a coupon schedule and return the adjusted payment dates."""
    cal = CALENDARS[calendar_name]()
    start = _date(y, m, d)
    end = start + Period(total_years, TimeUnit.Years)
    sched = Schedule.from_rule(
        start,
        end,
        Period(tenor_months, TimeUnit.Months),
        cal,
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    return [str(dt) for dt in sched]
