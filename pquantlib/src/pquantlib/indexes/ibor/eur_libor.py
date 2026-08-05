"""EURLibor — EUR ICE LIBOR, fixed in London against a TARGET value date.

# C++ parity: ql/indexes/ibor/eurlibor.{hpp,cpp} (v1.43).

EUR LIBOR is the odd one out in the LIBOR family and does *not* derive from
``Libor``: it is fixed on the joint(UK exchange, TARGET) calendar, but every
date roll — fixing→value, value→fixing and value→maturity — runs on TARGET
alone. Getting that split wrong silently shifts accruals whenever London and
TARGET holidays differ.

The daily tenors go through ``DailyTenorEURLibor``, whose fixing calendar is
TARGET only (no London leg at all).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.joint_calendar import JointCalendar, JointCalendarRule
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _eurlibor_convention(p: Period) -> BusinessDayConvention:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return BusinessDayConvention.Following
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return BusinessDayConvention.ModifiedFollowing
    qassert.fail("invalid time units")


def _eurlibor_eom(p: Period) -> bool:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return False
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return True
    qassert.fail("invalid time units")


class EURLibor(IborIndex):
    """EUR ICE LIBOR — fixed on joint(UK exchange, TARGET), rolled on TARGET."""

    def __init__(self, tenor: Period, h: YieldTermStructureProtocol | None = None) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        super().__init__(
            "EURLibor",
            tenor,
            2,
            EURCurrency(),
            JointCalendar(
                [UnitedKingdom(UnitedKingdom.Market.Exchange), TARGET()],
                JointCalendarRule.JoinHolidays,
            ),
            _eurlibor_convention(tenor),
            _eurlibor_eom(tenor),
            Actual360(),
            h,
        )
        self._target: Calendar = TARGET()

    # --- date overrides --------------------------------------------------------

    def fixing_date(self, value_date: Date) -> Date:
        """Mirror C++ ``EURLibor::fixingDate`` — step back on TARGET, adjust on the joint."""
        return self._fixing_calendar.adjust(
            self._target.advance(value_date, -self._fixing_days, TimeUnit.Days),
            BusinessDayConvention.Preceding,
        )

    def value_date(self, fixing_date: Date) -> Date:
        """Mirror C++ ``EURLibor::valueDate`` — advance ``fixingDays`` on TARGET."""
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )
        return self._target.advance(fixing_date, self._fixing_days, TimeUnit.Days)

    def maturity_date(self, value_date: Date) -> Date:
        """Mirror C++ ``EURLibor::maturityDate`` — advance by tenor on TARGET."""
        return self._target.advance(
            value_date, self._tenor.length, self._tenor.units,
            self._convention, self._end_of_month,
        )

    def clone(self, forecast_term_structure: YieldTermStructureProtocol | None) -> EURLibor:
        """Mirror C++ ``EURLibor::clone`` — a fresh EURLibor on the same tenor."""
        return EURLibor(self._tenor, forecast_term_structure)


class DailyTenorEURLibor(IborIndex):
    """O/N + T/N + S/N EUR LIBOR — fixed and rolled on TARGET alone."""

    def __init__(
        self, settlement_days: int, h: YieldTermStructureProtocol | None = None,
    ) -> None:
        one_day = Period(1, TimeUnit.Days)
        super().__init__(
            "EURLibor",
            one_day,
            settlement_days,
            EURCurrency(),
            TARGET(),
            _eurlibor_convention(one_day),
            _eurlibor_eom(one_day),
            Actual360(),
            h,
        )


class EURLiborON(DailyTenorEURLibor):
    """Overnight EUR LIBOR (0 settlement days)."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(0, h)


class EURLibor1M(EURLibor):
    """1-month EUR LIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Months), h)


class EURLibor3M(EURLibor):
    """3-month EUR LIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(3, TimeUnit.Months), h)


class EURLibor6M(EURLibor):
    """6-month EUR LIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(6, TimeUnit.Months), h)


class EURLibor1Y(EURLibor):
    """1-year EUR LIBOR."""

    def __init__(self, h: YieldTermStructureProtocol | None = None) -> None:
        super().__init__(Period(1, TimeUnit.Years), h)
