"""Libor — base for ICE LIBOR indexes (all currencies but EUR).

# C++ parity: ql/indexes/ibor/libor.{hpp,cpp} (v1.42.1)

LIBOR conventions:
- Fixing calendar = UK Exchange (London).
- Value/maturity dates use a JointCalendar of (UK Exchange, financial-centre
  calendar) with ``JoinHolidays``.
- Days/Weeks tenors: ``Following``, EOM=False; Months/Years: ``ModifiedFollowing``,
  EOM=True (parity with C++ ``liborConvention``/``liborEOM`` anonymous helpers).
- ``DailyTenorLibor`` for O/N + S/N: the *fixing* calendar is the joint
  calendar, not just UK Exchange.

C++ explicitly forbids EUR currencies here (use ``EurLibor`` instead).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.joint_calendar import JointCalendar, JointCalendarRule
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _libor_convention(p: Period) -> BusinessDayConvention:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return BusinessDayConvention.Following
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return BusinessDayConvention.ModifiedFollowing
    qassert.fail("invalid time units")


def _libor_eom(p: Period) -> bool:
    if p.units in (TimeUnit.Days, TimeUnit.Weeks):
        return False
    if p.units in (TimeUnit.Months, TimeUnit.Years):
        return True
    qassert.fail("invalid time units")


class Libor(IborIndex):
    """ICE LIBOR base — all currencies except EUR; tenor != 1*Days."""

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        fixing_days: int,
        currency: Currency,
        financial_center_calendar: Calendar,
        day_counter: DayCounter,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        qassert.require(
            tenor.units != TimeUnit.Days,
            f"for daily tenors ({tenor}) dedicated DailyTenor constructor must be used",
        )
        qassert.require(
            currency != EURCurrency(),
            "for EUR Libor dedicated EurLibor constructor must be used",
        )
        super().__init__(
            family_name,
            tenor,
            fixing_days,
            currency,
            UnitedKingdom(UnitedKingdom.Market.Exchange),
            _libor_convention(tenor),
            _libor_eom(tenor),
            day_counter,
            forecast_term_structure,
        )
        self._financial_center_calendar = financial_center_calendar
        self._joint_calendar = JointCalendar(
            [UnitedKingdom(UnitedKingdom.Market.Exchange), financial_center_calendar],
            JointCalendarRule.JoinHolidays,
        )

    # --- date overrides --------------------------------------------------------

    def value_date(self, fixing_date: Date) -> Date:
        """Mirror C++ ``Libor::valueDate`` — fixingDays advance on UK + joint adjust.

        Special rule: advance ``fixingDays`` business days under the UK exchange
        calendar, then *adjust* under the joint calendar (UK + centre).
        """
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )
        d = self._fixing_calendar.advance(
            fixing_date, self._fixing_days, TimeUnit.Days,
        )
        return self._joint_calendar.adjust(d)

    def maturity_date(self, value_date: Date) -> Date:
        """Mirror C++ ``Libor::maturityDate`` — uses the joint calendar."""
        return self._joint_calendar.advance(
            value_date, self._tenor.length, self._tenor.units,
            self._convention, self._end_of_month,
        )

    # --- inspectors ------------------------------------------------------------

    def joint_calendar(self) -> Calendar:
        return self._joint_calendar


class DailyTenorLibor(IborIndex):
    """O/N + S/N LIBOR base — fixing calendar = joint(UK, centre)."""

    def __init__(
        self,
        family_name: str,
        fixing_days: int,
        currency: Currency,
        financial_center_calendar: Calendar,
        day_counter: DayCounter,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        qassert.require(
            currency != EURCurrency(),
            "for EUR Libor dedicated EurLibor constructor must be used",
        )
        super().__init__(
            family_name,
            Period(1, TimeUnit.Days),
            fixing_days,
            currency,
            JointCalendar(
                [UnitedKingdom(UnitedKingdom.Market.Exchange), financial_center_calendar],
                JointCalendarRule.JoinHolidays,
            ),
            BusinessDayConvention.Following,
            False,
            day_counter,
            forecast_term_structure,
        )
