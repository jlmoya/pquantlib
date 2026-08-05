"""CustomIborIndex — IborIndex with separate fixing / value / maturity calendars.

# C++ parity: ql/indexes/ibor/custom.{hpp,cpp} (v1.43).

The plain ``IborIndex`` runs every date roll on the single fixing calendar.
Some markets (notably cross-currency conventions) publish a fixing on one
centre while settling and accruing on others, so this variant threads three
calendars: fixing dates are *adjusted* on the fixing calendar, value dates
*advance* on the value calendar and are then adjusted on the maturity
calendar, and maturities advance on the maturity calendar.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class CustomIborIndex(IborIndex):
    """IborIndex whose value and maturity rolls use their own calendars."""

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        settlement_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        value_calendar: Calendar,
        maturity_calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        day_counter: DayCounter,
        forecast_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            family_name,
            tenor,
            settlement_days,
            currency,
            fixing_calendar,
            convention,
            end_of_month,
            day_counter,
            forecast_term_structure,
        )
        self._value_calendar: Calendar = value_calendar
        self._maturity_calendar: Calendar = maturity_calendar

    # --- date overrides --------------------------------------------------------

    def fixing_date(self, value_date: Date) -> Date:
        """Mirror C++ ``CustomIborIndex::fixingDate``."""
        fixing_date = self._value_calendar.advance(
            value_date, -self._fixing_days, TimeUnit.Days,
        )
        return self._fixing_calendar.adjust(fixing_date, BusinessDayConvention.Preceding)

    def value_date(self, fixing_date: Date) -> Date:
        """Mirror C++ ``CustomIborIndex::valueDate``."""
        qassert.require(
            self.is_valid_fixing_date(fixing_date),
            f"Fixing date {fixing_date} is not valid",
        )
        d = self._value_calendar.advance(fixing_date, self._fixing_days, TimeUnit.Days)
        return self._maturity_calendar.adjust(d)

    def maturity_date(self, value_date: Date) -> Date:
        """Mirror C++ ``CustomIborIndex::maturityDate``."""
        return self._maturity_calendar.advance(
            value_date, self._tenor.length, self._tenor.units,
            self._convention, self._end_of_month,
        )

    def clone(
        self, forecast_term_structure: YieldTermStructureProtocol | None,
    ) -> CustomIborIndex:
        """Mirror C++ ``CustomIborIndex::clone`` — keeps all three calendars."""
        return CustomIborIndex(
            self._family_name,
            self._tenor,
            self._fixing_days,
            self._currency,
            self._fixing_calendar,
            self._value_calendar,
            self._maturity_calendar,
            self._convention,
            self._end_of_month,
            self._day_counter,
            forecast_term_structure,
        )

    # --- inspectors ------------------------------------------------------------

    def value_calendar(self) -> Calendar:
        return self._value_calendar

    def maturity_calendar(self) -> Calendar:
        return self._maturity_calendar
