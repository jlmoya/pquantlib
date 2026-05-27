"""FxSwapRateHelper — bootstrap from FX swap forward points.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class FxSwapRateHelper.

Implied quote (forward FX point) is calculated from two discount curves:

    if isFxBaseCurrencyCollateralCurrency:
        return (ratio / collRatio - 1) * spot
    else:
        return (collRatio / ratio - 1) * spot

where ``ratio = d_ts(earliest) / d_ts(latest)`` and
``collRatio = d_coll(earliest) / d_coll(latest)``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.joint_calendar import JointCalendar, JointCalendarRule
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class FxSwapRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """Bootstrap from FX swap forward-point quote."""

    def __init__(
        self,
        fwd_point: Quote | float,
        spot_fx: Quote,
        tenor: Period,
        fixing_days: int,
        calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        is_fx_base_currency_collateral_currency: bool,
        collateral_curve: YieldTermStructureProtocol,
        trading_calendar: Calendar | None = None,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(fwd_point)
        self._spot: Quote = spot_fx
        self._tenor: Period = tenor
        self._fixing_days: int = fixing_days
        self._calendar: Calendar = calendar
        self._convention: BusinessDayConvention = convention
        self._eom: bool = end_of_month
        self._is_base_collateral: bool = is_fx_base_currency_collateral_currency
        self._collateral: YieldTermStructureProtocol = collateral_curve
        self._trading_calendar: Calendar | None = trading_calendar
        if trading_calendar is not None:
            self._joint_calendar: Calendar = JointCalendar(
                [trading_calendar, calendar], JointCalendarRule.JoinHolidays,
            )
        else:
            self._joint_calendar = calendar
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    def implied_quote(self) -> float:
        qassert.require(self._term_structure is not None, "term structure not set")
        assert self._term_structure is not None
        qassert.require(self._earliest_date is not None, "dates not initialized")
        qassert.require(self._latest_date is not None, "dates not initialized")
        assert self._earliest_date is not None
        assert self._latest_date is not None
        d1c = self._collateral.discount(self._earliest_date)
        d2c = self._collateral.discount(self._latest_date)
        coll_ratio = d1c / d2c
        d1 = self._term_structure.discount(self._earliest_date)
        d2 = self._term_structure.discount(self._latest_date)
        ratio = d1 / d2
        spot = self._spot.value()
        if self._is_base_collateral:
            return (ratio / coll_ratio - 1) * spot
        return (coll_ratio / ratio - 1) * spot

    def initialize_dates(self, evaluation_date: Date) -> None:
        ref_date = self._calendar.adjust(
            evaluation_date, BusinessDayConvention.Following,
        )
        earliest = self._calendar.advance(
            ref_date, self._fixing_days, TimeUnit.Days,
        )
        if self._trading_calendar is not None:
            # joint-calendar adjustment for cross pairs without USD
            earliest = self._joint_calendar.adjust(
                earliest, BusinessDayConvention.Following,
            )
            latest = self._joint_calendar.advance(
                earliest, self._tenor.length, self._tenor.units,
                self._convention, self._eom,
            )
        else:
            latest = self._calendar.advance(
                earliest, self._tenor.length, self._tenor.units,
                self._convention, self._eom,
            )
        self._earliest_date = earliest
        self._latest_date = latest
        self._maturity_date = latest
        self._latest_relevant_date = latest
        self._pillar_date = latest

    # --- inspectors ----------------------------------------------------------

    def spot(self) -> float:
        return self._spot.value()

    def tenor(self) -> Period:
        return self._tenor

    def fixing_days(self) -> int:
        return self._fixing_days

    def calendar(self) -> Calendar:
        return self._calendar

    def business_day_convention(self) -> BusinessDayConvention:
        return self._convention

    def end_of_month(self) -> bool:
        return self._eom

    def is_fx_base_currency_collateral_currency(self) -> bool:
        return self._is_base_collateral

    def trading_calendar(self) -> Calendar | None:
        return self._trading_calendar

    def adjustment_calendar(self) -> Calendar:
        return self._joint_calendar
