"""MakeCreditDefaultSwap — fluent builder for standard market CDS.

# C++ parity: ql/instruments/makecds.{hpp,cpp} ``MakeCreditDefaultSwap`` (v1.43).

The C++ class has three constructor overloads —
``MakeCreditDefaultSwap(tenor, runningSpread)``,
``MakeCreditDefaultSwap(termDate, runningSpread)`` and
``MakeCreditDefaultSwap(schedule, runningSpread)`` — plus an ``operator
CreditDefaultSwap()`` conversion. PQuantLib ports the three overloads as three
mutually-exclusive keyword arguments and the conversion as
:meth:`MakeCreditDefaultSwap.build` (``__call__`` delegates to it).

``MakeCDS`` is kept as an alias of the C++-named class for existing call sites.

Python divergences from C++:

- ``Settings::instance().evaluationDate()`` is
  :meth:`~pquantlib.patterns.observable_settings.ObservableSettings.evaluation_date_or_today`.
- C++ hard-codes ``WeekendsOnly()`` both as the generated schedule's calendar
  and for the cash-settlement advance, and hard-codes ``Unadjusted`` as the
  schedule's termination-date convention. This port keeps those as the defaults
  of two extra setters, :meth:`with_calendar` and
  :meth:`with_termination_date_convention`, so leaving them alone reproduces C++
  exactly.
- Several setters carry a second, non-C++ name (``with_notional``,
  ``settles_accrual``, ``pays_at_default_time``, ``rebates_accrual``,
  ``with_rule``) because those were this port's original spellings; they simply
  delegate to the C++-named method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
    cds_maturity,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.instruments.claim import Claim
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.time.calendar import Calendar

_NULL_DATE: Date = Date()
_CDS_RULES: tuple[DateGeneration, ...] = (
    DateGeneration.CDS2015,
    DateGeneration.CDS,
    DateGeneration.OldCDS,
)
_POST_BIG_BANG_RULES: tuple[DateGeneration, ...] = (
    DateGeneration.CDS2015,
    DateGeneration.CDS,
)


class MakeCreditDefaultSwap:
    """Fluent builder for :class:`~pquantlib.instruments.credit_default_swap.CreditDefaultSwap`.

    # C++ parity: ``MakeCreditDefaultSwap`` (makecds.hpp:35-84).

    Exactly one of ``tenor`` / ``termination_date`` / ``schedule`` must be given
    — they are the three C++ constructor overloads.
    """

    def __init__(
        self,
        *,
        termination_date: Date | None = None,
        tenor: Period | None = None,
        schedule: Schedule | None = None,
        running_spread: float = 0.01,
        ibor_index_for_calendar: IborIndex | None = None,
    ) -> None:
        provided = sum(x is not None for x in (termination_date, tenor, schedule))
        qassert.require(
            provided == 1,
            "MakeCDS: provide exactly one of (termination_date, tenor, schedule)",
        )

        self._termination_date: Date | None = termination_date
        self._tenor: Period | None = tenor
        self._schedule: Schedule | None = schedule
        self._running_spread: float = running_spread
        self._ibor_index_for_calendar: IborIndex | None = ibor_index_for_calendar

        # Defaults mirror the C++ private members (makecds.hpp:66-83).
        self._side: ProtectionSide = ProtectionSide.Buyer
        self._nominal: float = 1.0
        self._upfront_rate: float = 0.0
        self._coupon_tenor: Period = Period(3, TimeUnit.Months)
        self._rule: DateGeneration = DateGeneration.CDS
        self._convention: BusinessDayConvention = BusinessDayConvention.Following
        self._day_counter: DayCounter = Actual360()
        self._settles_accrual: bool = True
        self._pays_at_default_time: bool = True
        self._protection_start: Date | None = None
        self._upfront_date: Date | None = None
        self._claim: Claim | None = None
        self._last_period_day_counter: DayCounter = Actual360(include_last_day=True)
        self._rebates_accrual: bool = True
        self._trade_date: Date | None = None
        self._cash_settlement_days: int = 3
        self._engine: PricingEngine | None = None

        # PQuantLib extensions; the defaults are C++'s hard-coded values.
        self._termination_date_convention: BusinessDayConvention = (
            BusinessDayConvention.Unadjusted
        )
        self._calendar: Calendar | None = (
            ibor_index_for_calendar.fixing_calendar()
            if ibor_index_for_calendar is not None
            else None
        )

    # --- chained setters (C++ names) ---------------------------------------

    def with_side(self, side: ProtectionSide) -> MakeCreditDefaultSwap:
        self._side = side
        return self

    def with_nominal(self, nominal: float) -> MakeCreditDefaultSwap:
        self._nominal = nominal
        return self

    def with_upfront_rate(self, upfront_rate: float) -> MakeCreditDefaultSwap:
        self._upfront_rate = upfront_rate
        return self

    def with_coupon_tenor(self, coupon_tenor: Period) -> MakeCreditDefaultSwap:
        self._coupon_tenor = coupon_tenor
        return self

    def with_date_generation_rule(self, rule: DateGeneration) -> MakeCreditDefaultSwap:
        self._rule = rule
        return self

    def with_convention(self, convention: BusinessDayConvention) -> MakeCreditDefaultSwap:
        self._convention = convention
        return self

    def with_day_counter(self, day_counter: DayCounter) -> MakeCreditDefaultSwap:
        self._day_counter = day_counter
        return self

    def settle_accrual(self, b: bool = True) -> MakeCreditDefaultSwap:
        self._settles_accrual = b
        return self

    def pay_at_default_time(self, b: bool = True) -> MakeCreditDefaultSwap:
        self._pays_at_default_time = b
        return self

    def with_protection_start(self, d: Date) -> MakeCreditDefaultSwap:
        self._protection_start = None if d == _NULL_DATE else d
        return self

    def with_upfront_date(self, d: Date) -> MakeCreditDefaultSwap:
        self._upfront_date = None if d == _NULL_DATE else d
        return self

    def with_claim(self, claim: Claim) -> MakeCreditDefaultSwap:
        self._claim = claim
        return self

    def with_last_period_day_counter(
        self, last_period_day_counter: DayCounter
    ) -> MakeCreditDefaultSwap:
        self._last_period_day_counter = last_period_day_counter
        return self

    def rebate_accrual(self, b: bool = True) -> MakeCreditDefaultSwap:
        self._rebates_accrual = b
        return self

    def with_trade_date(self, trade_date: Date) -> MakeCreditDefaultSwap:
        self._trade_date = None if trade_date == _NULL_DATE else trade_date
        return self

    def with_cash_settlement_days(self, cash_settlement_days: int) -> MakeCreditDefaultSwap:
        self._cash_settlement_days = cash_settlement_days
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeCreditDefaultSwap:
        self._engine = engine
        return self

    # --- chained setters (PQuantLib extensions / legacy spellings) ---------

    def with_calendar(self, calendar: Calendar) -> MakeCreditDefaultSwap:
        """Calendar for the generated schedule and the cash-settlement advance.

        PQuantLib extension: C++ hard-codes ``WeekendsOnly()`` for both, which is
        also the default here.
        """
        self._calendar = calendar
        return self

    def with_termination_date_convention(
        self, bdc: BusinessDayConvention
    ) -> MakeCreditDefaultSwap:
        """PQuantLib extension: C++ hard-codes ``Unadjusted``, the default here."""
        self._termination_date_convention = bdc
        return self

    def with_notional(self, n: float) -> MakeCreditDefaultSwap:
        """Legacy spelling of :meth:`with_nominal` (C++ ``withNominal``)."""
        return self.with_nominal(n)

    def with_rule(self, rule: DateGeneration) -> MakeCreditDefaultSwap:
        """Legacy spelling of :meth:`with_date_generation_rule`."""
        return self.with_date_generation_rule(rule)

    def settles_accrual(self, b: bool = True) -> MakeCreditDefaultSwap:
        """Legacy spelling of :meth:`settle_accrual` (C++ ``settleAccrual``)."""
        return self.settle_accrual(b)

    def pays_at_default_time(self, b: bool = True) -> MakeCreditDefaultSwap:
        """Legacy spelling of :meth:`pay_at_default_time` (C++ ``payAtDefaultTime``)."""
        return self.pay_at_default_time(b)

    def rebates_accrual(self, b: bool = True) -> MakeCreditDefaultSwap:
        """Legacy spelling of :meth:`rebate_accrual` (C++ ``rebateAccrual``)."""
        return self.rebate_accrual(b)

    # --- build -------------------------------------------------------------

    def build(self) -> CreditDefaultSwap:
        """Construct the CDS.

        # C++ parity: ``operator ext::shared_ptr<CreditDefaultSwap>()``
        (makecds.cpp:44-92).
        """
        cal = self._calendar if self._calendar is not None else WeekendsOnly()
        trade_date = (
            self._trade_date
            if self._trade_date is not None
            else ObservableSettings().evaluation_date_or_today()
        )
        upfront_date = (
            self._upfront_date
            if self._upfront_date is not None
            else cal.advance(trade_date, self._cash_settlement_days, TimeUnit.Days)
        )

        protection_start = self._protection_start
        if protection_start is None:
            if self._schedule is not None:
                protection_start = self._schedule.date(0)
            elif self._rule in _POST_BIG_BANG_RULES:
                protection_start = trade_date
            else:
                protection_start = trade_date + 1

        if self._schedule is not None:
            schedule = self._schedule
        else:
            if self._tenor is not None:
                end = (
                    cds_maturity(trade_date, self._tenor, self._rule)
                    if self._rule in _CDS_RULES
                    else trade_date + self._tenor
                )
            else:
                assert self._termination_date is not None
                end = self._termination_date
            schedule = Schedule.from_rule(
                effective_date=protection_start,
                termination_date=end,
                tenor=self._coupon_tenor,
                calendar=cal,
                convention=self._convention,
                termination_date_convention=self._termination_date_convention,
                rule=self._rule,
                end_of_month=False,
            )

        # C++ always uses the upfront constructor, with upfrontRate_ defaulting
        # to 0.0 — so the upfront date reaches the instrument (and therefore the
        # accrual-rebate payment date) even when no upfront rate was set.
        cds = CreditDefaultSwap.with_upfront(
            self._side,
            self._nominal,
            self._upfront_rate,
            self._running_spread,
            schedule,
            self._convention,
            self._day_counter,
            self._settles_accrual,
            self._pays_at_default_time,
            protection_start,
            upfront_date,
            self._claim,
            self._last_period_day_counter,
            self._rebates_accrual,
            trade_date,
            self._cash_settlement_days,
        )
        if self._engine is not None:
            cds.set_pricing_engine(self._engine)
        return cds

    def __call__(self) -> CreditDefaultSwap:
        """Alias for :meth:`build`, mirroring C++ ``operator CreditDefaultSwap()``."""
        return self.build()


#: Legacy alias — this port's original name for ``MakeCreditDefaultSwap``.
MakeCDS = MakeCreditDefaultSwap


def make_cds(
    termination_date: Date | None = None,
    tenor: Period | None = None,
    schedule: Schedule | None = None,
    running_spread: float = 0.01,
    ibor_index_for_calendar: IborIndex | None = None,
) -> MakeCreditDefaultSwap:
    """Positional-friendly façade over :class:`MakeCreditDefaultSwap`.

    Returns the builder, not the instrument — chain ``with_*`` setters and call
    ``build()`` as usual.
    """
    return MakeCreditDefaultSwap(
        termination_date=termination_date,
        tenor=tenor,
        schedule=schedule,
        running_spread=running_spread,
        ibor_index_for_calendar=ibor_index_for_calendar,
    )


def coupon_tenor_from_frequency(freq: Frequency) -> Period:
    """Convenience for callers holding a :class:`Frequency` rather than a tenor."""
    return Period.from_frequency(freq)


__all__ = [
    "MakeCDS",
    "MakeCreditDefaultSwap",
    "coupon_tenor_from_frequency",
    "make_cds",
]
