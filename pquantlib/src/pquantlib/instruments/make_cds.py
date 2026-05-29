"""MakeCDS — fluent factory for CreditDefaultSwap.

# C++ parity: ql/instruments/makecds.{hpp,cpp} ``MakeCreditDefaultSwap`` (v1.42.1).

The C++ class is a chained builder with three constructor overloads
(``MakeCreditDefaultSwap(tenor, runningSpread)``,
``MakeCreditDefaultSwap(termDate, runningSpread)``,
``MakeCreditDefaultSwap(schedule, runningSpread)``) and an ``operator
CreditDefaultSwap()`` / ``operator shared_ptr<CreditDefaultSwap>``
conversion at the end. PQuantLib ports the fluent surface as a Python
class with chainable ``with_*()`` setters and a terminal ``build()``
method that produces the CDS.

Use either:
- ``MakeCDS(tenor=Period(5,Years), running_spread=0.02)``
- ``MakeCDS(termination_date=Date(...), running_spread=0.02)``
- ``MakeCDS(schedule=Schedule(...), running_spread=0.02)``

Then chain ``.with_side(Side.Buyer).with_notional(10e6) ... .build()``.

# C++ parity divergence: the C++ class uses ``cdsMaturity(tradeDate,
   tenor, rule)`` to anchor CDS / CDS2015 / OldCDS rules to standard
   IMM dates. PQuantLib doesn't yet expose ``cds_maturity``; the
   Python factory falls back to ``trade_date + tenor`` for those
   rules. The non-CDS rules (Forward / Backward / etc.) are
   bit-identical to C++.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.claim import Claim
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    ProtectionSide,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.weekends_only import WeekendsOnly
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

_NULL_DATE: Date = Date()


class MakeCDS:
    """Fluent factory for :class:`CreditDefaultSwap`.

    # C++ parity: ``MakeCreditDefaultSwap`` (makecds.hpp).

    Three construction modes:
    1. ``MakeCDS(termination_date=..., running_spread=...)``: builds an
       internal schedule from protection_start to termination_date.
    2. ``MakeCDS(tenor=..., running_spread=...)``: schedule spans
       ``tenor`` starting at trade_date (or trade_date+1 for non-CDS
       rules).
    3. ``MakeCDS(schedule=..., running_spread=...)``: caller-supplied
       schedule.

    The ``ibor_index_for_calendar`` optional argument bridges the C++
    convention of taking an IborIndex to source the calendar. If None,
    ``WeekendsOnly()`` is used (matches C++ default).
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
        """One of ``termination_date`` / ``tenor`` / ``schedule`` must be set."""
        provided = sum(
            x is not None
            for x in (termination_date, tenor, schedule)
        )
        qassert.require(
            provided == 1,
            "MakeCDS: provide exactly one of (termination_date, tenor, schedule)",
        )

        self._termination_date: Date | None = termination_date
        self._tenor: Period | None = tenor
        self._schedule: Schedule | None = schedule
        self._running_spread: float = running_spread
        self._ibor_index_for_calendar: IborIndex | None = ibor_index_for_calendar

        # Defaults mirror C++ MakeCreditDefaultSwap private members.
        self._side: ProtectionSide = ProtectionSide.Buyer
        self._nominal: float = 1.0
        self._upfront_rate: float = 0.0
        self._coupon_tenor: Period = Period(3, TimeUnit.Months)
        self._rule: DateGeneration = DateGeneration.CDS
        self._convention: BusinessDayConvention = BusinessDayConvention.Following
        self._termination_date_convention: BusinessDayConvention = (
            BusinessDayConvention.Unadjusted
        )
        # C++ default DayCounter is Actual360(); lastPeriodDayCounter is
        # Actual360(includeLastDay=True). PQuantLib's Actual360 accepts
        # an include_last bool.
        self._day_counter: DayCounter = Actual360()
        self._last_period_day_counter: DayCounter = Actual360(include_last_day=True)
        self._settles_accrual: bool = True
        self._pays_at_default_time: bool = True
        self._protection_start: Date | None = None
        self._upfront_date: Date | None = None
        self._claim: Claim | None = None
        self._rebates_accrual: bool = True
        self._trade_date: Date | None = None
        self._cash_settlement_days: int = 3
        self._calendar: Calendar | None = (
            ibor_index_for_calendar.fixing_calendar()
            if ibor_index_for_calendar is not None
            else None
        )
        self._engine: PricingEngine | None = None

    # ---- chainable setters ------------------------------------------------

    def with_side(self, side: ProtectionSide) -> MakeCDS:
        self._side = side
        return self

    def with_notional(self, n: float) -> MakeCDS:
        self._nominal = n
        return self

    def with_upfront_rate(self, r: float) -> MakeCDS:
        self._upfront_rate = r
        return self

    def with_coupon_tenor(self, t: Period) -> MakeCDS:
        self._coupon_tenor = t
        return self

    def with_calendar(self, cal: Calendar) -> MakeCDS:
        self._calendar = cal
        return self

    def with_convention(self, bdc: BusinessDayConvention) -> MakeCDS:
        self._convention = bdc
        return self

    def with_termination_date_convention(
        self, bdc: BusinessDayConvention,
    ) -> MakeCDS:
        self._termination_date_convention = bdc
        return self

    def with_rule(self, rule: DateGeneration) -> MakeCDS:
        self._rule = rule
        return self

    def with_day_counter(self, dc: DayCounter) -> MakeCDS:
        self._day_counter = dc
        return self

    def with_last_period_day_counter(self, dc: DayCounter) -> MakeCDS:
        self._last_period_day_counter = dc
        return self

    def settles_accrual(self, b: bool = True) -> MakeCDS:
        self._settles_accrual = b
        return self

    def pays_at_default_time(self, b: bool = True) -> MakeCDS:
        self._pays_at_default_time = b
        return self

    def with_protection_start(self, d: Date) -> MakeCDS:
        self._protection_start = d
        return self

    def with_upfront_date(self, d: Date) -> MakeCDS:
        self._upfront_date = d
        return self

    def with_claim(self, claim: Claim) -> MakeCDS:
        self._claim = claim
        return self

    def rebates_accrual(self, b: bool = True) -> MakeCDS:
        self._rebates_accrual = b
        return self

    def with_trade_date(self, d: Date) -> MakeCDS:
        self._trade_date = d
        return self

    def with_cash_settlement_days(self, n: int) -> MakeCDS:
        self._cash_settlement_days = n
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeCDS:
        self._engine = engine
        return self

    # ---- terminal ---------------------------------------------------------

    def build(self) -> CreditDefaultSwap:
        """Construct the :class:`CreditDefaultSwap`.

        # C++ parity: ``operator shared_ptr<CreditDefaultSwap>()``
        # (makecds.cpp:47-92).
        """
        cal = self._calendar if self._calendar is not None else WeekendsOnly()
        trade_date = (
            self._trade_date if self._trade_date is not None and self._trade_date != _NULL_DATE
            else ObservableSettings().evaluation_date_or_today()
        )
        upfront_date = (
            self._upfront_date if self._upfront_date is not None
            and self._upfront_date != _NULL_DATE
            else cal.advance(trade_date, self._cash_settlement_days, TimeUnit.Days)
        )

        # Default protection_start per C++ makecds.cpp:53-63.
        protection_start = self._protection_start
        if protection_start is None or protection_start == _NULL_DATE:
            if self._schedule is not None:
                protection_start = self._schedule.date(0)
            elif self._rule in (DateGeneration.CDS, DateGeneration.CDS2015):
                protection_start = trade_date
            else:
                protection_start = trade_date + 1

        # Build the schedule if not user-supplied.
        if self._schedule is not None:
            schedule = self._schedule
        else:
            if self._tenor is not None:
                # # C++ parity divergence: cdsMaturity is deferred — for
                # CDS / CDS2015 / OldCDS we fall back to trade_date+tenor.
                end = trade_date + self._tenor
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

        if self._upfront_rate != 0.0:
            cds = CreditDefaultSwap.with_upfront(
                self._side, self._nominal, self._upfront_rate, self._running_spread,
                schedule, self._convention, self._day_counter,
                self._settles_accrual, self._pays_at_default_time,
                protection_start, upfront_date,
                self._claim, self._last_period_day_counter,
                self._rebates_accrual, trade_date,
                self._cash_settlement_days,
            )
        else:
            cds = CreditDefaultSwap(
                self._side, self._nominal, self._running_spread, schedule,
                self._convention, self._day_counter,
                self._settles_accrual, self._pays_at_default_time,
                protection_start, self._claim, self._last_period_day_counter,
                self._rebates_accrual, trade_date, self._cash_settlement_days,
            )

        if self._engine is not None:
            cds.set_pricing_engine(self._engine)
        return cds

    # ---- conversion-protocol fallback ------------------------------------

    def __call__(self) -> CreditDefaultSwap:
        """Allow ``MakeCDS(...).with_...()`` followed by ``()``.

        Mirrors the C++ ``operator CreditDefaultSwap()`` conversion.
        """
        return self.build()


def make_cds(
    termination_date: Date | None = None,
    tenor: Period | None = None,
    schedule: Schedule | None = None,
    running_spread: float = 0.01,
    ibor_index_for_calendar: IborIndex | None = None,
) -> MakeCDS:
    """Free-function alias for :class:`MakeCDS` construction.

    Identical to ``MakeCDS(...)`` but uses the leading-positional style
    matching :func:`make_vanilla_swap` from L3-C.
    """
    return MakeCDS(
        termination_date=termination_date, tenor=tenor, schedule=schedule,
        running_spread=running_spread,
        ibor_index_for_calendar=ibor_index_for_calendar,
    )


# Some callers prefer the C++ frequency-based coupon tenor — expose a
# convenience for it.
def coupon_tenor_from_frequency(freq: Frequency) -> Period:
    return Period.from_frequency(freq)


__all__ = ["MakeCDS", "coupon_tenor_from_frequency", "make_cds"]
