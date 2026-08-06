"""MakeVanillaSwap — fluent builder for market-standard vanilla swaps.

# C++ parity: ql/instruments/makevanillaswap.{hpp,cpp} (v1.43).

The C++ class is a chained builder ending in an ``operator VanillaSwap()``
conversion. PQuantLib ports it as a class with chained ``with_*()`` setters and
a terminal :meth:`MakeVanillaSwap.build` (``__call__`` delegates to it), the
same shape as ``MakeSchedule`` in :mod:`pquantlib.time.schedule`.

:func:`make_vanilla_swap` is a keyword-argument façade over the class — it does
no work of its own, so there is exactly one implementation of the resolution
logic (spot date, end date, currency-driven fixed-leg defaults, fair-rate
fallback, engine wiring).

Python divergences from C++:

- ``Settings::instance().evaluationDate()`` is
  :meth:`pquantlib.patterns.observable_settings.ObservableSettings.evaluation_date_or_today`.
  :meth:`MakeVanillaSwap.with_evaluation_date` additionally lets a caller pin
  the reference date locally without mutating the global — the rate helpers
  bootstrap against a curve's reference date and must not touch global state.
- C++ spells "unset" as ``Date()`` / ``Period()`` / ``Null<Natural>()``; this
  port uses ``None`` internally. The setters still accept the C++ null date and
  normalise it.
- When no engine and no discount curve are given and the index carries no
  forwarding curve, C++ still attaches a ``DiscountingSwapEngine`` over the
  empty handle (which throws at pricing time); this port leaves the swap
  without an engine (which also throws at pricing time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.asia import HKDCurrency, JPYCurrency, THBCurrency
from pquantlib.currencies.europe import CHFCurrency, EURCurrency, GBPCurrency, SEKCurrency
from pquantlib.currencies.oceania import AUDCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule, allows_end_of_month
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.currencies.currency import Currency
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.calendar import Calendar

_ZERO_PERIOD: Period = Period(0, TimeUnit.Days)
_NULL_DATE: Date = Date()

_ONE_YEAR: Period = Period(1, TimeUnit.Years)
_SIX_MONTHS: Period = Period(6, TimeUnit.Months)
_THREE_MONTHS: Period = Period(3, TimeUnit.Months)
_FOUR_YEARS: Period = Period(4, TimeUnit.Years)

# C++ ``(12 * days + 182) / 365`` — 182 is 365/2, i.e. round-to-nearest month.
_HALF_YEAR_DAYS: int = 182
_YEAR_DAYS: int = 365
_MONTHS_PER_YEAR: int = 12


def _at_most(p: Period, q: Period) -> bool:
    """``p <= q`` with C++ semantics — ``!(q < p)`` (period.hpp:174-176).

    ``Period.__le__`` is ``self < other or self == other`` and ``Period.__eq__``
    is the frozen-dataclass field-wise comparison, so ``Period(12, Months)`` is
    neither less than nor equal to ``Period(1, Years)`` in this port even though
    C++ says both. The currency switch below compares an inferred
    months-expressed tenor against ``1 * Years``, so it needs the C++ spelling.
    """
    return not q < p


def _at_least(p: Period, q: Period) -> bool:
    """``p >= q`` with C++ semantics — ``!(p < q)`` (period.hpp:178-180)."""
    return not p < q


def _default_fixed_tenor(ccy: Currency, tenor: Period) -> Period:
    """Currency-driven default fixed-leg tenor.

    # C++ parity: makevanillaswap.cpp:117-134 — the currency switch inside
    ``MakeVanillaSwap::operator ext::shared_ptr<VanillaSwap>()``.
    """
    if ccy in (EURCurrency(), USDCurrency(), CHFCurrency(), SEKCurrency()) or (
        ccy == GBPCurrency() and _at_most(tenor, _ONE_YEAR)
    ):
        return _ONE_YEAR
    if (
        (ccy == GBPCurrency() and not _at_most(tenor, _ONE_YEAR))
        or ccy == JPYCurrency()
        or (ccy == AUDCurrency() and _at_least(tenor, _FOUR_YEARS))
    ):
        return _SIX_MONTHS
    if ccy == HKDCurrency() or (ccy == AUDCurrency() and not _at_least(tenor, _FOUR_YEARS)):
        return _THREE_MONTHS
    qassert.fail(f"unknown fixed leg default tenor for {ccy.code}")


def _default_fixed_day_count(ccy: Currency) -> DayCounter:
    """Currency-driven default fixed-leg day counter.

    # C++ parity: makevanillaswap.cpp:150-162.
    """
    if ccy == USDCurrency():
        return Actual360()
    if ccy in (EURCurrency(), CHFCurrency(), SEKCurrency()):
        return Thirty360(Thirty360Convention.BondBasis)
    if ccy in (
        GBPCurrency(),
        JPYCurrency(),
        AUDCurrency(),
        HKDCurrency(),
        THBCurrency(),
    ):
        return Actual365Fixed()
    qassert.fail(f"unknown fixed leg day counter for {ccy.code}")


class MakeVanillaSwap:
    """Fluent builder for :class:`~pquantlib.instruments.vanilla_swap.VanillaSwap`.

    # C++ parity: ``MakeVanillaSwap`` (makevanillaswap.hpp:38-118).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns ``self``;
    C++'s ``operator VanillaSwap()`` is :meth:`build`.
    """

    def __init__(
        self,
        swap_tenor: Period,
        ibor_index: IborIndex,
        fixed_rate: float | None = None,
        forward_start: Period = _ZERO_PERIOD,
    ) -> None:
        # C++ parity: makevanillaswap.cpp:39-49 — the calendars, float tenor,
        # float conventions and float day count all default off the index.
        self._swap_tenor: Period | None = swap_tenor
        self._ibor_index: IborIndex = ibor_index
        self._fixed_rate: float | None = fixed_rate
        self._forward_start: Period = forward_start

        self._settlement_days: int | None = None
        self._effective_date: Date | None = None
        self._termination_date: Date | None = None
        self._fixed_calendar: Calendar = ibor_index.fixing_calendar()
        self._float_calendar: Calendar = ibor_index.fixing_calendar()

        self._type: SwapType = SwapType.Payer
        self._nominal: float = 1.0
        self._fixed_tenor: Period | None = None
        self._float_tenor: Period = ibor_index.tenor()
        self._fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._fixed_termination_date_convention: BusinessDayConvention = (
            BusinessDayConvention.ModifiedFollowing
        )
        self._float_convention: BusinessDayConvention = ibor_index.business_day_convention()
        self._float_termination_date_convention: BusinessDayConvention = (
            ibor_index.business_day_convention()
        )
        self._fixed_rule: DateGeneration = DateGeneration.Backward
        self._float_rule: DateGeneration = DateGeneration.Backward
        self._fixed_end_of_month: bool = False
        self._float_end_of_month: bool = False
        self._maturity_end_of_month: bool | None = None
        self._fixed_first_date: Date = _NULL_DATE
        self._fixed_next_to_last_date: Date = _NULL_DATE
        self._float_first_date: Date = _NULL_DATE
        self._float_next_to_last_date: Date = _NULL_DATE
        self._float_spread: float = 0.0
        self._fixed_day_count: DayCounter | None = None
        self._float_day_count: DayCounter = ibor_index.day_counter()
        self._use_indexed_coupons: bool | None = None
        self._payment_convention: BusinessDayConvention | None = None

        self._engine: PricingEngine | None = None
        # PQuantLib addition — see the module docstring.
        self._evaluation_date: Date | None = None

    # --- chained setters ---------------------------------------------------

    def receive_fixed(self, flag: bool = True) -> MakeVanillaSwap:
        self._type = SwapType.Receiver if flag else SwapType.Payer
        return self

    def with_type(self, swap_type: SwapType) -> MakeVanillaSwap:
        self._type = swap_type
        return self

    def with_nominal(self, n: float) -> MakeVanillaSwap:
        self._nominal = n
        return self

    def with_settlement_days(self, settlement_days: int) -> MakeVanillaSwap:
        self._settlement_days = settlement_days
        return self

    def with_effective_date(self, effective_date: Date) -> MakeVanillaSwap:
        self._effective_date = None if effective_date == _NULL_DATE else effective_date
        return self

    def with_termination_date(self, termination_date: Date) -> MakeVanillaSwap:
        """Set the end date; a non-null one clears the swap tenor.

        # C++ parity: makevanillaswap.cpp:212-218.
        """
        if termination_date == _NULL_DATE:
            self._termination_date = None
        else:
            self._termination_date = termination_date
            self._swap_tenor = None
        return self

    def with_rule(self, rule: DateGeneration) -> MakeVanillaSwap:
        self._fixed_rule = rule
        self._float_rule = rule
        return self

    def with_payment_convention(self, bdc: BusinessDayConvention) -> MakeVanillaSwap:
        self._payment_convention = bdc
        return self

    def with_fixed_leg_tenor(self, tenor: Period) -> MakeVanillaSwap:
        self._fixed_tenor = tenor
        return self

    def with_fixed_leg_calendar(self, calendar: Calendar) -> MakeVanillaSwap:
        self._fixed_calendar = calendar
        return self

    def with_fixed_leg_convention(self, bdc: BusinessDayConvention) -> MakeVanillaSwap:
        self._fixed_convention = bdc
        return self

    def with_fixed_leg_termination_date_convention(
        self, bdc: BusinessDayConvention
    ) -> MakeVanillaSwap:
        self._fixed_termination_date_convention = bdc
        return self

    def with_fixed_leg_rule(self, rule: DateGeneration) -> MakeVanillaSwap:
        self._fixed_rule = rule
        return self

    def with_fixed_leg_end_of_month(self, flag: bool = True) -> MakeVanillaSwap:
        self._fixed_end_of_month = flag
        return self

    def with_fixed_leg_first_date(self, d: Date) -> MakeVanillaSwap:
        self._fixed_first_date = d
        return self

    def with_fixed_leg_next_to_last_date(self, d: Date) -> MakeVanillaSwap:
        self._fixed_next_to_last_date = d
        return self

    def with_fixed_leg_day_count(self, day_counter: DayCounter) -> MakeVanillaSwap:
        self._fixed_day_count = day_counter
        return self

    def with_floating_leg_tenor(self, tenor: Period) -> MakeVanillaSwap:
        self._float_tenor = tenor
        return self

    def with_floating_leg_calendar(self, calendar: Calendar) -> MakeVanillaSwap:
        self._float_calendar = calendar
        return self

    def with_floating_leg_convention(self, bdc: BusinessDayConvention) -> MakeVanillaSwap:
        self._float_convention = bdc
        return self

    def with_floating_leg_termination_date_convention(
        self, bdc: BusinessDayConvention
    ) -> MakeVanillaSwap:
        self._float_termination_date_convention = bdc
        return self

    def with_floating_leg_rule(self, rule: DateGeneration) -> MakeVanillaSwap:
        self._float_rule = rule
        return self

    def with_floating_leg_end_of_month(self, flag: bool = True) -> MakeVanillaSwap:
        self._float_end_of_month = flag
        return self

    def with_maturity_end_of_month(self, flag: bool = True) -> MakeVanillaSwap:
        self._maturity_end_of_month = flag
        return self

    def with_floating_leg_first_date(self, d: Date) -> MakeVanillaSwap:
        self._float_first_date = d
        return self

    def with_floating_leg_next_to_last_date(self, d: Date) -> MakeVanillaSwap:
        self._float_next_to_last_date = d
        return self

    def with_floating_leg_day_count(self, day_counter: DayCounter) -> MakeVanillaSwap:
        self._float_day_count = day_counter
        return self

    def with_floating_leg_spread(self, spread: float) -> MakeVanillaSwap:
        self._float_spread = spread
        return self

    def with_discounting_term_structure(
        self, discount_curve: YieldTermStructureProtocol
    ) -> MakeVanillaSwap:
        self._engine = DiscountingSwapEngine(discount_curve, include_settlement_date_flows=False)
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeVanillaSwap:
        self._engine = engine
        return self

    def with_indexed_coupons(self, flag: bool | None = True) -> MakeVanillaSwap:
        self._use_indexed_coupons = flag
        return self

    def with_at_par_coupons(self, flag: bool = True) -> MakeVanillaSwap:
        self._use_indexed_coupons = not flag
        return self

    def with_evaluation_date(self, d: Date) -> MakeVanillaSwap:
        """Pin the reference date used for the spot-date calculation.

        PQuantLib addition (no C++ counterpart): C++ reads
        ``Settings::instance().evaluationDate()``, which this builder also does
        by default. This setter lets a caller — a rate helper bootstrapping
        against a curve's reference date, say — override it without mutating
        global state.
        """
        self._evaluation_date = d
        return self

    # --- build -------------------------------------------------------------

    def _start_date(self) -> Date:
        """Resolve the swap's start date.

        # C++ parity: makevanillaswap.cpp:60-91.
        """
        if self._effective_date is not None:
            return self._effective_date

        ref_date = (
            self._evaluation_date
            if self._evaluation_date is not None
            else ObservableSettings().evaluation_date_or_today()
        )
        if self._settlement_days is None:
            # The spot date is defined by the index, so the reference date must
            # be adjusted on the index fixing calendar (not the float/payment
            # calendar) to stay consistent with value_date's own advance.
            spot_date = self._ibor_index.value_date(
                self._ibor_index.fixing_calendar().adjust(ref_date)
            )
        else:
            # An explicit settlement-day count is advanced on the float/payment
            # calendar, so adjust the reference date there.
            spot_date = self._float_calendar.advance(
                self._float_calendar.adjust(ref_date), self._settlement_days, TimeUnit.Days
            )
        start_date = spot_date + self._forward_start
        if self._forward_start.length < 0:
            return self._float_calendar.adjust(start_date, BusinessDayConvention.Preceding)
        if self._forward_start.length > 0:
            return self._float_calendar.adjust(start_date, BusinessDayConvention.Following)
        return start_date

    def _end_date(self, start_date: Date) -> Date:
        """Resolve the swap's end date.

        # C++ parity: makevanillaswap.cpp:93-102.
        """
        if self._termination_date is not None:
            return self._termination_date
        assert self._swap_tenor is not None
        end_date = start_date + self._swap_tenor
        maturity_end_of_month = (
            self._maturity_end_of_month
            if self._maturity_end_of_month is not None
            else self._float_end_of_month
        )
        if (
            maturity_end_of_month
            and allows_end_of_month(self._swap_tenor)
            and self._float_calendar.is_end_of_month(start_date)
        ):
            end_date = self._float_calendar.end_of_month(end_date)
        return end_date

    def _effective_fixed_tenor(self, start_date: Date, end_date: Date) -> Period:
        """Fixed-leg tenor: explicit if set, else currency-driven.

        # C++ parity: makevanillaswap.cpp:105-135. When ``with_termination_date``
        cleared the swap tenor, the currency switch is fed a tenor approximated
        from the actual swap length.
        """
        if self._fixed_tenor is not None:
            return self._fixed_tenor
        tenor = self._swap_tenor
        if tenor is None:
            tenor = _ZERO_PERIOD
            if end_date > start_date:
                months = (
                    _MONTHS_PER_YEAR * (end_date - start_date) + _HALF_YEAR_DAYS
                ) // _YEAR_DAYS
                tenor = Period(months, TimeUnit.Months)
        return _default_fixed_tenor(self._ibor_index.currency(), tenor)

    def _discount_curve(self) -> YieldTermStructureProtocol | None:
        return self._ibor_index.forecast_term_structure()

    def build(self) -> VanillaSwap:
        """Construct the swap.

        # C++ parity: ``MakeVanillaSwap::operator ext::shared_ptr<VanillaSwap>()``
        (makevanillaswap.cpp:56-195).
        """
        qassert.require(
            self._effective_date is None or self._settlement_days is None,
            "cannot set both an explicit effective date and settlement days; "
            "use one or the other",
        )
        start_date = self._start_date()
        end_date = self._end_date(start_date)

        currency = self._ibor_index.currency()
        fixed_tenor = self._effective_fixed_tenor(start_date, end_date)
        fixed_day_count = (
            self._fixed_day_count
            if self._fixed_day_count is not None
            else _default_fixed_day_count(currency)
        )

        fixed_schedule = Schedule.from_rule(
            start_date,
            end_date,
            fixed_tenor,
            self._fixed_calendar,
            self._fixed_convention,
            self._fixed_termination_date_convention,
            self._fixed_rule,
            self._fixed_end_of_month,
            self._fixed_first_date,
            self._fixed_next_to_last_date,
        )
        float_schedule = Schedule.from_rule(
            start_date,
            end_date,
            self._float_tenor,
            self._float_calendar,
            self._float_convention,
            self._float_termination_date_convention,
            self._float_rule,
            self._float_end_of_month,
            self._float_first_date,
            self._float_next_to_last_date,
        )

        used_fixed_rate = self._fixed_rate
        if used_fixed_rate is None:
            # C++ prices a 100-nominal probe swap at a 0% fixed rate and reads
            # its fair rate (makevanillaswap.cpp:164-184).
            temp = VanillaSwap(
                self._type,
                100.00,
                fixed_schedule,
                0.0,
                fixed_day_count,
                float_schedule,
                self._ibor_index,
                self._float_spread,
                self._float_day_count,
                payment_convention=self._payment_convention,
                use_indexed_coupons=self._use_indexed_coupons,
            )
            if self._engine is None:
                discount = self._discount_curve()
                qassert.require(
                    discount is not None,
                    f"null term structure set to this instance of {self._ibor_index.name()}",
                )
                assert discount is not None
                temp.set_pricing_engine(
                    DiscountingSwapEngine(discount, include_settlement_date_flows=False)
                )
            else:
                temp.set_pricing_engine(self._engine)
            used_fixed_rate = temp.fair_rate()

        swap = VanillaSwap(
            self._type,
            self._nominal,
            fixed_schedule,
            used_fixed_rate,
            fixed_day_count,
            float_schedule,
            self._ibor_index,
            self._float_spread,
            self._float_day_count,
            payment_convention=self._payment_convention,
            use_indexed_coupons=self._use_indexed_coupons,
        )

        if self._engine is None:
            discount = self._discount_curve()
            if discount is not None:
                swap.set_pricing_engine(
                    DiscountingSwapEngine(discount, include_settlement_date_flows=False)
                )
        else:
            swap.set_pricing_engine(self._engine)
        return swap

    def __call__(self) -> VanillaSwap:
        """Alias for :meth:`build`, mirroring C++ ``operator VanillaSwap()``."""
        return self.build()


def make_vanilla_swap(  # noqa: PLR0915 — one keyword per MakeVanillaSwap setter, deliberately flat
    swap_tenor: Period,
    ibor_index: IborIndex,
    fixed_rate: float | None = None,
    forward_start: Period = _ZERO_PERIOD,
    *,
    nominal: float = 1.0,
    swap_type: SwapType = SwapType.Payer,
    settlement_days: int | None = None,
    effective_date: Date | None = None,
    termination_date: Date | None = None,
    fixed_leg_tenor: Period | None = None,
    fixed_leg_calendar: Calendar | None = None,
    fixed_leg_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    fixed_leg_termination_convention: BusinessDayConvention = (
        BusinessDayConvention.ModifiedFollowing
    ),
    fixed_leg_rule: DateGeneration = DateGeneration.Backward,
    fixed_leg_end_of_month: bool = False,
    fixed_leg_first_date: Date | None = None,
    fixed_leg_next_to_last_date: Date | None = None,
    fixed_leg_day_count: DayCounter | None = None,
    floating_leg_tenor: Period | None = None,
    floating_leg_calendar: Calendar | None = None,
    floating_leg_convention: BusinessDayConvention | None = None,
    floating_leg_termination_convention: BusinessDayConvention | None = None,
    floating_leg_rule: DateGeneration = DateGeneration.Backward,
    floating_leg_end_of_month: bool = False,
    floating_leg_first_date: Date | None = None,
    floating_leg_next_to_last_date: Date | None = None,
    floating_leg_day_count: DayCounter | None = None,
    floating_leg_spread: float = 0.0,
    maturity_end_of_month: bool | None = None,
    payment_convention: BusinessDayConvention | None = None,
    discount_curve: YieldTermStructureProtocol | None = None,
    pricing_engine: PricingEngine | None = None,
    use_indexed_coupons: bool | None = None,
    evaluation_date: Date | None = None,
) -> VanillaSwap:
    """Keyword-argument façade over :class:`MakeVanillaSwap`.

    Every keyword maps to the identically-named chained setter, so the class is
    the single implementation. ``evaluation_date`` defaults to the global
    evaluation date (C++ ``Settings::instance().evaluationDate()``).
    """
    builder = MakeVanillaSwap(swap_tenor, ibor_index, fixed_rate, forward_start)
    builder.with_type(swap_type).with_nominal(nominal)
    builder.with_fixed_leg_convention(fixed_leg_convention)
    builder.with_fixed_leg_termination_date_convention(fixed_leg_termination_convention)
    builder.with_fixed_leg_rule(fixed_leg_rule)
    builder.with_fixed_leg_end_of_month(fixed_leg_end_of_month)
    builder.with_floating_leg_rule(floating_leg_rule)
    builder.with_floating_leg_end_of_month(floating_leg_end_of_month)
    builder.with_floating_leg_spread(floating_leg_spread)
    if settlement_days is not None:
        builder.with_settlement_days(settlement_days)
    if effective_date is not None:
        builder.with_effective_date(effective_date)
    if termination_date is not None:
        builder.with_termination_date(termination_date)
    if fixed_leg_tenor is not None:
        builder.with_fixed_leg_tenor(fixed_leg_tenor)
    if fixed_leg_calendar is not None:
        builder.with_fixed_leg_calendar(fixed_leg_calendar)
    if fixed_leg_first_date is not None:
        builder.with_fixed_leg_first_date(fixed_leg_first_date)
    if fixed_leg_next_to_last_date is not None:
        builder.with_fixed_leg_next_to_last_date(fixed_leg_next_to_last_date)
    if fixed_leg_day_count is not None:
        builder.with_fixed_leg_day_count(fixed_leg_day_count)
    if floating_leg_tenor is not None:
        builder.with_floating_leg_tenor(floating_leg_tenor)
    if floating_leg_calendar is not None:
        builder.with_floating_leg_calendar(floating_leg_calendar)
    if floating_leg_convention is not None:
        builder.with_floating_leg_convention(floating_leg_convention)
    if floating_leg_termination_convention is not None:
        builder.with_floating_leg_termination_date_convention(
            floating_leg_termination_convention
        )
    if floating_leg_first_date is not None:
        builder.with_floating_leg_first_date(floating_leg_first_date)
    if floating_leg_next_to_last_date is not None:
        builder.with_floating_leg_next_to_last_date(floating_leg_next_to_last_date)
    if floating_leg_day_count is not None:
        builder.with_floating_leg_day_count(floating_leg_day_count)
    if maturity_end_of_month is not None:
        builder.with_maturity_end_of_month(maturity_end_of_month)
    if payment_convention is not None:
        builder.with_payment_convention(payment_convention)
    if discount_curve is not None:
        builder.with_discounting_term_structure(discount_curve)
    if pricing_engine is not None:
        builder.with_pricing_engine(pricing_engine)
    if use_indexed_coupons is not None:
        builder.with_indexed_coupons(use_indexed_coupons)
    if evaluation_date is not None:
        builder.with_evaluation_date(evaluation_date)
    return builder.build()


__all__ = ["MakeVanillaSwap", "make_vanilla_swap"]
