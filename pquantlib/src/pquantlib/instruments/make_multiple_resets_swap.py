"""MakeMultipleResetsSwap — fluent builder for multiple-resets swaps.

# C++ parity: ql/instruments/makemultipleresetsswap.{hpp,cpp} (v1.43).

The C++ class is a chained builder ending in an ``operator MultipleResetsSwap()``
conversion; PQuantLib ports it as a class with chained ``with_*()`` setters and a
terminal :meth:`MakeMultipleResetsSwap.build` (``__call__`` delegates to it).

By default the fixed-leg frequency matches the coupon period implied by the
index tenor and ``resets_per_coupon``.

Python divergences from C++:

- ``Settings::instance().evaluationDate()`` is
  :meth:`~pquantlib.patterns.observable_settings.ObservableSettings.evaluation_date_or_today`;
  :meth:`MakeMultipleResetsSwap.with_evaluation_date` pins it locally without
  touching global state (see :mod:`pquantlib.instruments.make_vanilla_swap`).
- C++ spells "unset" as ``Date()`` / ``Period()`` / ``Null<Natural>()`` /
  ``Null<Rate>()``; this port uses ``None`` internally.

Note a sharp edge inherited from C++: the end date is business-day-adjusted
*before* the reset schedule is generated, so a tenor whose unadjusted end lands
on a non-business day seeds the backward generation one roll away from the start
and yields an odd number of reset periods — which then trips
:class:`~pquantlib.instruments.multiple_resets_swap.MultipleResetsSwap`'s
"not a multiple of resets_per_coupon" requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.instruments.multiple_resets_swap import MultipleResetsSwap
from pquantlib.instruments.swap import SwapType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol

_ZERO_PERIOD: Period = Period(0, TimeUnit.Days)
_NULL_DATE: Date = Date()


class MakeMultipleResetsSwap:
    """Fluent builder for :class:`~pquantlib.instruments.multiple_resets_swap.MultipleResetsSwap`.

    # C++ parity: ``MakeMultipleResetsSwap`` (makemultipleresetsswap.hpp:39-82).
    """

    def __init__(
        self,
        tenor: Period,
        ibor_index: IborIndex,
        resets_per_coupon: int,
    ) -> None:
        # C++ parity: makemultipleresetsswap.cpp:29-33.
        self._tenor: Period | None = tenor
        self._ibor_index: IborIndex = ibor_index
        self._resets_per_coupon: int = resets_per_coupon
        self._fixed_rate: float | None = None
        self._forward_start: Period = _ZERO_PERIOD

        self._settlement_days: int | None = None
        self._effective_date: Date | None = None
        self._termination_date: Date | None = None
        self._type: SwapType = SwapType.Payer
        self._nominal: float = 1.0
        self._fixed_frequency: Frequency = Frequency.NoFrequency
        self._fixed_day_count: DayCounter = ibor_index.day_counter()
        self._fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing
        self._spread: float = 0.0
        self._averaging_method: RateAveraging = RateAveraging.Compound
        self._engine: PricingEngine | None = None
        # PQuantLib addition — see the module docstring.
        self._evaluation_date: Date | None = None

    # --- chained setters ---------------------------------------------------

    def receive_fixed(self, flag: bool = True) -> MakeMultipleResetsSwap:
        self._type = SwapType.Receiver if flag else SwapType.Payer
        return self

    def with_type(self, swap_type: SwapType) -> MakeMultipleResetsSwap:
        self._type = swap_type
        return self

    def with_nominal(self, n: float) -> MakeMultipleResetsSwap:
        self._nominal = n
        return self

    def with_fixed_rate(self, fixed_rate: float) -> MakeMultipleResetsSwap:
        self._fixed_rate = fixed_rate
        return self

    def with_settlement_days(self, settlement_days: int) -> MakeMultipleResetsSwap:
        self._settlement_days = settlement_days
        return self

    def with_effective_date(self, d: Date) -> MakeMultipleResetsSwap:
        self._effective_date = None if d == _NULL_DATE else d
        return self

    def with_termination_date(self, d: Date) -> MakeMultipleResetsSwap:
        """Set the end date; a non-null one clears the tenor.

        # C++ parity: makemultipleresetsswap.cpp:130-135.
        """
        if d == _NULL_DATE:
            self._termination_date = None
        else:
            self._termination_date = d
            self._tenor = None
        return self

    def with_forward_start(self, fwd_start: Period) -> MakeMultipleResetsSwap:
        self._forward_start = fwd_start
        return self

    def with_fixed_leg_frequency(self, f: Frequency) -> MakeMultipleResetsSwap:
        self._fixed_frequency = f
        return self

    def with_fixed_leg_day_count(self, day_counter: DayCounter) -> MakeMultipleResetsSwap:
        self._fixed_day_count = day_counter
        return self

    def with_fixed_leg_convention(self, bdc: BusinessDayConvention) -> MakeMultipleResetsSwap:
        self._fixed_convention = bdc
        return self

    def with_floating_leg_spread(self, spread: float) -> MakeMultipleResetsSwap:
        self._spread = spread
        return self

    def with_averaging_method(self, m: RateAveraging) -> MakeMultipleResetsSwap:
        self._averaging_method = m
        return self

    def with_discounting_term_structure(
        self, discount_curve: YieldTermStructureProtocol
    ) -> MakeMultipleResetsSwap:
        self._engine = DiscountingSwapEngine(discount_curve, include_settlement_date_flows=False)
        return self

    def with_pricing_engine(self, engine: PricingEngine) -> MakeMultipleResetsSwap:
        self._engine = engine
        return self

    def with_evaluation_date(self, d: Date) -> MakeMultipleResetsSwap:
        """Pin the reference date used for the spot-date calculation.

        PQuantLib addition — see :meth:`~pquantlib.instruments.make_vanilla_swap.MakeVanillaSwap.with_evaluation_date`.
        """
        self._evaluation_date = d
        return self

    # --- build -------------------------------------------------------------

    def build(self) -> MultipleResetsSwap:
        """Construct the swap.

        # C++ parity: ``MakeMultipleResetsSwap::operator
        ext::shared_ptr<MultipleResetsSwap>()`` (makemultipleresetsswap.cpp:40-101).
        """
        cal = self._ibor_index.fixing_calendar()
        bdc = self._ibor_index.business_day_convention()

        qassert.require(
            self._effective_date is None or self._settlement_days is None,
            "withEffectiveDate and withSettlementDays are mutually exclusive",
        )

        if self._effective_date is not None:
            start_date = self._effective_date
        else:
            settlement_days = (
                self._settlement_days
                if self._settlement_days is not None
                else self._ibor_index.fixing_days()
            )
            ref_date = (
                self._evaluation_date
                if self._evaluation_date is not None
                else ObservableSettings().evaluation_date_or_today()
            )
            start_date = cal.advance(cal.adjust(ref_date), settlement_days, TimeUnit.Days)
            start_date = cal.advance_period(
                start_date,
                self._forward_start,
                BusinessDayConvention.Preceding
                if self._forward_start.length < 0
                else BusinessDayConvention.Following,
            )

        if self._termination_date is not None:
            end_date = self._termination_date
        else:
            assert self._tenor is not None
            end_date = cal.advance_period(start_date, self._tenor, bdc)

        reset_tenor = self._ibor_index.tenor()
        fixed_frequency = self._fixed_frequency
        if fixed_frequency == Frequency.NoFrequency:
            coupon_tenor = Period(
                self._resets_per_coupon * reset_tenor.length, reset_tenor.units
            )
            fixed_frequency = coupon_tenor.frequency()

        fixed_schedule = Schedule.from_rule(
            start_date,
            end_date,
            Period.from_frequency(fixed_frequency),
            cal,
            self._fixed_convention,
            self._fixed_convention,
            DateGeneration.Backward,
            False,
        )
        full_reset_schedule = Schedule.from_rule(
            start_date, end_date, reset_tenor, cal, bdc, bdc, DateGeneration.Backward, False
        )

        used_fixed_rate = self._fixed_rate
        if used_fixed_rate is None:
            temp = self._make_swap(fixed_schedule, full_reset_schedule, 0.0)
            if self._engine is None:
                discount = self._ibor_index.forecast_term_structure()
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

        swap = self._make_swap(fixed_schedule, full_reset_schedule, used_fixed_rate)
        if self._engine is None:
            discount = self._ibor_index.forecast_term_structure()
            if discount is not None:
                swap.set_pricing_engine(
                    DiscountingSwapEngine(discount, include_settlement_date_flows=False)
                )
        else:
            swap.set_pricing_engine(self._engine)
        return swap

    def _make_swap(
        self, fixed_schedule: Schedule, full_reset_schedule: Schedule, fixed_rate: float
    ) -> MultipleResetsSwap:
        return MultipleResetsSwap(
            self._type,
            self._nominal,
            fixed_schedule,
            fixed_rate,
            self._fixed_day_count,
            full_reset_schedule,
            self._ibor_index,
            self._resets_per_coupon,
            self._spread,
            self._averaging_method,
        )

    def __call__(self) -> MultipleResetsSwap:
        """Alias for :meth:`build`, mirroring C++ ``operator MultipleResetsSwap()``."""
        return self.build()


__all__ = ["MakeMultipleResetsSwap"]
