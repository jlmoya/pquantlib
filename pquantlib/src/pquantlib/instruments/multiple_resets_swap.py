"""MultipleResetsSwap — fixed leg vs a multiple-resets floating leg.

# C++ parity: ql/instruments/multipleresetsswap.{hpp,cpp} (v1.43).

The floating leg carries coupons whose rate compounds or averages
``resets_per_coupon`` consecutive Ibor fixings taken inside each accrual period
(see :mod:`pquantlib.cashflows.multiple_resets_coupon`). The coupon schedule is
derived from the reset schedule by taking every ``resets_per_coupon``-th date,
so the number of reset periods must be an exact multiple of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.multiple_resets_coupon import MultipleResetsCoupon, MultipleResetsLeg
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwap,
    FixedVsFloatingSwapArguments,
)
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.swap import SwapType
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.business_day_convention import BusinessDayConvention
    from pquantlib.time.calendar import Calendar


def _build_coupon_schedule(full_reset_schedule: Schedule, resets_per_coupon: int) -> Schedule:
    """Every ``resets_per_coupon``-th date of the reset schedule.

    # C++ parity: anonymous-namespace ``buildCouponSchedule``
    (multipleresetsswap.cpp:28-36). The result is a bare date-list schedule, so
    its business-day convention is ``Unadjusted`` — which is what
    ``FixedVsFloatingSwap`` falls back to when no payment convention is given.
    """
    n = (full_reset_schedule.size() - 1) // resets_per_coupon
    return Schedule([full_reset_schedule.date(i * resets_per_coupon) for i in range(n + 1)])


class MultipleResetsSwap(FixedVsFloatingSwap):
    """Swap with a fixed leg and a multiple-resets floating leg.

    # C++ parity: ``MultipleResetsSwap`` (multipleresetsswap.hpp:39-66,
    .cpp:38-77).
    """

    def __init__(
        self,
        swap_type: SwapType,
        nominal: float,
        fixed_schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        full_reset_schedule: Schedule,
        ibor_index: IborIndexProtocol,
        resets_per_coupon: int,
        spread: float = 0.0,
        averaging_method: RateAveraging = RateAveraging.Compound,
        payment_convention: BusinessDayConvention | None = None,
        payment_lag: int = 0,
        payment_calendar: Calendar | None = None,
    ) -> None:
        n_coupons = (full_reset_schedule.size() - 1) // resets_per_coupon
        super().__init__(
            swap_type=swap_type,
            fixed_nominals=[nominal] * (fixed_schedule.size() - 1),
            fixed_schedule=fixed_schedule,
            fixed_rate=fixed_rate,
            fixed_day_count=fixed_day_count,
            floating_nominals=[nominal] * n_coupons,
            floating_schedule=_build_coupon_schedule(full_reset_schedule, resets_per_coupon),
            ibor_index=ibor_index,
            spread=spread,
            floating_day_count=ibor_index.day_counter(),
            payment_convention=payment_convention,
            payment_lag=payment_lag,
            payment_calendar=payment_calendar,
        )
        self._full_reset_schedule: Schedule = full_reset_schedule
        self._resets_per_coupon: int = resets_per_coupon
        self._averaging_method: RateAveraging = averaging_method

        n_resets = full_reset_schedule.size() - 1
        qassert.require(
            n_resets % resets_per_coupon == 0,
            f"number of reset periods ({n_resets}) is not a multiple of "
            f"resetsPerCoupon ({resets_per_coupon})",
        )

        self._legs[1] = (
            MultipleResetsLeg(self._full_reset_schedule, self.ibor_index(), resets_per_coupon)
            .with_notionals(self.floating_nominals())
            .with_rate_spreads(spread)
            .with_averaging_method(averaging_method)
            .with_payment_adjustment(self.payment_convention())
            .with_payment_lag(payment_lag)
            .with_payment_calendar(
                self._full_reset_schedule.calendar
                if payment_calendar is None
                else payment_calendar
            )
            .build()
        )
        for cf in self._legs[1]:
            cf.register_with(self)

    # --- inspectors --------------------------------------------------------

    def full_reset_schedule(self) -> Schedule:
        return self._full_reset_schedule

    def resets_per_coupon(self) -> int:
        return self._resets_per_coupon

    def averaging_method(self) -> RateAveraging:
        return self._averaging_method

    # --- FixedVsFloatingSwap hook ------------------------------------------

    def _setup_floating_arguments(self, args: FixedVsFloatingSwapArguments) -> None:
        """Fill floating-leg fields in the engine argument carrier.

        # C++ parity: ``MultipleResetsSwap::setupFloatingArguments``
        (multipleresetsswap.cpp:79-102).
        """
        leg = self._legs[1]
        n = len(leg)
        args.floating_reset_dates = [None] * n
        args.floating_pay_dates = [None] * n
        args.floating_fixing_dates = [None] * n
        args.floating_nominals = [0.0] * n
        args.floating_accrual_times = [0.0] * n
        args.floating_spreads = [0.0] * n
        args.floating_coupons = [None] * n
        for i, cf in enumerate(leg):
            assert isinstance(cf, MultipleResetsCoupon)
            args.floating_reset_dates[i] = cf.accrual_start_date()
            args.floating_pay_dates[i] = cf.date()
            args.floating_fixing_dates[i] = cf.fixing_date()
            args.floating_nominals[i] = cf.nominal()
            args.floating_accrual_times[i] = cf.accrual_period()
            args.floating_spreads[i] = cf.spread()
            try:
                args.floating_coupons[i] = cf.amount()
            except Exception:
                args.floating_coupons[i] = None


__all__ = ["MultipleResetsSwap"]
