"""VanillaSwap — fixed-rate vs Ibor swap.

# C++ parity: ql/instruments/vanillaswap.{hpp,cpp} (v1.42.1).

C++ ``VanillaSwap`` is a thin concrete subclass of
``FixedVsFloatingSwap``: it provides constructor delegation and builds
the floating leg via ``IborLeg``. The Python port follows the same
layout.

The C++ constructor takes a single ``nominal: Real`` and broadcasts
that into the two underlying nominal vectors via the
``FixedVsFloatingSwap`` parent. PQuantLib's parent ``__init__`` takes
``fixed_nominals`` + ``floating_nominals`` directly (more idiomatic in
Python); the single-nominal ergonomics is preserved via the
``VanillaSwap`` constructor.
"""

from __future__ import annotations

from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwap,
    FixedVsFloatingSwapArguments,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.schedule import Schedule


class VanillaSwap(FixedVsFloatingSwap):
    """Plain-vanilla swap: single nominal, fixed leg vs IBOR-linked floating leg.

    # C++ parity: ``class VanillaSwap : public FixedVsFloatingSwap``.
    """

    def __init__(
        self,
        swap_type: SwapType,
        nominal: float,
        fixed_schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        float_schedule: Schedule,
        ibor_index: IborIndexProtocol,
        spread: float,
        floating_day_count: DayCounter,
        payment_convention: BusinessDayConvention | None = None,
        use_indexed_coupons: bool | None = None,
    ) -> None:
        # # C++ parity: VanillaSwap::VanillaSwap (vanillaswap.cpp:30-53).
        super().__init__(
            swap_type=swap_type,
            fixed_nominals=[nominal],
            fixed_schedule=fixed_schedule,
            fixed_rate=fixed_rate,
            fixed_day_count=fixed_day_count,
            floating_nominals=[nominal],
            floating_schedule=float_schedule,
            ibor_index=ibor_index,
            spread=spread,
            floating_day_count=floating_day_count,
            payment_convention=payment_convention,
        )
        # Build the floating leg now that the parent has settled
        # payment_convention.
        self._legs[1] = ibor_leg(
            float_schedule,
            ibor_index,
            self.floating_nominals(),
            payment_day_counter=floating_day_count,
            payment_adjustment=self.payment_convention(),
            spreads=spread,
        )
        # Auto-attach a default IborCouponPricer to each floating coupon.
        # Python divergence: in C++ the pricer wiring happens via the
        # global PricerCleaner / setCouponPricer registry; in PQuantLib
        # we attach the default plain-IBOR pricer at construction time so
        # ``amount()`` works without an explicit setup call. Override via
        # ``set_coupon_pricer`` on ``floating_leg()`` if you want Black-vol
        # or any other pricer.
        set_coupon_pricer(self._legs[1], IborCouponPricer())
        for cf in self._legs[1]:
            cf.register_with(self)

    # --- FixedVsFloatingSwap hook ------------------------------------------

    def _setup_floating_arguments(self, args: FixedVsFloatingSwapArguments) -> None:
        """Fill floating-leg fields in the engine argument carrier.

        # C++ parity: ``VanillaSwap::setupFloatingArguments`` (vanillaswap.cpp:55-80).
        """
        leg = self._legs[1]
        n = len(leg)
        args.floating_reset_dates = [None] * n
        args.floating_pay_dates = [None] * n
        args.floating_fixing_dates = [None] * n
        args.floating_accrual_times = [0.0] * n
        args.floating_spreads = [0.0] * n
        args.floating_coupons = [None] * n
        args.floating_nominals = [0.0] * n
        for i, cf in enumerate(leg):
            assert isinstance(cf, IborCoupon)
            args.floating_reset_dates[i] = cf.accrual_start_date()
            args.floating_pay_dates[i] = cf.date()
            args.floating_nominals[i] = cf.nominal()
            args.floating_fixing_dates[i] = cf.fixing_date()
            args.floating_accrual_times[i] = cf.accrual_period()
            args.floating_spreads[i] = cf.spread()
            try:
                args.floating_coupons[i] = cf.amount()
            except Exception:
                args.floating_coupons[i] = None


__all__ = ["VanillaSwap"]
