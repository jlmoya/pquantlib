"""OvernightIndexedSwap — fixed vs compounded overnight rate.

# C++ parity: ql/instruments/overnightindexedswap.{hpp,cpp} (v1.42.1).

OIS swap: fixed leg vs compounded overnight rate leg (Sofr, Eonia,
Sonia, etc.). Mirrors ``FixedVsFloatingSwap`` design but builds the
floating leg from ``OvernightLeg`` instead of ``IborLeg``.

The C++ class accepts a ``RateAveraging::Type`` (Compound or Simple) +
lookback / lockout / observation-shift modifiers. PQuantLib L3-C ports
the compound-default flavour only; the modifiers are propagated to the
underlying OvernightLeg but the L2-D ``overnight_leg`` builder is itself
the compound-default flavour (lookback/lockout/observation_shift are
deferred per L2-D carve-out). The ``averagingMethod`` / lookback /
lockout parameters are accepted at the constructor level for parity but
are not yet plumbed through.
"""

from __future__ import annotations

from typing import cast

from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwap,
    FixedVsFloatingSwapArguments,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.termstructures.protocols import IborIndexProtocol, OvernightIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.schedule import Schedule


class OvernightIndexedSwap(FixedVsFloatingSwap):
    """OIS — fixed-rate leg vs compounded overnight-rate leg.

    Two-schedule constructor mirrors the C++ four-argument-fixed-and-overnight
    variant; the single-schedule sugar (C++ ctor 1) is reproduced by
    passing the same schedule for both legs.
    """

    def __init__(
        self,
        swap_type: SwapType,
        nominal: float | list[float],
        schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        overnight_index: OvernightIndexProtocol,
        spread: float = 0.0,
        payment_lag: int = 0,
        payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
        payment_calendar: Calendar | None = None,
        telescopic_value_dates: bool = False,
        overnight_schedule: Schedule | None = None,
    ) -> None:
        # Normalize nominals: accept scalar or list.
        nominals_list = (
            [float(nominal)] if isinstance(nominal, int | float) else list(nominal)
        )

        # The C++ class accepts either a single schedule (sugar) or two
        # explicit schedules. Default the overnight schedule to the fixed.
        ov_sched = overnight_schedule if overnight_schedule is not None else schedule

        super().__init__(
            swap_type=swap_type,
            fixed_nominals=nominals_list,
            fixed_schedule=schedule,
            fixed_rate=fixed_rate,
            fixed_day_count=fixed_day_count,
            floating_nominals=nominals_list,
            floating_schedule=ov_sched,
            # The base class wants an IborIndexProtocol — overnight indexes
            # satisfy the same call surface (``fixing_calendar``, ``day_counter``,
            # ``business_day_convention``, ``end_of_month``, ``fixing``).
            ibor_index=cast(IborIndexProtocol, overnight_index),
            spread=spread,
            floating_day_count=overnight_index.day_counter(),
            payment_convention=payment_adjustment,
            payment_calendar=payment_calendar,
        )
        self._overnight_index: OvernightIndexProtocol = overnight_index

        # Build the overnight floating leg now.
        self._legs[1] = overnight_leg(
            ov_sched,
            overnight_index,
            nominals_list,
            payment_adjustment=payment_adjustment,
            payment_calendar=payment_calendar,
            spreads=spread,
        )
        for cf in self._legs[1]:
            cf.register_with(self)

    # --- FixedVsFloatingSwap hook ------------------------------------------

    def _setup_floating_arguments(self, args: FixedVsFloatingSwapArguments) -> None:
        """Fill floating-leg fields in the engine argument carrier.

        # C++ parity: ``OvernightIndexedSwap::setupFloatingArguments``
        # (overnightindexedswap.cpp:172-197).
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
            assert isinstance(cf, OvernightIndexedCoupon)
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

    # --- inspectors ---------------------------------------------------------

    def overnight_nominals(self) -> list[float]:
        return self.floating_nominals()

    def overnight_schedule(self) -> Schedule:
        return self.floating_schedule()

    def overnight_index(self) -> OvernightIndexProtocol:
        return self._overnight_index

    def overnight_leg(self):  # type: ignore[no-untyped-def]
        return self.floating_leg()

    def overnight_leg_bps(self) -> float:
        return self.floating_leg_bps()

    def overnight_leg_npv(self) -> float:
        return self.floating_leg_npv()


__all__ = ["OvernightIndexedSwap"]
