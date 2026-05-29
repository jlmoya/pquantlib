"""NonstandardSwap — fixed-vs-floating swap with per-coupon nominal / rate / spread.

# C++ parity: ql/instruments/nonstandardswap.{hpp,cpp} @ v1.42.1.

A NonstandardSwap is a fixed-vs-floating IRS where each period can
have its own notional, fixed rate, gearing, and spread. The amortizing
and step-up amortizing swap structures are the principal use cases.

PQuantLib minimal scope (Phase 11 W1-B):

- Per-coupon vector nominals, fixed rates, gearings, spreads.
- IborIndex on the floating leg.
- Intermediate / final capital-exchange flows are **carved out**.

The class is consumed directly by
:class:`Gaussian1dNonstandardSwaptionEngine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.instruments.swap import Swap, SwapArguments, SwapResults, SwapType
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.time.schedule import Schedule


class NonstandardSwapArguments(SwapArguments):
    """Engine-arguments carrier for NonstandardSwap.

    # C++ parity: ``NonstandardSwap::arguments`` (nonstandardswap.hpp:126-151).
    """

    def __init__(self) -> None:
        super().__init__()
        self.type: SwapType = SwapType.Receiver
        self.fixed_nominal: list[float] = []
        self.floating_nominal: list[float] = []
        self.fixed_reset_dates: list[Date] = []
        self.fixed_pay_dates: list[Date] = []
        self.floating_accrual_times: list[float] = []
        self.floating_reset_dates: list[Date] = []
        self.floating_fixing_dates: list[Date] = []
        self.floating_pay_dates: list[Date] = []
        self.fixed_coupons: list[float] = []
        self.fixed_rate: list[float] = []
        self.floating_spreads: list[float] = []
        self.floating_gearings: list[float] = []
        self.floating_coupons: list[float] = []
        self.ibor_index: IborIndex | None = None
        self.fixed_is_redemption_flow: list[bool] = []
        self.floating_is_redemption_flow: list[bool] = []
        self.swap: NonstandardSwap | None = None
        self.settlement_type: int = 0
        self.settlement_method: int = 0

    def validate(self) -> None:
        super().validate()
        qassert.require(self.ibor_index is not None, "ibor_index is null")


class NonstandardSwapResults(SwapResults):
    """Engine-results carrier for NonstandardSwap."""


class NonstandardSwap(Swap):
    """Fixed-vs-floating swap with per-coupon parameters.

    # C++ parity: ``class NonstandardSwap`` (nonstandardswap.hpp:40-123).
    """

    def __init__(
        self,
        type_: SwapType,
        fixed_nominal: Sequence[float],
        floating_nominal: Sequence[float],
        fixed_schedule: Schedule,
        fixed_rate: Sequence[float],
        fixed_day_count: DayCounter,
        floating_schedule: Schedule,
        ibor_index: IborIndex,
        gearing: float | Sequence[float] = 1.0,
        spread: float | Sequence[float] = 0.0,
        floating_day_count: DayCounter | None = None,
        intermediate_capital_exchange: bool = False,
        final_capital_exchange: bool = False,
        payment_convention: BusinessDayConvention | None = None,
    ) -> None:
        super().__init__(n_legs=2)
        self._type: SwapType = type_
        self._fixed_schedule: Schedule = fixed_schedule
        self._floating_schedule: Schedule = floating_schedule
        self._ibor_index: IborIndex = ibor_index
        self._fixed_day_count: DayCounter = fixed_day_count
        self._floating_day_count: DayCounter = (
            floating_day_count if floating_day_count is not None
            else ibor_index.day_counter()
        )

        n_fix = len(fixed_schedule) - 1
        n_float = len(floating_schedule) - 1
        self._fixed_nominal: list[float] = self._expand(fixed_nominal, n_fix, 0.0)
        self._floating_nominal: list[float] = self._expand(floating_nominal, n_float, 0.0)
        self._fixed_rate: list[float] = self._expand(fixed_rate, n_fix, 0.0)
        self._gearing: list[float] = self._expand(gearing, n_float, 1.0)
        self._spread: list[float] = self._expand(spread, n_float, 0.0)

        qassert.require(
            not intermediate_capital_exchange,
            "intermediate_capital_exchange is carved out in Phase 11 W1-B",
        )
        qassert.require(
            not final_capital_exchange,
            "final_capital_exchange is carved out in Phase 11 W1-B",
        )
        self._intermediate_capital_exchange = intermediate_capital_exchange
        self._final_capital_exchange = final_capital_exchange

        self._payment_convention: BusinessDayConvention = (
            payment_convention if payment_convention is not None
            else fixed_schedule.business_day_convention
        )

        # Build fixed leg: one FixedRateCoupon per period.
        fixed_leg: list[CashFlow] = []
        for i in range(n_fix):
            start = fixed_schedule.date(i)
            end = fixed_schedule.date(i + 1)
            pay = fixed_schedule.calendar.adjust(end, self._payment_convention)
            fixed_leg.append(
                FixedRateCoupon.from_rate(
                    payment_date=pay,
                    nominal=self._fixed_nominal[i],
                    rate=self._fixed_rate[i],
                    day_counter=fixed_day_count,
                    accrual_start_date=start,
                    accrual_end_date=end,
                )
            )

        # Build floating leg: re-use the ibor_leg builder.
        floating_leg = ibor_leg(
            floating_schedule,
            ibor_index,
            nominals=self._floating_nominal,
            payment_day_counter=self._floating_day_count,
            payment_adjustment=self._payment_convention,
            gearings=self._gearing,
            spreads=self._spread,
        )

        self._legs[0] = fixed_leg
        self._legs[1] = floating_leg

        # # C++ parity: payer sign per leg per type.
        # Payer swap = paying fixed, receiving floating.
        if type_ == SwapType.Payer:
            self._payer = [-1.0, 1.0]
        else:
            self._payer = [1.0, -1.0]

        for leg in self._legs:
            for cf in leg:
                cf.register_with(self)

    @staticmethod
    def _expand(v: float | Sequence[float], n: int, default: float) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v)] * n
        seq = list(v)
        if not seq:
            return [default] * n
        qassert.require(
            len(seq) == n,
            f"vector size ({len(seq)}) does not match expected ({n})",
        )
        return [float(x) for x in seq]

    # --- inspectors ------------------------------------------------------

    def type(self) -> SwapType:
        return self._type

    def fixed_nominal(self) -> list[float]:
        return list(self._fixed_nominal)

    def floating_nominal(self) -> list[float]:
        return list(self._floating_nominal)

    def fixed_rate(self) -> list[float]:
        return list(self._fixed_rate)

    def ibor_index(self) -> IborIndex:
        return self._ibor_index

    def fixed_leg(self) -> list[CashFlow]:
        return list(self._legs[0])

    def floating_leg(self) -> list[CashFlow]:
        return list(self._legs[1])

    # --- Instrument interface -------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # # C++ parity: nonstandardswap.cpp:210-310 (paraphrased).
        super().setup_arguments(args)
        if not isinstance(args, NonstandardSwapArguments):
            return  # plain SwapEngine path
        args.type = self._type
        args.fixed_nominal = list(self._fixed_nominal)
        args.floating_nominal = list(self._floating_nominal)
        args.ibor_index = self._ibor_index
        args.swap = self

        n_fix = len(self._legs[0])
        n_float = len(self._legs[1])
        args.fixed_reset_dates = [Date()] * n_fix
        args.fixed_pay_dates = [Date()] * n_fix
        args.fixed_coupons = [0.0] * n_fix
        args.fixed_rate = list(self._fixed_rate)
        args.fixed_is_redemption_flow = [False] * n_fix
        args.floating_reset_dates = [Date()] * n_float
        args.floating_fixing_dates = [Date()] * n_float
        args.floating_pay_dates = [Date()] * n_float
        args.floating_accrual_times = [0.0] * n_float
        args.floating_spreads = list(self._spread)
        args.floating_gearings = list(self._gearing)
        args.floating_coupons = [float("nan")] * n_float
        args.floating_is_redemption_flow = [False] * n_float

        for i, cf in enumerate(self._legs[0]):
            assert isinstance(cf, FixedRateCoupon)
            args.fixed_reset_dates[i] = cf.accrual_start_date()
            args.fixed_pay_dates[i] = cf.date()
            try:
                args.fixed_coupons[i] = cf.amount()
            except Exception:
                args.fixed_coupons[i] = 0.0

        for i, cf2 in enumerate(self._legs[1]):
            if isinstance(cf2, FloatingRateCoupon):
                args.floating_accrual_times[i] = cf2.accrual_period()
                args.floating_pay_dates[i] = cf2.date()
                args.floating_reset_dates[i] = cf2.accrual_start_date()
                args.floating_fixing_dates[i] = cf2.fixing_date()
                try:
                    args.floating_coupons[i] = cf2.amount()
                except Exception:
                    args.floating_coupons[i] = float("nan")


__all__ = [
    "NonstandardSwap",
    "NonstandardSwapArguments",
    "NonstandardSwapResults",
]
