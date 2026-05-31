"""IrregularSwap — fixed-vs-floating swap with a non-standard (e.g. step-down)
notional / schedule.

# C++ parity: ql/experimental/swaptions/irregularswap.hpp + .cpp (v1.42.1,
# 099987f0).

Unlike VanillaSwap, an IrregularSwap is built directly from a pre-assembled
fixed leg and floating leg (so per-coupon notionals can differ). It prices
fine through the ordinary :class:`DiscountingSwapEngine` (which only reads the
generic legs + payer multipliers); the irregular-specific argument fields are
populated only when an irregular engine is attached.

# C++ parity divergence (fairRate/fairSpread): the C++ ``fetchResults`` leaves
# fairRate_/fairSpread_ at a "Debug: to be done" 0.0 stub when the engine is a
# plain swap engine (the proper formula is commented out in irregularswap.cpp
# lines 186-195). PQuantLib mirrors that 0.0 fallback exactly.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.instruments.swap import (
    Leg,
    LegInput,
    Swap,
    SwapArguments,
    SwapResults,
    SwapType,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.date import Date


class IrregularSwapArguments(SwapArguments):
    """Engine argument carrier for IrregularSwap.

    # C++ parity: ``IrregularSwap::arguments`` (irregularswap.hpp:83-103).
    """

    def __init__(self) -> None:
        super().__init__()
        self.type: SwapType = SwapType.Receiver

        self.fixed_reset_dates: list[Date] = []
        self.fixed_pay_dates: list[Date] = []
        self.fixed_coupons: list[float] = []
        self.fixed_nominals: list[float] = []

        self.floating_reset_dates: list[Date] = []
        self.floating_fixing_dates: list[Date] = []
        self.floating_pay_dates: list[Date] = []
        self.floating_accrual_times: list[float] = []
        self.floating_nominals: list[float] = []
        self.floating_spreads: list[float] = []
        self.floating_coupons: list[float | None] = []

    def validate(self) -> None:
        # # C++ parity: irregularswap.cpp:198-223.
        super().validate()
        qassert.require(
            len(self.fixed_reset_dates) == len(self.fixed_pay_dates),
            "number of fixed start dates different from number of fixed payment dates",
        )
        qassert.require(
            len(self.fixed_pay_dates) == len(self.fixed_coupons),
            "number of fixed payment dates different from number of fixed coupon amounts",
        )
        qassert.require(
            len(self.floating_reset_dates) == len(self.floating_pay_dates),
            "number of floating start dates different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_fixing_dates) == len(self.floating_pay_dates),
            "number of floating fixing dates different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_accrual_times) == len(self.floating_pay_dates),
            "number of floating accrual Times different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_spreads) == len(self.floating_pay_dates),
            "number of floating spreads different from number of floating payment dates",
        )
        qassert.require(
            len(self.floating_pay_dates) == len(self.floating_coupons),
            "number of floating payment dates different from number of floating coupon amounts",
        )


class IrregularSwapResults(SwapResults):
    """Engine results carrier for IrregularSwap.

    # C++ parity: ``IrregularSwap::results`` (irregularswap.hpp:106-111).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_rate: float | None = None
        self.fair_spread: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_rate = None
        self.fair_spread = None


class IrregularSwapEngine(GenericEngine[IrregularSwapArguments, IrregularSwapResults]):
    """Base engine for irregular-swap pricing.

    # C++ parity: ``IrregularSwap::engine`` typedef.
    """

    def __init__(self) -> None:
        super().__init__(IrregularSwapArguments(), IrregularSwapResults())


class IrregularSwap(Swap):
    """Irregular fixed-vs-floating swap built from explicit legs."""

    def __init__(
        self, swap_type: SwapType, fixed_leg: LegInput, floating_leg: LegInput
    ) -> None:
        # # C++ parity: irregularswap.cpp:34-66 — Payer => fixed paid (-1),
        # # floating received (+1); Receiver => the reverse.
        super().__init__(2)
        self._type: SwapType = swap_type
        if swap_type == SwapType.Payer:
            self._payer = [-1.0, 1.0]
        elif swap_type == SwapType.Receiver:
            self._payer = [1.0, -1.0]
        else:  # pragma: no cover - exhaustive guard
            qassert.fail("Unknown Irregular-swap type")

        self._legs = [list(fixed_leg), list(floating_leg)]
        for cf in self._legs[0]:
            cf.register_with(self)
        for cf in self._legs[1]:
            cf.register_with(self)

        self._fair_rate: float | None = None
        self._fair_spread: float | None = None

    # --- inspectors ------------------------------------------------------------

    def type(self) -> SwapType:
        return self._type

    def fixed_leg(self) -> Leg:
        return self._legs[0]

    def floating_leg(self) -> Leg:
        return self._legs[1]

    def fixed_leg_bps(self) -> float:
        return self.leg_bps(0)

    def fixed_leg_npv(self) -> float:
        return self.leg_npv(0)

    def floating_leg_bps(self) -> float:
        return self.leg_bps(1)

    def floating_leg_npv(self) -> float:
        return self.leg_npv(1)

    def fair_rate(self) -> float:
        self.calculate()
        qassert.require(self._fair_rate is not None, "result not available")
        assert self._fair_rate is not None
        return self._fair_rate

    def fair_spread(self) -> float:
        self.calculate()
        qassert.require(self._fair_spread is not None, "result not available")
        assert self._fair_spread is not None
        return self._fair_spread

    # --- engine wiring ---------------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Fill the generic swap fields, then the irregular-specific arrays.

        # C++ parity: irregularswap.cpp:69-128.
        """
        super().setup_arguments(args)

        if not isinstance(args, IrregularSwapArguments):
            # it's a plain swap engine (e.g. DiscountingSwapEngine) — done.
            return

        args.type = self._type

        fixed = self.fixed_leg()
        args.fixed_reset_dates = [Date.min_date()] * len(fixed)
        args.fixed_pay_dates = [Date.min_date()] * len(fixed)
        args.fixed_coupons = [0.0] * len(fixed)
        args.fixed_nominals = [0.0] * len(fixed)
        for i, cf in enumerate(fixed):
            qassert.require(
                isinstance(cf, FixedRateCoupon),
                "dynamic cast of fixed leg coupon failed",
            )
            assert isinstance(cf, FixedRateCoupon)
            args.fixed_pay_dates[i] = cf.date()
            args.fixed_reset_dates[i] = cf.accrual_start_date()
            args.fixed_coupons[i] = cf.amount()
            args.fixed_nominals[i] = cf.nominal()

        floating = self.floating_leg()
        n = len(floating)
        args.floating_reset_dates = [Date.min_date()] * n
        args.floating_pay_dates = [Date.min_date()] * n
        args.floating_fixing_dates = [Date.min_date()] * n
        args.floating_accrual_times = [0.0] * n
        args.floating_spreads = [0.0] * n
        args.floating_nominals = [0.0] * n
        args.floating_coupons = [None] * n
        for i, cf in enumerate(floating):
            qassert.require(
                isinstance(cf, IborCoupon),
                "dynamic cast of floating leg coupon failed",
            )
            assert isinstance(cf, IborCoupon)
            args.floating_reset_dates[i] = cf.accrual_start_date()
            args.floating_pay_dates[i] = cf.date()
            args.floating_fixing_dates[i] = cf.fixing_date()
            args.floating_accrual_times[i] = cf.accrual_period()
            args.floating_spreads[i] = cf.spread()
            args.floating_nominals[i] = cf.nominal()
            try:
                args.floating_coupons[i] = cf.amount()
            except Exception:
                args.floating_coupons[i] = None

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull leg NPVs/BPS, then the (stubbed) fair-rate/spread.

        # C++ parity: irregularswap.cpp:174-196.
        """
        super().fetch_results(results)
        if isinstance(results, IrregularSwapResults):
            self._fair_rate = results.fair_rate
            self._fair_spread = results.fair_spread
        else:
            self._fair_rate = None
            self._fair_spread = None

        # C++ "Debug" fallback: when no engine-provided fairRate, derive 0.0
        # if a fixed-leg BPS is available (the proper formula is a TODO in C++).
        if self._fair_rate is None and self._leg_bps[0] is not None:
            self._fair_rate = 0.0
        if self._fair_spread is None and self._leg_bps[1] is not None:
            self._fair_spread = 0.0

    def setup_expired(self) -> None:
        """# C++ parity: irregularswap.cpp:167-172."""
        super().setup_expired()
        self._fair_rate = None
        self._fair_spread = None


__all__ = [
    "IrregularSwap",
    "IrregularSwapArguments",
    "IrregularSwapEngine",
    "IrregularSwapResults",
]
