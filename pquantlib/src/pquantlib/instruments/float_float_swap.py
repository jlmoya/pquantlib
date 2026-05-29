"""FloatFloatSwap — two-leg swap exchanging two floating (Ibor) coupon streams.

# C++ parity: ql/instruments/floatfloatswap.{hpp,cpp} @ v1.42.1.

A FloatFloatSwap pays floating coupons on both legs, with potentially
distinct indexes, gearings, spreads, capped/floored caps, and notional
schedules. The C++ class is the workhorse instrument for cross-currency,
quanto, and dual-currency swap structures, and is consumed directly by
the :class:`Gaussian1dFloatFloatSwaptionEngine`.

PQuantLib minimal scope (Phase 11 W1-B):

- Two **Ibor** indexes (one per leg). CMS/swap-rate indexes and
  ``SwapSpreadIndex`` are **carved out** (the engine's ``index1/index2``
  dispatch chains assume IborIndex; CMS and CMS-spread require
  separate coupon pricers — not yet ported).
- Per-leg flat scalar nominal, gearing, spread (vectorized per-coupon
  variants accepted via the constructor for compatibility; capped/
  floored coupons are **carved out**).
- Intermediate / final capital-exchange flows are **carved out**.

These carve-outs collectively cover the W1-B test scenario (a plain
Ibor3M vs Ibor6M+spread swaption). The full multi-index variants are
straightforward extensions for a future cluster.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
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


# # C++ parity: ``Null<Real>()`` sentinel for capped/floored caps.
NULL_RATE: float = float("nan")


class FloatFloatSwapArguments(SwapArguments):
    """Engine-arguments carrier for FloatFloatSwap.

    # C++ parity: ``FloatFloatSwap::arguments`` (floatfloatswap.hpp:156-177).

    The carrier carries leg1 + leg2 per-coupon dates / accrual times /
    gearings / spreads / nominals / capped/floored rates plus the two
    indexes and the type. Engines (e.g.
    :class:`Gaussian1dFloatFloatSwaptionEngine`) read these directly
    without traversing the legs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type: SwapType = SwapType.Receiver
        self.nominal1: list[float] = []
        self.nominal2: list[float] = []
        self.leg1_reset_dates: list[Date] = []
        self.leg1_fixing_dates: list[Date] = []
        self.leg1_pay_dates: list[Date] = []
        self.leg2_reset_dates: list[Date] = []
        self.leg2_fixing_dates: list[Date] = []
        self.leg2_pay_dates: list[Date] = []
        self.leg1_spreads: list[float] = []
        self.leg2_spreads: list[float] = []
        self.leg1_gearings: list[float] = []
        self.leg2_gearings: list[float] = []
        self.leg1_capped_rates: list[float] = []
        self.leg1_floored_rates: list[float] = []
        self.leg2_capped_rates: list[float] = []
        self.leg2_floored_rates: list[float] = []
        self.leg1_coupons: list[float] = []
        self.leg2_coupons: list[float] = []
        self.leg1_accrual_times: list[float] = []
        self.leg2_accrual_times: list[float] = []
        self.index1: IborIndex | None = None
        self.index2: IborIndex | None = None
        self.leg1_is_redemption_flow: list[bool] = []
        self.leg2_is_redemption_flow: list[bool] = []
        # Back-reference to the swap so the engine's index1 lookups can
        # resolve the IborIndex (the engine reads
        # arguments.swap.ibor_index() in a few places).
        self.swap: FloatFloatSwap | None = None
        # Swaption-side settlement type / method are surfaced by
        # FloatFloatSwaption.setup_arguments via the C++
        # multi-inheritance equivalent. We carry them here to keep the
        # engine's ParYieldCurve check straightforward.
        self.settlement_type: int = 0  # SettlementType.Physical
        self.settlement_method: int = 0  # SettlementMethod.PhysicalOTC
        # ``exercise`` is left to the OptionArguments multi-inherited
        # mixin (FloatFloatSwaptionArguments) to declare with the right
        # typed-Exercise annotation.

    def validate(self) -> None:
        # # C++ parity: floatfloatswap.cpp:573-621.
        super().validate()
        for v in (
            self.leg1_reset_dates,
            self.leg1_fixing_dates,
            self.leg1_pay_dates,
            self.leg1_spreads,
            self.leg1_gearings,
            self.leg1_capped_rates,
            self.leg1_floored_rates,
            self.leg1_coupons,
            self.leg1_accrual_times,
            self.leg1_is_redemption_flow,
        ):
            qassert.require(
                len(v) == len(self.nominal1),
                "leg1 vector size mismatch with nominal1",
            )
        for v2 in (
            self.leg2_reset_dates,
            self.leg2_fixing_dates,
            self.leg2_pay_dates,
            self.leg2_spreads,
            self.leg2_gearings,
            self.leg2_capped_rates,
            self.leg2_floored_rates,
            self.leg2_coupons,
            self.leg2_accrual_times,
            self.leg2_is_redemption_flow,
        ):
            qassert.require(
                len(v2) == len(self.nominal2),
                "leg2 vector size mismatch with nominal2",
            )
        qassert.require(self.index1 is not None, "index1 is null")
        qassert.require(self.index2 is not None, "index2 is null")


class FloatFloatSwapResults(SwapResults):
    """Engine-results carrier for FloatFloatSwap.

    # C++ parity: ``FloatFloatSwap::results`` (floatfloatswap.hpp:180-185).

    Adds ``fair_spread1`` / ``fair_spread2`` on top of the standard
    swap results. These are populated by engines that solve for the
    spread that makes each leg's NPV equal to zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_spread1: float | None = None
        self.fair_spread2: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_spread1 = None
        self.fair_spread2 = None


class FloatFloatSwap(Swap):
    """Two-leg swap exchanging Ibor-coupon streams with potentially distinct indexes.

    # C++ parity: ``class FloatFloatSwap`` (floatfloatswap.hpp:43-153).

    The constructor accepts either scalar (per-leg) or vector
    (per-coupon) gearings / spreads / capped / floored rates; vector
    entries take precedence when supplied.

    Carve-outs from the C++ surface (Phase 11 W1-B):

    - Only IborIndex on both legs. CMS / SwapSpreadIndex are deferred.
    - ``intermediate_capital_exchange`` and ``final_capital_exchange``
      are accepted-but-asserted-false (no redemption flows).
    - ``capped_rate*`` / ``floored_rate*`` are accepted as ``NaN``-only
      (no Capped/Floored coupon construction).
    """

    def __init__(
        self,
        type_: SwapType,
        nominal1: float | Sequence[float],
        nominal2: float | Sequence[float],
        schedule1: Schedule,
        index1: IborIndex,
        day_count1: DayCounter,
        schedule2: Schedule,
        index2: IborIndex,
        day_count2: DayCounter,
        intermediate_capital_exchange: bool = False,
        final_capital_exchange: bool = False,
        gearing1: float | Sequence[float] = 1.0,
        spread1: float | Sequence[float] = 0.0,
        capped_rate1: float | Sequence[float] = NULL_RATE,
        floored_rate1: float | Sequence[float] = NULL_RATE,
        gearing2: float | Sequence[float] = 1.0,
        spread2: float | Sequence[float] = 0.0,
        capped_rate2: float | Sequence[float] = NULL_RATE,
        floored_rate2: float | Sequence[float] = NULL_RATE,
        payment_convention1: BusinessDayConvention | None = None,
        payment_convention2: BusinessDayConvention | None = None,
    ) -> None:
        # # C++ parity: floatfloatswap.cpp:37-109.
        super().__init__(n_legs=2)

        self._type: SwapType = type_
        self._schedule1: Schedule = schedule1
        self._schedule2: Schedule = schedule2
        self._index1: IborIndex = index1
        self._index2: IborIndex = index2
        self._day_count1: DayCounter = day_count1
        self._day_count2: DayCounter = day_count2
        self._intermediate_capital_exchange: bool = intermediate_capital_exchange
        self._final_capital_exchange: bool = final_capital_exchange

        n1 = len(schedule1) - 1
        n2 = len(schedule2) - 1

        self._nominal1: list[float] = self._expand(nominal1, n1, 0.0)
        self._nominal2: list[float] = self._expand(nominal2, n2, 0.0)
        self._gearing1: list[float] = self._expand(gearing1, n1, 1.0)
        self._gearing2: list[float] = self._expand(gearing2, n2, 1.0)
        self._spread1: list[float] = self._expand(spread1, n1, 0.0)
        self._spread2: list[float] = self._expand(spread2, n2, 0.0)
        self._capped_rate1: list[float] = self._expand(capped_rate1, n1, NULL_RATE)
        self._capped_rate2: list[float] = self._expand(capped_rate2, n2, NULL_RATE)
        self._floored_rate1: list[float] = self._expand(floored_rate1, n1, NULL_RATE)
        self._floored_rate2: list[float] = self._expand(floored_rate2, n2, NULL_RATE)

        # Validate carve-outs.
        qassert.require(
            not intermediate_capital_exchange,
            "intermediate_capital_exchange is carved out in Phase 11 W1-B",
        )
        qassert.require(
            not final_capital_exchange,
            "final_capital_exchange is carved out in Phase 11 W1-B",
        )
        for v in self._capped_rate1 + self._capped_rate2:
            qassert.require(
                _is_null_rate(v),
                "capped_rate* must be NULL_RATE for all entries (W1-B carve-out)",
            )
        for v_floor in self._floored_rate1 + self._floored_rate2:
            qassert.require(
                _is_null_rate(v_floor),
                "floored_rate* must be NULL_RATE for all entries (W1-B carve-out)",
            )

        self._payment_convention1: BusinessDayConvention = (
            payment_convention1 if payment_convention1 is not None
            else schedule1.business_day_convention
        )
        self._payment_convention2: BusinessDayConvention = (
            payment_convention2 if payment_convention2 is not None
            else schedule2.business_day_convention
        )

        # Build the legs via ibor_leg.
        self._legs[0] = ibor_leg(
            schedule1,
            index1,
            nominals=self._nominal1,
            payment_day_counter=day_count1,
            payment_adjustment=self._payment_convention1,
            gearings=self._gearing1,
            spreads=self._spread1,
        )
        self._legs[1] = ibor_leg(
            schedule2,
            index2,
            nominals=self._nominal2,
            payment_day_counter=day_count2,
            payment_adjustment=self._payment_convention2,
            gearings=self._gearing2,
            spreads=self._spread2,
        )

        # # C++ parity: payer sign per leg per type
        # (floatfloatswap.cpp:377-388). Payer: leg1=-1 (pays), leg2=+1
        # (receives). Receiver flips.
        if type_ == SwapType.Payer:
            self._payer = [-1.0, 1.0]
        else:
            self._payer = [1.0, -1.0]

        # Register observers.
        for leg in self._legs:
            for cf in leg:
                cf.register_with(self)

    @staticmethod
    def _expand(v: float | Sequence[float], n: int, default: float) -> list[float]:
        """Normalize scalar or sequence → length-``n`` list."""
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

    def nominal1(self) -> list[float]:
        return list(self._nominal1)

    def nominal2(self) -> list[float]:
        return list(self._nominal2)

    def index1(self) -> IborIndex:
        return self._index1

    def index2(self) -> IborIndex:
        return self._index2

    def ibor_index(self) -> IborIndex:
        """For the Gaussian1d engine — return leg2 index by convention.

        # C++ parity: not a single method, but the C++ engine reads
        # ``arguments_.swap->iborIndex()`` for legacy reasons; our shim
        # surfaces ``index2`` which is the lookup the engine performs.
        """
        return self._index2

    def leg1(self) -> list[CashFlow]:
        return list(self._legs[0])

    def leg2(self) -> list[CashFlow]:
        return list(self._legs[1])

    # --- Instrument interface -------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate engine arguments carrier with per-coupon dates + amounts.

        # C++ parity: floatfloatswap.cpp:391-525.
        """
        super().setup_arguments(args)
        if not isinstance(args, FloatFloatSwapArguments):
            return  # plain SwapEngine path
        args.type = self._type
        args.nominal1 = list(self._nominal1)
        args.nominal2 = list(self._nominal2)
        args.index1 = self._index1
        args.index2 = self._index2
        args.swap = self

        n1 = len(self._legs[0])
        n2 = len(self._legs[1])
        args.leg1_reset_dates = [Date()] * n1
        args.leg1_fixing_dates = [Date()] * n1
        args.leg1_pay_dates = [Date()] * n1
        args.leg2_reset_dates = [Date()] * n2
        args.leg2_fixing_dates = [Date()] * n2
        args.leg2_pay_dates = [Date()] * n2
        args.leg1_accrual_times = [0.0] * n1
        args.leg2_accrual_times = [0.0] * n2
        args.leg1_spreads = list(self._spread1)
        args.leg2_spreads = list(self._spread2)
        args.leg1_gearings = list(self._gearing1)
        args.leg2_gearings = list(self._gearing2)
        args.leg1_capped_rates = list(self._capped_rate1)
        args.leg1_floored_rates = list(self._floored_rate1)
        args.leg2_capped_rates = list(self._capped_rate2)
        args.leg2_floored_rates = list(self._floored_rate2)
        args.leg1_coupons = [float("nan")] * n1
        args.leg2_coupons = [float("nan")] * n2
        args.leg1_is_redemption_flow = [False] * n1
        args.leg2_is_redemption_flow = [False] * n2

        for i, cf in enumerate(self._legs[0]):
            if isinstance(cf, FloatingRateCoupon):
                args.leg1_accrual_times[i] = cf.accrual_period()
                args.leg1_pay_dates[i] = cf.date()
                args.leg1_reset_dates[i] = cf.accrual_start_date()
                args.leg1_fixing_dates[i] = cf.fixing_date()
                try:
                    args.leg1_coupons[i] = cf.amount()
                except Exception:
                    args.leg1_coupons[i] = float("nan")

        for i, cf2 in enumerate(self._legs[1]):
            if isinstance(cf2, FloatingRateCoupon):
                args.leg2_accrual_times[i] = cf2.accrual_period()
                args.leg2_pay_dates[i] = cf2.date()
                args.leg2_reset_dates[i] = cf2.accrual_start_date()
                args.leg2_fixing_dates[i] = cf2.fixing_date()
                try:
                    args.leg2_coupons[i] = cf2.amount()
                except Exception:
                    args.leg2_coupons[i] = float("nan")


def _is_null_rate(x: float) -> bool:
    """``NaN`` check used to detect NULL_RATE sentinels."""
    import math  # noqa: PLC0415
    return math.isnan(x)


__all__ = [
    "NULL_RATE",
    "FloatFloatSwap",
    "FloatFloatSwapArguments",
    "FloatFloatSwapResults",
]
