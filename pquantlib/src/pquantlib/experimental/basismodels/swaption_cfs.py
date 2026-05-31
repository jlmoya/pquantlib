"""SwaptionCashFlows — translate a swaption into deterministic fixed/float flows.

# C++ parity: ql/experimental/basismodels/swaptioncfs.hpp + .cpp (v1.42.1,
# 099987f0).

Decomposes a swaption's underlying swap into deterministic cash flows (with a
tenor-basis spread on the float leg) and the corresponding raw times / weights,
plus the swaption's future exercise times. Used to build the affine-TSR models
in TenorSwaptionVTS.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwap
from pquantlib.instruments.swaption import Swaption
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.date import Date


class IborLegCashFlows:
    """Deterministic-flow decomposition of an ibor (floating) leg.

    # C++ parity: ``IborLegCashFlows`` (swaptioncfs.hpp:34-50, .cpp:34-92).
    """

    def __init__(
        self,
        ibor_leg: list[CashFlow] | None = None,
        discount_curve: YieldTermStructureProtocol | None = None,
        cont_tenor_spread: bool = True,
    ) -> None:
        self._ref_date: Date | None = None
        self._float_leg: list[CashFlow] = []
        self._float_times: list[float] = []
        self._float_weights: list[float] = []

        if ibor_leg is None or discount_curve is None:
            # default constructor "which does nothing" (swaptioncfs.hpp:48).
            return

        self._ref_date = discount_curve.reference_date()
        ref_date = self._ref_date

        # find the first coupon for the initial payment.
        float_idx = 0
        while (float_idx + 1 < len(ibor_leg)) and (
            ref_date > _as_coupon(ibor_leg[float_idx]).accrual_start_date()
        ):
            float_idx += 1

        if ref_date <= _as_coupon(ibor_leg[float_idx]).accrual_start_date():
            first_float = _as_coupon(ibor_leg[float_idx])
            self._float_leg.append(
                SimpleCashFlow(first_float.nominal(), first_float.accrual_start_date())
            )
            # spread payments.
            for k in range(float_idx, len(ibor_leg)):
                coupon = _as_coupon_or_fail(ibor_leg[k])
                start_date = coupon.accrual_start_date()
                end_date = coupon.accrual_end_date()
                libor_forward_rate = coupon.rate()
                disc_forward_rate = (
                    discount_curve.discount(start_date) / discount_curve.discount(end_date) - 1.0
                ) / coupon.accrual_period()
                if cont_tenor_spread:
                    # Db = (1 + Delta L^libor) / (1 + Delta L^ois); spread paid at start.
                    spread = (
                        (1.0 + coupon.accrual_period() * libor_forward_rate)
                        / (1.0 + coupon.accrual_period() * disc_forward_rate)
                        - 1.0
                    ) / coupon.accrual_period()
                    pay_date = start_date
                else:
                    spread = libor_forward_rate - disc_forward_rate
                    pay_date = coupon.date()
                self._float_leg.append(
                    FixedRateCoupon.from_rate(
                        pay_date,
                        coupon.nominal(),
                        spread,
                        coupon.day_counter(),
                        start_date,
                        end_date,
                    )
                )
            # add the notional at the last date.
            last_float = _as_coupon(ibor_leg[-1])
            self._float_leg.append(
                SimpleCashFlow(-1.0 * last_float.nominal(), last_float.accrual_end_date())
            )

        dc = Actual365Fixed()
        for cf in self._float_leg:
            self._float_times.append(dc.year_fraction(ref_date, cf.date()))
        for cf in self._float_leg:
            self._float_weights.append(cf.amount())

    # --- inspectors ------------------------------------------------------------

    def float_leg(self) -> list[CashFlow]:
        return self._float_leg

    def float_times(self) -> list[float]:
        return self._float_times

    def float_weights(self) -> list[float]:
        return self._float_weights


class SwapCashFlows(IborLegCashFlows):
    """Deterministic-flow decomposition of a fixed-vs-floating swap.

    # C++ parity: ``SwapCashFlows`` (swaptioncfs.hpp:53-72, .cpp:94-115).
    """

    def __init__(
        self,
        swap: FixedVsFloatingSwap | None = None,
        discount_curve: YieldTermStructureProtocol | None = None,
        cont_tenor_spread: bool = True,
    ) -> None:
        self._fixed_leg: list[CashFlow] = []
        self._fixed_times: list[float] = []
        self._fixed_weights: list[float] = []
        self._annuity_weights: list[float] = []

        if swap is None or discount_curve is None:
            super().__init__()
            return

        super().__init__(swap.floating_leg(), discount_curve, cont_tenor_spread)
        assert self._ref_date is not None
        ref_date = self._ref_date

        # copy fixed-leg coupons starting on/after the reference date.
        for cf in swap.fixed_leg():
            if _as_coupon(cf).accrual_start_date() >= ref_date:
                self._fixed_leg.append(cf)

        dc = Actual365Fixed()
        for cf in self._fixed_leg:
            self._fixed_times.append(dc.year_fraction(ref_date, cf.date()))
        for cf in self._fixed_leg:
            self._fixed_weights.append(cf.amount())
        for cf in self._fixed_leg:
            coupon = _as_coupon_or_none(cf)
            if coupon is not None:
                self._annuity_weights.append(coupon.nominal() * coupon.accrual_period())

    # --- inspectors ------------------------------------------------------------

    def fixed_leg(self) -> list[CashFlow]:
        return self._fixed_leg

    def fixed_times(self) -> list[float]:
        return self._fixed_times

    def fixed_weights(self) -> list[float]:
        return self._fixed_weights

    def annuity_weights(self) -> list[float]:
        return self._annuity_weights


class SwaptionCashFlows(SwapCashFlows):
    """Deterministic-flow decomposition of a swaption (+ exercise times).

    # C++ parity: ``SwaptionCashFlows`` (swaptioncfs.hpp:75-89, .cpp:119-131).
    """

    def __init__(
        self,
        swaption: Swaption | None = None,
        discount_curve: YieldTermStructureProtocol | None = None,
        cont_tenor_spread: bool = True,
    ) -> None:
        self._swaption: Swaption | None = swaption
        self._exercise_times: list[float] = []

        if swaption is None or discount_curve is None:
            super().__init__()
            return

        super().__init__(swaption.underlying_swap(), discount_curve, cont_tenor_spread)
        assert self._ref_date is not None
        ref_date = self._ref_date

        dc = Actual365Fixed()
        for d in swaption.exercise().dates():
            if d > ref_date:  # only future exercise dates
                self._exercise_times.append(dc.year_fraction(ref_date, d))

    # --- inspectors ------------------------------------------------------------

    def swaption(self) -> Swaption | None:
        return self._swaption

    def exercise_times(self) -> list[float]:
        return self._exercise_times


def swaption_cashflows(
    swaption: Swaption,
    discount_curve: YieldTermStructureProtocol,
    cont_tenor_spread: bool = True,
) -> SwaptionCashFlows:
    """Free-function convenience wrapper around :class:`SwaptionCashFlows`."""
    return SwaptionCashFlows(swaption, discount_curve, cont_tenor_spread)


# --- helpers ------------------------------------------------------------------


def _as_coupon(cf: CashFlow) -> Coupon:
    assert isinstance(cf, Coupon)
    return cf


def _as_coupon_or_fail(cf: CashFlow) -> Coupon:
    qassert.require(isinstance(cf, Coupon), "FloatingLeg CashFlow is no Coupon.")
    assert isinstance(cf, Coupon)
    return cf


def _as_coupon_or_none(cf: CashFlow) -> Coupon | None:
    return cf if isinstance(cf, Coupon) else None


__all__ = [
    "IborLegCashFlows",
    "SwapCashFlows",
    "SwaptionCashFlows",
    "swaption_cashflows",
]
