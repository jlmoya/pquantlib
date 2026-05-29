"""IntegralCdsEngine — Riemann-integral CDS engine.

# C++ parity: ql/pricingengines/credit/integralcdsengine.{hpp,cpp} (v1.42.1).

Refines the midpoint engine by partitioning each accrual period into
``integration_step``-sized sub-intervals and accumulating both the
accrual contribution and the protection-leg payoff as a Riemann sum
on the survival-probability differences ``dP``.

For a finer step the result approaches the continuous integral; the
midpoint engine is the special case ``integration_step = (end - start)``.
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwapArguments,
    CreditDefaultSwapResults,
    ProtectionSide,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date
from pquantlib.time.period import Period

_BASIS_POINT: float = 1.0e-4


class IntegralCdsEngine(
    GenericEngine[CreditDefaultSwapArguments, CreditDefaultSwapResults],
):
    """Riemann-integral CDS engine.

    Construct with an ``integration_step`` Period (e.g. 1 Month).
    """

    def __init__(
        self,
        integration_step: Period,
        probability: DefaultProbabilityTermStructure,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        include_settlement_date_flows: bool | None = None,
    ) -> None:
        super().__init__(CreditDefaultSwapArguments(), CreditDefaultSwapResults())
        self._integration_step: Period = integration_step
        self._probability: DefaultProbabilityTermStructure = probability
        self._recovery_rate: float = recovery_rate
        self._discount_curve: YieldTermStructure = discount_curve
        self._include_settlement_date_flows: bool | None = include_settlement_date_flows
        probability.register_with(self)
        discount_curve.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """C++ parity: integralcdsengine.cpp:43-195.

        # C++ parity divergence: the C++ engine guards against empty
        # ``Handle<>``s; Python takes non-optional curves at construction
        # and the runtime null check is dropped.

        Statement count is high due to the per-coupon sub-integration +
        side flipping + result population (matches C++ method length).
        """
        qassert.require(
            self._integration_step != Period(0),
            "null period set",
        )

        args = self._arguments
        results = self._results
        assert args.notional is not None
        assert args.spread is not None
        assert args.side is not None
        assert args.claim is not None
        assert args.upfront_payment is not None

        today = ObservableSettings().evaluation_date_or_today()
        settlement_date = self._discount_curve.reference_date()
        include_ref_date = self._include_settlement_date_flows

        # ---- upfront -----------------------------------------------------
        upf_pvo1 = 0.0
        results.upfront_npv = 0.0
        if not args.upfront_payment.has_occurred(settlement_date, include_ref_date):
            upf_pvo1 = self._discount_curve.discount(args.upfront_payment.date())
            results.upfront_npv = upf_pvo1 * args.upfront_payment.amount()

        # ---- accrual rebate ---------------------------------------------
        results.accrual_rebate_npv = 0.0
        if args.accrual_rebate is not None and not args.accrual_rebate.has_occurred(
            settlement_date, include_ref_date,
        ):
            results.accrual_rebate_npv = (
                self._discount_curve.discount(args.accrual_rebate.date())
                * args.accrual_rebate.amount()
            )

        # ---- premium + protection legs ----------------------------------
        results.coupon_leg_npv = 0.0
        results.default_leg_npv = 0.0
        for i, cf in enumerate(args.leg):
            if cf.has_occurred(settlement_date, include_ref_date):
                continue
            coupon = cast("FixedRateCoupon", cf)
            payment_date = coupon.date()
            start_date = (
                args.protection_start if i == 0 else coupon.accrual_start_date()
            )
            end_date = coupon.accrual_end_date()
            effective_start_date = (
                today if start_date <= today <= end_date else start_date
            )
            coupon_amount = coupon.amount()

            s_payment = self._probability.survival_probability(payment_date)

            # Survival-side coupon contribution.
            results.coupon_leg_npv += (
                s_payment * coupon_amount * self._discount_curve.discount(payment_date)
            )

            # Integral approximation for default-side terms.
            step = self._integration_step
            d0: Date = effective_start_date
            d1: Date = _min_date(d0 + step, end_date)
            p0 = self._probability.default_probability(d0)
            end_discount = self._discount_curve.discount(payment_date)

            while True:
                b = (
                    self._discount_curve.discount(d1)
                    if args.pays_at_default_time
                    else end_discount
                )
                p1 = self._probability.default_probability(d1)
                dp = p1 - p0

                if args.settles_accrual:
                    if args.pays_at_default_time:
                        results.coupon_leg_npv += coupon.accrued_amount(d1) * b * dp
                    else:
                        results.coupon_leg_npv += coupon_amount * b * dp
                claim_amt = args.claim.amount(d1, args.notional, self._recovery_rate)
                results.default_leg_npv += claim_amt * b * dp

                p0 = p1
                d0 = d1
                if d0 >= end_date:
                    break
                d1 = _min_date(d0 + step, end_date)

        upfront_sign = 1.0
        if args.side == ProtectionSide.Seller:
            results.default_leg_npv *= -1.0
            results.accrual_rebate_npv *= -1.0
        elif args.side == ProtectionSide.Buyer:
            results.coupon_leg_npv *= -1.0
            results.upfront_npv *= -1.0
            upfront_sign = -1.0
        else:
            qassert.fail(f"unknown protection side: {args.side}")

        results.value = (
            results.default_leg_npv + results.coupon_leg_npv
            + results.upfront_npv + results.accrual_rebate_npv
        )
        results.error_estimate = None

        if results.coupon_leg_npv != 0.0:
            results.fair_spread = -(
                results.default_leg_npv * args.spread
            ) / (results.coupon_leg_npv + results.accrual_rebate_npv)
        else:
            results.fair_spread = None

        if upf_pvo1 > 0.0:
            results.fair_upfront = -upfront_sign * (
                results.default_leg_npv + results.coupon_leg_npv
                + results.accrual_rebate_npv
            ) / (upf_pvo1 * args.notional)
        else:
            results.fair_upfront = None

        if args.spread != 0.0:
            results.coupon_leg_bps = results.coupon_leg_npv * _BASIS_POINT / args.spread
        else:
            results.coupon_leg_bps = None

        if args.upfront is not None and args.upfront != 0.0:
            results.upfront_bps = results.upfront_npv * _BASIS_POINT / args.upfront
        else:
            results.upfront_bps = None


def _min_date(a: Date, b: Date) -> Date:
    return a if a <= b else b


__all__ = ["IntegralCdsEngine"]
