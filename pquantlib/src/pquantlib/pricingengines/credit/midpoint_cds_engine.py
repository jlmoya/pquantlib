"""MidPointCdsEngine — midpoint-Riemann CDS engine.

# C++ parity: ql/pricingengines/credit/midpointcdsengine.{hpp,cpp} (v1.42.1).

Computes CDS NPV by approximating each accrual period's default time
with its midpoint and summing:

- Premium leg (paid if survival to coupon payment):
  ``sum_i  S(payment_i) * coupon_amount_i * D(payment_i)``
- Default accrual contribution (if settles_accrual):
  ``sum_i  (P(start_i, end_i) * coupon.accrued_amount(midpoint_i)
            * D(midpoint_i)  if pays_at_default else  D(payment_i))``
- Protection leg (claim paid at midpoint default time):
  ``sum_i  P(start_i, end_i) * claim_amount * D(midpoint_i)``

Side conventions:
- Buyer: pays premium leg + upfront; receives protection.
- Seller: receives premium leg + upfront; pays protection.

Result fair_spread is the running spread that zeros the contract NPV
(ignoring upfront).
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

_BASIS_POINT: float = 1.0e-4


class MidPointCdsEngine(
    GenericEngine[CreditDefaultSwapArguments, CreditDefaultSwapResults],
):
    """Midpoint-Riemann CDS engine."""

    def __init__(
        self,
        probability: DefaultProbabilityTermStructure,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        include_settlement_date_flows: bool | None = None,
    ) -> None:
        super().__init__(CreditDefaultSwapArguments(), CreditDefaultSwapResults())
        self._probability: DefaultProbabilityTermStructure = probability
        self._recovery_rate: float = recovery_rate
        self._discount_curve: YieldTermStructure = discount_curve
        self._include_settlement_date_flows: bool | None = include_settlement_date_flows
        probability.register_with(self)
        discount_curve.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Compute CDS results.

        # C++ parity: midpointcdsengine.cpp:42-185.

        # C++ parity divergence: the C++ engine guards against empty
        # ``Handle<>``s. The Python port takes non-optional protocol-typed
        # curves at construction so the runtime null check is dropped.

        Statement count is high due to the per-coupon NPV accumulation +
        side flipping + result population (matches C++ method length).
        """
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
            start_date = coupon.accrual_start_date()
            end_date = coupon.accrual_end_date()
            if i == 0:
                start_date = args.protection_start
            effective_start_date = (
                today if start_date <= today <= end_date else start_date
            )
            # Midpoint between effective_start_date and end_date.
            default_date = effective_start_date + (end_date - effective_start_date) // 2

            s_payment = self._probability.survival_probability(payment_date)
            p_period = self._probability.default_probability(
                effective_start_date, end_date,
            )

            # Premium leg: pays coupon at payment_date if survival.
            results.coupon_leg_npv += (
                s_payment * coupon.amount() * self._discount_curve.discount(payment_date)
            )
            # Accrual on default.
            if args.settles_accrual:
                if args.pays_at_default_time:
                    results.coupon_leg_npv += (
                        p_period
                        * coupon.accrued_amount(default_date)
                        * self._discount_curve.discount(default_date)
                    )
                else:
                    results.coupon_leg_npv += (
                        p_period
                        * coupon.amount()
                        * self._discount_curve.discount(payment_date)
                    )
            # Protection leg.
            claim_amt = args.claim.amount(
                default_date, args.notional, self._recovery_rate,
            )
            if args.pays_at_default_time:
                results.default_leg_npv += (
                    p_period * claim_amt * self._discount_curve.discount(default_date)
                )
            else:
                results.default_leg_npv += (
                    p_period * claim_amt * self._discount_curve.discount(payment_date)
                )

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


__all__ = ["MidPointCdsEngine"]
