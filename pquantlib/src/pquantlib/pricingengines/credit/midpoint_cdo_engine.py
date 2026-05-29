"""MidPointCDOEngine — midpoint-Riemann CDO engine.

# C++ parity: ql/experimental/credit/midpointcdoengine.{hpp,cpp} (v1.42.1).

For each coupon, approximates the default time by the midpoint of the
accrual period and computes:

- Premium leg: ``(1 - E_2 / tranche_notional) * coupon_amount * D(payment)``
  where ``E_2`` is the expected tranche loss at the accrual end and
  ``D`` is the discount factor.
- Protection leg: ``(E_2 - E_1) * D(midpoint)`` for the loss
  accumulated within this period.
- Upfront leg: ``tranche_notional * upfront_rate * D(first_accrual_start)``.

Side convention mirrors the integral engine.
"""

from __future__ import annotations

from typing import cast

from pquantlib.cashflows.coupon import Coupon
from pquantlib.experimental.credit.synthetic_cdo import (
    ProtectionSide,
    SyntheticCDOArguments,
    SyntheticCDOResults,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class MidPointCDOEngine(
    GenericEngine[SyntheticCDOArguments, SyntheticCDOResults],
):
    """Midpoint-Riemann CDO engine.

    Parameters
    ----------
    discount_curve
        Yield-term structure for cash-flow discounting.
    """

    def __init__(
        self,
        discount_curve: YieldTermStructure,
    ) -> None:
        super().__init__(SyntheticCDOArguments(), SyntheticCDOResults())
        self._discount_curve: YieldTermStructure = discount_curve
        discount_curve.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Compute Synthetic CDO NPV / premium / protection / upfront.

        # C++ parity: midpointcdoengine.cpp:30-128.

        Statement count high due to per-coupon premium + protection +
        upfront accumulation.
        """
        args = self._arguments
        results = self._results
        assert args.basket is not None
        assert args.side is not None
        assert args.upfront_rate is not None
        assert args.running_rate is not None
        assert args.day_counter is not None

        today = ObservableSettings().evaluation_date_or_today()
        basket = args.basket

        results.premium_value = 0.0
        results.protection_value = 0.0
        results.upfront_premium_value = 0.0
        results.error = 0
        results.expected_tranche_loss = []
        results.x_min = basket.attachment_amount()
        results.x_max = basket.detachment_amount()
        results.remaining_notional = results.x_max - results.x_min
        inception_tranche_notional = basket.tranche_notional()

        # E_1: expected tranche loss at the start of the first relevant period.
        e1: float = 0.0
        first_cf = args.normalized_leg[0]
        if not first_cf.has_occurred(today):
            first_coupon = cast("Coupon", first_cf)
            e1 = basket.expected_tranche_loss(first_coupon.accrual_start_date())
        results.expected_tranche_loss.append(e1)

        for cf in args.normalized_leg:
            if cf.has_occurred(today):
                results.expected_tranche_loss.append(0.0)
                continue

            coupon = cast("Coupon", cf)
            payment_date = coupon.date()
            start_date = max(
                coupon.accrual_start_date(),
                self._discount_curve.reference_date(),
            )
            end_date = coupon.accrual_end_date()
            # The midpoint between effective_start_date and end_date.
            default_date = start_date + (end_date - start_date) // 2

            e2 = basket.expected_tranche_loss(end_date)
            results.expected_tranche_loss.append(e2)

            results.premium_value += (
                (inception_tranche_notional - e2)
                / inception_tranche_notional
                * coupon.amount()
                * self._discount_curve.discount(payment_date)
            )

            discount = self._discount_curve.discount(default_date)
            results.protection_value += discount * (e2 - e1)
            e1 = e2

        # Upfront — same as IntegralCDOEngine.
        if not args.normalized_leg[0].has_occurred(today):
            first_coupon = cast("Coupon", args.normalized_leg[0])
            results.upfront_premium_value = (
                inception_tranche_notional
                * args.upfront_rate
                * self._discount_curve.discount(first_coupon.accrual_start_date())
            )

        if args.side == ProtectionSide.Buyer:
            results.protection_value *= -1.0
            results.premium_value *= -1.0
            results.upfront_premium_value *= -1.0

        results.value = (
            results.premium_value
            - results.protection_value
            + results.upfront_premium_value
        )
        results.error_estimate = None

        fair_spread = 0.0
        if results.premium_value != 0.0:
            fair_spread = -(
                (results.protection_value + results.upfront_premium_value)
                * args.running_rate
                / results.premium_value
            )
        results.additional_results["fair_premium"] = fair_spread
        results.additional_results["premium_leg_npv"] = (
            results.premium_value + results.upfront_premium_value
        )
        results.additional_results["protection_leg_npv"] = results.protection_value


__all__ = ["MidPointCDOEngine"]
