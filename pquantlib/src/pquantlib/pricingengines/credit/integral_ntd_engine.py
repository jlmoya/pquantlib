"""IntegralNTDEngine — Riemann-integral N-th-to-default engine.

# C++ parity: ql/experimental/credit/integralntdengine.{hpp,cpp} (v1.42.1).

Prices an ``NthToDefault`` instrument by Riemann-integrating the
default-loss probability ``prob_at_least_n_events(n, t)`` of the
basket over each accrual period.

For each coupon period [start, end]:

* Premium leg: ``coupon_amount * discount(payment_date) *
  (1 - prob_at_least_n_events(n, payment_date))`` (survival-side).

* Accrual + protection legs: step ``d`` from ``start`` in
  ``integration_step`` increments to ``end``; let
  ``dcfdd = prob_at_least_n_events(n, d) - prob_at_least_n_events(n, d0)``.
  Then ``protection_value -= dcfdd * claim_amount * discount(d)``
  (claim flips sign at side correction) and, if accrual settles,
  ``accrual_value += coupon.accrued_amount(d) * discount(d) * dcfdd``.

* Upfront: ``basket.remaining_notional() * upfront_rate *
  discount(first_coupon.accrual_start_date())``.

* Side correction: ``Protection.Buyer`` flips premium / accrual /
  claim / upfront signs.

This implements the *homogeneous-basket* branch of the C++ engine
(``basketIsHomogeneous = true``) — equal recoveries, equal notionals
across names. The heterogeneous branch using
``probsBeingNthEvent`` is left as a deferred follow-up; the
``NthToDefault`` test fixture builds homogeneous baskets only.

# C++ parity divergence: the dual-branch (homogeneous /
# heterogeneous) code path in the C++ engine is collapsed to the
# homogeneous branch only — the heterogeneous branch needs
# ``Basket.probsBeingNthEvent`` (per-name triggering probabilities),
# which the W3-D ``BasketProtocol`` slice does not expose. The
# heterogeneous path is a deferred follow-up tracked in the cluster
# completion doc.
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.experimental.credit.nth_to_default import (
    NthToDefaultArguments,
    NthToDefaultResults,
)
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


def _min_date(a: Date, b: Date) -> Date:
    return a if a <= b else b


class IntegralNTDEngine(
    GenericEngine[NthToDefaultArguments, NthToDefaultResults],
):
    """Riemann-integral N-th-to-default engine.

    Construct with an ``integration_step`` Period (e.g. 1 Month) and a
    discount curve. The basket's loss-model is consumed via the
    ``prob_at_least_n_events`` protocol method.
    """

    def __init__(
        self,
        integration_step: Period,
        discount_curve: YieldTermStructure,
    ) -> None:
        super().__init__(NthToDefaultArguments(), NthToDefaultResults())
        self._integration_step: Period = integration_step
        self._discount_curve: YieldTermStructure = discount_curve
        discount_curve.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915
        """Run the Riemann integration.

        # C++ parity: integralntdengine.cpp:28-184.

        Statement count is high due to the per-coupon integration +
        side flipping + result population (matches C++ method length).
        """
        args = self._arguments
        results = self._results
        assert args.basket is not None
        assert args.notional is not None
        assert args.premium_rate is not None
        assert args.upfront_rate is not None
        assert args.ntd_order is not None
        assert args.side is not None

        today = ObservableSettings().evaluation_date_or_today()
        ref_date = self._discount_curve.reference_date()

        results.error_estimate = None
        results.value = 0.0
        results.premium_value = 0.0
        results.upfront_premium_value = 0.0
        accrual_value = 0.0
        claim_value = 0.0

        for coupon_cf in args.premium_leg:
            coupon = cast("FixedRateCoupon", coupon_cf)
            payment_date = coupon_cf.date()
            if payment_date <= ref_date:
                continue

            # Survival-side: probability of no trigger by payment date.
            prob_triggered_at_pay = args.basket.prob_at_least_n_events(
                args.ntd_order, payment_date
            )
            prob_non_triggered = 1.0 - prob_triggered_at_pay
            results.premium_value += (
                coupon_cf.amount()
                * self._discount_curve.discount(payment_date)
                * prob_non_triggered
            )

            # Integration window: max(coupon-accrual-start, curve-ref-date).
            d_start = (
                coupon.accrual_start_date()
                if coupon.accrual_start_date() >= ref_date
                else ref_date
            )
            d0 = d_start
            step = self._integration_step
            def_prob0 = args.basket.prob_at_least_n_events(args.ntd_order, d0)

            d = _min_date(d0 + step, coupon.accrual_end_date())
            adaptive_step = step
            while True:
                disc = self._discount_curve.discount(d)
                def_prob1 = args.basket.prob_at_least_n_events(args.ntd_order, d)
                dcfdd = def_prob1 - def_prob0

                # Claim amount uses recovery of name 0 (homogeneous-basket
                # branch in C++).
                claim_amt = args.basket.claim().amount(
                    d, args.notional, args.basket.recovery_rate(d, 0)
                )
                claim_value -= dcfdd * claim_amt * disc

                if args.settle_premium_accrual:
                    accrual_value += coupon.accrued_amount(d) * disc * dcfdd

                def_prob0 = def_prob1
                d0 = d
                if d0 >= coupon.accrual_end_date():
                    break

                # Step adaptation (matches C++ — once the proposed next
                # step exceeds accrual_end, fall back to 1 day until end).
                next_d = d0 + adaptive_step
                one_day = Period(1, TimeUnit.Days)
                if (
                    adaptive_step != one_day
                    and next_d > coupon.accrual_end_date()
                ):
                    adaptive_step = one_day
                    next_d = d0 + adaptive_step
                d = _min_date(next_d, coupon.accrual_end_date())

        # Upfront premium: paid up-front against the basket's remaining
        # notional, discounted to the first coupon's accrual-start date.
        if not args.premium_leg[0].has_occurred(today):
            first_coupon = cast("FixedRateCoupon", args.premium_leg[0])
            results.upfront_premium_value = (
                args.basket.remaining_notional()
                * args.upfront_rate
                * self._discount_curve.discount(first_coupon.accrual_start_date())
            )

        # Side flip — Buyer pays premium + upfront, receives protection.
        if args.side == ProtectionSide.Buyer:
            results.premium_value *= -1.0
            accrual_value *= -1.0
            claim_value *= -1.0
            results.upfront_premium_value *= -1.0

        results.value = (
            results.premium_value
            + accrual_value
            + claim_value
            + results.upfront_premium_value
        )

        # Fair premium = -spread * claim_value / (premium + accrual_value),
        # matches C++ integralntdengine.cpp:174-175.
        denom = results.premium_value + accrual_value
        if denom != 0.0:
            results.fair_premium = -args.premium_rate * claim_value / denom
        else:
            results.fair_premium = None
        results.protection_value = claim_value

        if results.fair_premium is not None:
            results.additional_results["fair_premium"] = results.fair_premium
        results.additional_results["premium_leg_npv"] = (
            results.premium_value + results.upfront_premium_value
        )
        results.additional_results["protection_leg_npv"] = results.protection_value

        # Sanity-check on missing fair_premium.
        if results.fair_premium is None:
            qassert.require(
                claim_value == 0.0,
                "fair premium undefined: zero premium + accrual NPV with non-zero claim",
            )
            results.fair_premium = 0.0


__all__ = ["IntegralNTDEngine"]
