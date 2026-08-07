"""IntegralNtdEngine — Riemann-integral N-th-to-default engine.

# C++ parity: ql/experimental/credit/integralntdengine.{hpp,cpp} (v1.43).

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

Only the homogeneous branch exists, and that is not a carve-out: C++
opens ``calculate()`` with ``bool basketIsHomogeneous = true;// hardcoded
by now`` (integralntdengine.cpp:46) and never assigns it again, so the
``probsBeingNthEvent`` branch guarded by ``else`` at line 105 is dead
code in v1.43. Porting it would add an untestable path that C++ cannot
be made to execute.
"""

from __future__ import annotations

import math
from typing import Final, cast

from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.experimental.credit.nth_to_default import (
    NthToDefaultArguments,
    NthToDefaultResults,
)
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

#: C++ compares the running step against the literal ``1*Days``
#: (integralntdengine.cpp:147). Hoisted so the comparison does not
#: allocate a Period per iteration.
_ONE_DAY: Final[Period] = Period(1, TimeUnit.Days)


class IntegralNtdEngine(
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
            # C++ integralntdengine.cpp:78-152, transcribed statement for
            # statement. Two things about this loop are easy to "improve"
            # and must not be:
            #
            #  * it is a do-while, so the FIRST evaluation is at d == d0,
            #    contributing dcfdd == 0. Skipping it is harmless; what is
            #    NOT harmless is the corollary that the step is taken from
            #    d0 AFTER the body, never before.
            #  * the grid is never clamped to accrual_end. The step shrinks
            #    to one day exactly once, the first time d0 + step would
            #    overshoot, and the walk then lands on accrual_end exactly.
            #    Clamping instead (``d = min(d0 + step, accrual_end)``)
            #    collapses the whole tail of a short accrual period into a
            #    single lump at accrual_end, with one discount factor and
            #    one accrued amount instead of the daily sequence.
            d = d_start
            d0 = d
            step = self._integration_step
            def_prob0 = args.basket.prob_at_least_n_events(args.ntd_order, d0)

            while True:
                disc = self._discount_curve.discount(d)
                def_prob1 = args.basket.prob_at_least_n_events(args.ntd_order, d)

                # Claim amount uses recovery of name 0: C++ hardcodes
                # ``bool basketIsHomogeneous = true`` (integralntdengine.cpp:46,
                # comment "hardcoded by now"), so the per-name
                # probsBeingNthEvent branch below it is unreachable in v1.43.
                claim_amt = args.basket.claim().amount(
                    d, args.notional, args.basket.recovery_rate(d, 0)
                )
                claim_value -= (def_prob1 - def_prob0) * claim_amt * disc

                dcfdd = def_prob1 - def_prob0
                def_prob0 = def_prob1

                if args.settle_premium_accrual:
                    accrual_value += coupon.accrued_amount(d) * disc * dcfdd

                d0 = d
                d = d0 + step
                if step != _ONE_DAY and d > coupon.accrual_end_date():
                    step = _ONE_DAY
                    d = d0 + step
                if d > coupon.accrual_end_date():
                    break

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
        # C++ integralntdengine.cpp:174-175. C++ divides unconditionally, so
        # a zero denominator gives an IEEE-754 infinity (or NaN for 0/0), not
        # an exception and not a fabricated 0.0. Reproduced explicitly because
        # Python raises ZeroDivisionError where C++ does not.
        denom = results.premium_value + accrual_value
        numer = -args.premium_rate * claim_value
        if denom == 0.0:
            results.fair_premium = (
                math.nan
                if numer == 0.0
                else math.copysign(1.0, numer) * math.copysign(1.0, denom) * math.inf
            )
        else:
            results.fair_premium = numer / denom
        results.protection_value = claim_value

        # Keys are the C++ additionalResults keys verbatim
        # (integralntdengine.cpp:179-183).
        results.additional_results["fairPremium"] = results.fair_premium
        results.additional_results["premiumLegNPV"] = (
            results.premium_value + results.upfront_premium_value
        )
        results.additional_results["protectionLegNPV"] = results.protection_value


__all__ = ["IntegralNtdEngine"]
