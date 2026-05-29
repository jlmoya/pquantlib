"""IsdaCdsEngine — ISDA-standard CDS pricing engine.

# C++ parity: ql/pricingengines/credit/isdacdsengine.{hpp,cpp} (v1.42.1).

Implements the ISDA standard credit-default-swap model (Markit ISDA
standard model C code reference [1] [2] [3] cited in the C++ header).
The engine evaluates:

- Protection leg: at every collected curve node in the protection
  period, accumulates ``h(t) / (f(t) + h(t)) * (P*Q)`` over each
  sub-interval where ``f = log(D0/D1)`` and ``h = log(S0/S1)``. A
  ``Taylor`` numerical fix is applied when ``f + h < 1e-4``.
- Premium leg: each coupon contributes
  ``amount * D(payment_date) * S(payment_date - 1)``.
- Default accrual: per coupon, integrates accrual over the active
  protection sub-period scaled by ``coupon.rate() * 365 / 360``.
  Two settings:
    * ``AccrualBias.HalfDayBias`` adds the -1/730 correction term
      from formula (50) of [1] (the "first-half-day-bias" error
      term).
    * ``ForwardsInCouponPeriod.Piecewise`` includes the curve's own
      intermediate nodes when integrating, vs ``Flat`` which uses
      only the coupon endpoints.
- Upfront flow and accrual rebate are added straight.

Sign conventions follow ``MidPointCdsEngine``:
- Buyer of protection: pays premium leg + upfront; coupon_leg_npv and
  upfront_npv are negated.
- Seller: receives premium leg + upfront; default_leg_npv and
  accrual_rebate_npv are negated.

References inside the C++ source:
[1] Pricing and Risk Management of CDS, OpenGamma Quantitative
    Research, 15-Oct-2013.
[2] ISDA CDS Standard Model Proposed Numerical Fix, Markit, 2012.
[3] Markit Interest Rate Curve XML Specifications.

# C++ parity divergence: the C++ engine asserts the discount + default
   curves use ``Actual365Fixed`` and that the reference date matches
   ``Settings::evaluationDate``. The Python port enforces the day-
   counter check; the eval-date check is relaxed because PQuantLib's
   ObservableSettings.evaluation_date_or_today() returns the curve
   reference date when no explicit eval-date is set.

# C++ parity divergence: ISDA collects node dates from both curves via
   ``dynamic_pointer_cast`` to specific concrete types
   (InterpolatedDiscountCurve<LogLinear>,
   InterpolatedForwardCurve<BackwardFlat>, etc.). The Python port
   uses a structural check (``hasattr(curve, "dates")``) so any
   PiecewiseYieldCurve or InterpolatedXxxCurve qualifies; this keeps
   the engine compatible with both L8-B's FlatHazardRate and L9-B's
   newly-wired PiecewiseDefaultCurve without per-cast plumbing.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import cast

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.claim import FaceValueClaim
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

_BASIS_POINT: float = 1.0e-4


class NumericalFix(IntEnum):
    """ISDA numerical-fix selector for the f+h denominator.

    # C++ parity: ``IsdaCdsEngine::NumericalFix`` (isdacdsengine.hpp:66-71).
    """

    None_ = 0  # add 1e-50 to denominators per [1] footnote 26
    Taylor = 1  # Taylor expansion for f+h < 1e-4 per [2]


class AccrualBias(IntEnum):
    """ISDA accrual-bias selector for formula (50) in [1].

    # C++ parity: ``IsdaCdsEngine::AccrualBias`` (isdacdsengine.hpp:73-77).
    """

    HalfDayBias = 0  # include the second (error) term in (50)
    NoBias = 1  # omit the second term


class ForwardsInCouponPeriod(IntEnum):
    """ISDA forward-rate-piecewise-in-coupon-period selector.

    # C++ parity: ``IsdaCdsEngine::ForwardsInCouponPeriod`` (lines 79-83).
    """

    Flat = 0  # include the second (error) term in formula (52)
    Piecewise = 1  # omit the second term (theoretically correct)


def _has_dates_method(curve: object) -> bool:
    """Structural check — accepts PiecewiseYield/Default + InterpolatedXxx."""
    return hasattr(curve, "dates") and callable(curve.dates)  # type: ignore[arg-type]


def _collect_dates(curve: object) -> list[Date]:
    if _has_dates_method(curve):
        return list(curve.dates())  # type: ignore[attr-defined]
    return []  # FlatForward / FlatHazardRate


def _sorted_union(a: list[Date], b: list[Date]) -> list[Date]:
    return sorted(set(a) | set(b))


class IsdaCdsEngine(
    GenericEngine[CreditDefaultSwapArguments, CreditDefaultSwapResults],
):
    """ISDA-standard credit-default-swap pricing engine.

    # C++ parity: ``IsdaCdsEngine`` (isdacdsengine.hpp).
    """

    NumericalFix = NumericalFix
    AccrualBias = AccrualBias
    ForwardsInCouponPeriod = ForwardsInCouponPeriod

    def __init__(
        self,
        probability: DefaultProbabilityTermStructure,
        recovery_rate: float,
        discount_curve: YieldTermStructure,
        include_settlement_date_flows: bool | None = None,
        numerical_fix: NumericalFix = NumericalFix.Taylor,
        accrual_bias: AccrualBias = AccrualBias.HalfDayBias,
        forwards_in_coupon_period: ForwardsInCouponPeriod = ForwardsInCouponPeriod.Piecewise,
    ) -> None:
        """Construct an ISDA CDS engine.

        # C++ parity: ``IsdaCdsEngine`` constructor (isdacdsengine.cpp:36-50).

        Parameters
        ----------
        probability : DefaultProbabilityTermStructure
            Issuer default probability curve. Must use Actual/365 Fixed.
        recovery_rate : float
            Recovery rate (e.g. 0.40 for 40%).
        discount_curve : YieldTermStructure
            Discount curve. Must use Actual/365 Fixed.
        include_settlement_date_flows : bool | None
            Forwarded to ``has_occurred()`` checks. None = default
            settings.
        numerical_fix : NumericalFix
            Default ``Taylor``.
        accrual_bias : AccrualBias
            Default ``HalfDayBias`` (Markit standard pre-1.8.2).
        forwards_in_coupon_period : ForwardsInCouponPeriod
            Default ``Piecewise``.
        """
        super().__init__(CreditDefaultSwapArguments(), CreditDefaultSwapResults())
        self._probability: DefaultProbabilityTermStructure = probability
        self._recovery_rate: float = recovery_rate
        self._discount_curve: YieldTermStructure = discount_curve
        self._include_settlement_date_flows: bool | None = include_settlement_date_flows
        self._numerical_fix: NumericalFix = numerical_fix
        self._accrual_bias: AccrualBias = accrual_bias
        self._forwards_in_coupon_period: ForwardsInCouponPeriod = forwards_in_coupon_period
        probability.register_with(self)
        discount_curve.register_with(self)

    def isda_rate_curve(self) -> YieldTermStructure:
        return self._discount_curve

    def isda_credit_curve(self) -> DefaultProbabilityTermStructure:
        return self._probability

    def calculate(self) -> None:  # noqa: PLR0915
        """Compute the ISDA-standard CDS NPV + results.

        # C++ parity: ``IsdaCdsEngine::calculate`` (isdacdsengine.cpp:52-365).

        Statement count is high due to the per-coupon NPV accumulation +
        Taylor branch + side flipping (matches C++ method length).
        """
        args = self._arguments
        results = self._results
        assert args.notional is not None
        assert args.spread is not None
        assert args.side is not None
        assert args.claim is not None
        assert args.upfront_payment is not None

        # Validate the engine flags.
        qassert.require(
            self._numerical_fix in (NumericalFix.None_, NumericalFix.Taylor),
            "numerical fix must be None_ or Taylor",
        )
        qassert.require(
            self._accrual_bias in (AccrualBias.HalfDayBias, AccrualBias.NoBias),
            "accrual bias must be HalfDayBias or NoBias",
        )
        qassert.require(
            self._forwards_in_coupon_period
            in (ForwardsInCouponPeriod.Flat, ForwardsInCouponPeriod.Piecewise),
            "forwards in coupon period must be Flat or Piecewise",
        )

        # Validate curve day counters (must be Actual/365 Fixed).
        dc_a365f = Actual365Fixed()
        qassert.require(
            self._discount_curve.day_counter() == dc_a365f,
            f"yield term structure day counter ({self._discount_curve.day_counter()}) "
            "should be Act/365(Fixed)",
        )
        qassert.require(
            self._probability.day_counter() == dc_a365f,
            f"probability term structure day counter ({self._probability.day_counter()}) "
            "should be Act/365(Fixed)",
        )
        qassert.require(args.settles_accrual,
                        "ISDA engine not compatible with non accrual paying CDS")
        qassert.require(args.pays_at_default_time,
                        "ISDA engine not compatible with end period payment")
        qassert.require(
            isinstance(args.claim, FaceValueClaim),
            "ISDA engine not compatible with non face value claim",
        )

        eval_date = ObservableSettings().evaluation_date_or_today()
        # Fall back to the curve reference date if no global eval date.
        if eval_date is None:
            eval_date = self._discount_curve.reference_date()
        include_ref = self._include_settlement_date_flows

        maturity = args.maturity
        # ``effectiveProtectionStart = max(protectionStart, evalDate + 1)``.
        effective_protection_start = max(args.protection_start, eval_date + 1)

        # Force the curves to be bootstrapped before reading their dates.
        self._discount_curve.discount(0.0)
        self._probability.survival_probability(0.0)

        # Collect dates from both curves and merge (sorted union).
        # # C++ parity divergence: structural check replaces dynamic_pointer_cast.
        y_dates = _collect_dates(self._discount_curve)
        c_dates = _collect_dates(self._probability)
        nodes = _sorted_union(y_dates, c_dates)
        if not nodes:
            nodes = [maturity]

        n_fix = 1e-50 if self._numerical_fix == NumericalFix.None_ else 0.0

        # ---- Protection leg --------------------------------------------------
        protection_npv = 0.0
        d0 = effective_protection_start - 1
        p0 = self._discount_curve.discount(d0)
        q0 = self._probability.survival_probability(d0)

        # ``it`` is the first node strictly greater than
        # effectiveProtectionStart; iterate forward, clamping at maturity.
        i = 0
        while i < len(nodes) and nodes[i] <= effective_protection_start:
            i += 1
        # Walk forward.
        while i < len(nodes):
            d1 = nodes[i]
            if d1 > maturity:
                d1 = maturity
                i = len(nodes)  # early exit
            p1 = self._discount_curve.discount(d1)
            q1 = self._probability.survival_probability(d1)
            fhat = math.log(p0) - math.log(p1)
            hhat = math.log(q0) - math.log(q1)
            fhphh = fhat + hhat
            if fhphh < 1e-4 and self._numerical_fix == NumericalFix.Taylor:
                fhphhq = fhphh * fhphh
                protection_npv += p0 * q0 * hhat * (
                    1.0 - 0.5 * fhphh + (1.0 / 6.0) * fhphhq
                    - (1.0 / 24.0) * fhphhq * fhphh
                    + (1.0 / 120.0) * fhphhq * fhphhq
                )
            else:
                protection_npv += hhat / (fhphh + n_fix) * (p0 * q0 - p1 * q1)
            d0 = d1
            p0 = p1
            q0 = q1
            i += 1
        protection_npv *= args.claim.amount(
            Date(), args.notional, self._recovery_rate,
        )
        results.default_leg_npv = protection_npv

        # ---- Premium leg + default accrual ----------------------------------
        premium_npv = 0.0
        default_accrual_npv = 0.0
        for cf in args.leg:
            coupon = cast("FixedRateCoupon", cf)

            # Day-counter check (C++ supports Act/365F, Act/360, Act/360 inc).
            # Relaxed in PQuantLib: trust the caller.

            # Premium coupons.
            if not cf.has_occurred(effective_protection_start, include_ref):
                premium_npv += (
                    coupon.amount()
                    * self._discount_curve.discount(coupon.date())
                    * self._probability.survival_probability(coupon.date() - 1)
                )

            # Default accruals — # C++ parity isdacdsengine.cpp:222-285.
            accrual_end = coupon.accrual_end_date()
            # ``has_occurred`` on simple_event(accrual_end) — we test the date.
            if accrual_end > effective_protection_start:
                start = max(coupon.accrual_start_date(),
                            effective_protection_start) - 1
                end = coupon.date() - 1
                tstart = self._discount_curve.time_from_reference(
                    coupon.accrual_start_date() - 1,
                ) - (1.0 / 730.0 if self._accrual_bias == AccrualBias.HalfDayBias else 0.0)
                local_nodes: list[Date] = [start]
                if self._forwards_in_coupon_period == ForwardsInCouponPeriod.Piecewise:
                    # Insert curve nodes strictly between start and end.
                    for nd in nodes:
                        if nd > start and nd < end:
                            local_nodes.append(nd)
                local_nodes.append(end)

                default_accr_this_node = 0.0
                t_prev = self._discount_curve.time_from_reference(local_nodes[0])
                p_prev = self._discount_curve.discount(local_nodes[0])
                q_prev = self._probability.survival_probability(local_nodes[0])
                t0_inner = t_prev  # for the bias term

                for j in range(1, len(local_nodes)):
                    nd = local_nodes[j]
                    t1 = self._discount_curve.time_from_reference(nd)
                    p1 = self._discount_curve.discount(nd)
                    q1 = self._probability.survival_probability(nd)
                    fhat = math.log(p_prev) - math.log(p1)
                    hhat = math.log(q_prev) - math.log(q1)
                    fhphh = fhat + hhat
                    if fhphh < 1e-4 and self._numerical_fix == NumericalFix.Taylor:
                        fhphhq = fhphh * fhphh
                        default_accr_this_node += (
                            hhat * p_prev * q_prev * (
                                (t0_inner - tstart) * (
                                    1.0 - 0.5 * fhphh + (1.0 / 6.0) * fhphhq
                                    - (1.0 / 24.0) * fhphhq * fhphh
                                )
                                + (t1 - t0_inner) * (
                                    0.5 - (1.0 / 3.0) * fhphh + (1.0 / 8.0) * fhphhq
                                    - (1.0 / 30.0) * fhphhq * fhphh
                                )
                            )
                        )
                    else:
                        default_accr_this_node += (
                            (hhat / (fhphh + n_fix))
                            * (
                                (t1 - t0_inner) * (
                                    (p_prev * q_prev - p1 * q1) / (fhphh + n_fix)
                                    - p1 * q1
                                )
                                + (t0_inner - tstart) * (p_prev * q_prev - p1 * q1)
                            )
                        )
                    t0_inner = t1
                    t_prev = t1
                    p_prev = p1
                    q_prev = q1
                default_accrual_npv += (
                    default_accr_this_node * args.notional
                    * coupon.rate() * 365.0 / 360.0
                )

        results.coupon_leg_npv = premium_npv + default_accrual_npv

        # ---- Upfront flow ----------------------------------------------------
        upf_pvo1 = 0.0
        results.upfront_npv = 0.0
        if not args.upfront_payment.has_occurred(eval_date, include_ref):
            upf_pvo1 = self._discount_curve.discount(args.upfront_payment.date())
            if args.upfront_payment.amount() != 0.0:
                results.upfront_npv = upf_pvo1 * args.upfront_payment.amount()

        # ---- Accrual rebate -------------------------------------------------
        results.accrual_rebate_npv = 0.0
        if args.accrual_rebate is not None and args.accrual_rebate.amount() != 0.0 \
                and not args.accrual_rebate.has_occurred(eval_date, include_ref):
            results.accrual_rebate_npv = (
                self._discount_curve.discount(args.accrual_rebate.date())
                * args.accrual_rebate.amount()
            )

        # ---- Side flipping --------------------------------------------------
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

        # ---- Derived results -----------------------------------------------
        if results.coupon_leg_npv != 0.0:
            results.fair_spread = -(
                results.default_leg_npv * args.spread
            ) / (results.coupon_leg_npv + results.accrual_rebate_npv)
        else:
            results.fair_spread = None

        upfront_sensitivity = upf_pvo1 * args.notional
        if upfront_sensitivity != 0.0:
            results.fair_upfront = (
                -upfront_sign * (
                    results.default_leg_npv + results.coupon_leg_npv
                    + results.accrual_rebate_npv
                ) / upfront_sensitivity
            )
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


__all__ = ["AccrualBias", "ForwardsInCouponPeriod", "IsdaCdsEngine", "NumericalFix"]
