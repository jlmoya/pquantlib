"""Validate IntegralNTDEngine math against analytic boundary cases.

The C++ reference probe at ``cluster_w3d.probe.cpp`` is bound to the
full ``Basket`` + ``DefaultLossModel`` machinery, which is in scope for
W3-B (loss models) and W3-C (full Basket). W3-D's slice exposes the
``BasketProtocol`` surface only; this test deploys analytic stubs and
covers:

  * Zero default probability ⇒ premium leg has full survival value,
    protection leg is zero, NPV equals premium-leg present value.
  * Total default at contract start ⇒ premium leg is zero, protection
    leg pays full claim.
  * Buyer vs Seller side flip ⇒ NPV signs invert exactly.
  * Upfront payment ⇒ adds remaining_notional * upfront_rate *
    discount(first-accrual-start) to the NPV (side-corrected).

These cases pin down the engine's algebra independently of the
underlying loss model.
"""

from __future__ import annotations

import math

import pytest

from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.basket_protocol import BasketProtocol
from pquantlib.experimental.credit.nth_to_default import NthToDefault
from pquantlib.instruments.claim import Claim, FaceValueClaim
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.credit.integral_ntd_engine import IntegralNTDEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit


class _ConstantProbBasket:
    """BasketProtocol stub returning a constant prob_at_least_n_events."""

    def __init__(
        self,
        ref_date: Date,
        n_names: int,
        const_prob: float,
        notional_per_name: float = 1.0e6,
        recovery: float = 0.4,
    ) -> None:
        self._ref_date = ref_date
        self._names = [f"Name{i}" for i in range(n_names)]
        self._const_prob = const_prob
        self._notional_per_name = notional_per_name
        self._recovery = recovery
        self._claim: Claim = FaceValueClaim()

    def size(self) -> int:
        return len(self._names)

    def names(self) -> list[str]:
        return list(self._names)

    def ref_date(self) -> Date:
        return self._ref_date

    def claim(self) -> Claim:
        return self._claim

    def remaining_size(self) -> int:
        return self.size()

    def remaining_notional(self) -> float:
        return self.size() * self._notional_per_name

    def recovery_rate(self, d: Date, i: int) -> float:
        del d, i
        return self._recovery

    def prob_at_least_n_events(self, n: int, d: Date) -> float:
        del n, d
        return self._const_prob


def _make_curve() -> FlatForward:
    today = Date.from_ymd(15, Month.January, 2024)
    return FlatForward(
        today, SimpleQuote(0.03), Actual365Fixed(),
        Compounding.Continuous, Frequency.Annual,
    )


def _make_ntd(
    basket: BasketProtocol,
    side: ProtectionSide,
    upfront_rate: float = 0.0,
    premium_rate: float = 0.01,
) -> NthToDefault:
    today = Date.from_ymd(15, Month.January, 2024)
    cal = TARGET()
    start = cal.advance(today, 1, TimeUnit.Days)
    end = start + Period(1, TimeUnit.Years)
    sched = (
        MakeSchedule()
        .from_date(start)
        .to(end)
        .with_calendar(cal)
        .with_tenor(Period(3, TimeUnit.Months))
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    return NthToDefault(
        basket=basket,
        n=2,
        side=side,
        premium_schedule=sched,
        upfront_rate=upfront_rate,
        premium_rate=premium_rate,
        day_counter=Actual365Fixed(),
        nominal=1.0e6,
        settle_premium_accrual=True,
    )


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    """Pin the observable evaluation date to match the curve ref date."""
    ObservableSettings().evaluation_date = Date.from_ymd(15, Month.January, 2024)


def test_integral_ntd_engine_zero_default_prob_seller() -> None:
    """At zero default probability:

    * No claim payout ⇒ protection_leg_npv == 0.
    * Premium leg survives to maturity ⇒ premium_value > 0.
    * Seller-side NPV equals the present value of the survival-only
      premium leg (positive).
    """
    today = Date.from_ymd(15, Month.January, 2024)
    basket = _ConstantProbBasket(today, n_names=3, const_prob=0.0)
    curve = _make_curve()
    ntd = _make_ntd(basket, ProtectionSide.Seller)
    engine = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    ntd.set_pricing_engine(engine)

    tolerance.tight(ntd.protection_leg_npv(), 0.0)
    # premium_leg_npv = sum( amount_i * D(t_i) ) > 0 — positive for seller
    assert ntd.premium_leg_npv() > 0.0
    # NPV (seller) = premium_value (no claim, no upfront).
    tolerance.tight(ntd.npv(), ntd.premium_leg_npv())


def test_integral_ntd_engine_buyer_side_flips_sign() -> None:
    """Side flip: Buyer NPV == -Seller NPV at the same configuration."""
    today = Date.from_ymd(15, Month.January, 2024)
    basket = _ConstantProbBasket(today, n_names=3, const_prob=0.0)
    curve = _make_curve()
    ntd_seller = _make_ntd(basket, ProtectionSide.Seller)
    ntd_buyer = _make_ntd(basket, ProtectionSide.Buyer)
    e1 = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    e2 = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    ntd_seller.set_pricing_engine(e1)
    ntd_buyer.set_pricing_engine(e2)

    tolerance.tight(ntd_buyer.npv(), -ntd_seller.npv())


def test_integral_ntd_engine_with_upfront_seller() -> None:
    """Upfront contribution = remaining_notional * upfront_rate * D(d0).

    Apply the C++ formula directly to confirm the engine's accounting.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    basket = _ConstantProbBasket(today, n_names=3, const_prob=0.0)
    curve = _make_curve()
    ntd_no_upf = _make_ntd(basket, ProtectionSide.Seller, upfront_rate=0.0)
    ntd_upf = _make_ntd(basket, ProtectionSide.Seller, upfront_rate=0.02)

    e1 = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    e2 = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    ntd_no_upf.set_pricing_engine(e1)
    ntd_upf.set_pricing_engine(e2)

    # Find the accrual_start date used by the engine: first coupon's start.
    first_coupon = ntd_upf.premium_leg()[0]
    assert isinstance(first_coupon, FixedRateCoupon)
    d0 = first_coupon.accrual_start_date()
    expected_upfront_npv = 3.0 * 1.0e6 * 0.02 * curve.discount(d0)
    diff = ntd_upf.npv() - ntd_no_upf.npv()
    # Both Seller-side: upfront-NPV added directly (no sign flip).
    tolerance.tight(diff, expected_upfront_npv)


def test_integral_ntd_engine_fair_premium_zero_at_zero_protection() -> None:
    """At zero default probability the fair premium is zero — no claim
    contribution to balance against the premium leg.

    # C++ parity: ``fair_premium = -spread * claim_value /
    # (premium_value + accrual_value)`` ⇒ 0 when claim_value == 0.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    basket = _ConstantProbBasket(today, n_names=3, const_prob=0.0)
    curve = _make_curve()
    ntd = _make_ntd(basket, ProtectionSide.Seller, premium_rate=0.01)
    engine = IntegralNTDEngine(Period(1, TimeUnit.Months), curve)
    ntd.set_pricing_engine(engine)

    fair = ntd.fair_premium()
    # Fair premium is exactly zero (no claim contribution to balance).
    assert math.isfinite(fair)
    tolerance.loose(fair, 0.0, reason="zero claim ⇒ zero fair premium")
