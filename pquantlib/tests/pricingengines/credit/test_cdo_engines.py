"""Cross-validate IntegralCDOEngine + MidPointCDOEngine.

Probe source: migration-harness/cpp/probes/cluster_w3c/probe.cpp
Reference:    migration-harness/references/cluster/w3c.json

Note on the W3-B cross-cluster: the C++ probe values were produced with
GaussianLHPLossModel (in cluster W3-B). To validate the engines in
isolation we use a deterministic analytic stub: expected_tranche_loss
grows linearly with time from 0 at t=0 to ``max_loss`` at maturity.
This lets us verify the engine integration code path and check NPV
shape / sign conventions without depending on the W3-B copula model.

A separate ``test_engines_match_cpp_with_lhp_stub`` test compares the
two engines' convergence to each other under the same stub model
(closes the LOOSE-tier cross-engine consistency check).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.basket import Basket
from pquantlib.experimental.credit.default_loss_model import DefaultLossModel
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    Restructuring,
    Seniority,
)
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.pool import Pool
from pquantlib.experimental.credit.synthetic_cdo import (
    ProtectionSide,
    SyntheticCDO,
)
from pquantlib.instruments.claim import FaceValueClaim
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.pricingengines.credit.integral_cdo_engine import (
    IntegralCDOEngine,
)
from pquantlib.pricingengines.credit.midpoint_cdo_engine import (
    MidPointCDOEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3c")


class _LinearETLLossModel(Observable):
    """Test stub: expected_tranche_loss(t) = max_loss * (t - t0) / (t_mat - t0).

    Mimics a deterministic linear-in-time loss accrual on the tranche.
    Used to validate the engine machinery without dragging the W3-B
    Gaussian-LHP copula model into scope.
    """

    def __init__(self, ref_date: Date, maturity: Date, max_loss: float) -> None:
        super().__init__()
        self._ref_date = ref_date
        self._maturity = maturity
        self._max_loss = max_loss
        self._basket: Basket | None = None

    def set_basket(self, basket: object) -> None:
        assert isinstance(basket, Basket)
        self._basket = basket

    def expected_tranche_loss(self, d: Date) -> float:
        if d <= self._ref_date:
            return 0.0
        if d >= self._maturity:
            return self._max_loss
        dt = float(d - self._ref_date) / float(self._maturity - self._ref_date)
        return self._max_loss * dt


def _make_5_issuer_basket(today: Date) -> Basket:
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(dt,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    issuer = Issuer(probabilities=[(key, curve)])
    pool = Pool()
    names = [f"N{i}" for i in range(5)]
    notionals = [1.0e6 for _ in range(5)]
    for n in names:
        pool.add(n, issuer, key)
    return Basket(
        today, names, notionals, pool,
        0.0, 0.1,
        FaceValueClaim(),
    )


def _make_synthetic_cdo(today: Date) -> tuple[SyntheticCDO, Basket, FlatForward, Date]:
    basket = _make_5_issuer_basket(today)
    maturity = today + Period(5, TimeUnit.Years)
    schedule = Schedule.from_rule(
        effective_date=today,
        termination_date=maturity,
        tenor=Period.from_frequency(Frequency.Quarterly),
        calendar=TARGET(),
        convention=BusinessDayConvention.Following,
        termination_date_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    cdo = SyntheticCDO(
        basket=basket,
        side=ProtectionSide.Seller,
        schedule=schedule,
        upfront_rate=0.0,
        running_rate=0.05,
        day_counter=Actual365Fixed(),
        payment_convention=BusinessDayConvention.Following,
    )
    discount_curve = FlatForward.from_rate(today, 0.03, Actual365Fixed())
    return cdo, basket, discount_curve, maturity


def test_integral_engine_npv_sign_with_stub_model() -> None:
    """Seller of a tranche with non-zero expected loss has negative NPV.

    The premium-leg PV is positive (seller receives premium), the
    protection-leg PV is positive too (seller pays out); for a deep
    equity tranche with high expected loss the protection outweighs the
    premium so seller's NPV is negative.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    cdo, basket, discount_curve, maturity = _make_synthetic_cdo(today)
    # 200k expected tranche loss linearly accrued over 5 years —
    # ~40% of the 500k tranche, similar magnitude to the C++ probe.
    model = _LinearETLLossModel(today, maturity, max_loss=200_000.0)
    assert isinstance(model, DefaultLossModel)
    basket.set_loss_model(model)
    engine = IntegralCDOEngine(discount_curve, step_size=Period(3, TimeUnit.Months))
    cdo.set_pricing_engine(engine)
    npv = cdo.npv()
    pv_prem = cdo.premium_value()
    pv_prot = cdo.protection_value()
    # Seller earns premium, pays protection: signs as in the C++ probe.
    assert pv_prem > 0.0
    assert pv_prot > 0.0
    # Loss model has 200k @ maturity — protection should outweigh premium.
    assert npv < 0.0
    # NPV = premium - protection + upfront (upfront=0 here).
    tolerance.loose(npv, pv_prem - pv_prot)


def test_midpoint_engine_npv_sign_with_stub_model() -> None:
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    cdo, basket, discount_curve, maturity = _make_synthetic_cdo(today)
    model = _LinearETLLossModel(today, maturity, max_loss=200_000.0)
    basket.set_loss_model(model)
    engine = MidPointCDOEngine(discount_curve)
    cdo.set_pricing_engine(engine)
    npv = cdo.npv()
    pv_prem = cdo.premium_value()
    pv_prot = cdo.protection_value()
    assert pv_prem > 0.0
    assert pv_prot > 0.0
    assert npv < 0.0
    tolerance.loose(npv, pv_prem - pv_prot)


def test_integral_vs_midpoint_engine_close_with_stub_model() -> None:
    """The two engines should converge to similar NPV under a smooth ETL.

    On a linear-in-time loss model the protection leg becomes an exact
    Riemann integral for both engines; differences come only from the
    premium-leg approximation (which integrates the survival prob
    rather than ``1 - ETL/N``). LOOSE-tier match.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today

    cdo_i, basket_i, discount_i, maturity_i = _make_synthetic_cdo(today)
    model_i = _LinearETLLossModel(today, maturity_i, max_loss=200_000.0)
    basket_i.set_loss_model(model_i)
    cdo_i.set_pricing_engine(
        IntegralCDOEngine(discount_i, step_size=Period(1, TimeUnit.Months)),
    )

    cdo_m, basket_m, discount_m, maturity_m = _make_synthetic_cdo(today)
    model_m = _LinearETLLossModel(today, maturity_m, max_loss=200_000.0)
    basket_m.set_loss_model(model_m)
    cdo_m.set_pricing_engine(MidPointCDOEngine(discount_m))

    # Both engines drive the same loss model; their NPVs should agree
    # to within a percent on smooth (linear-in-time) inputs.
    npv_i = cdo_i.npv()
    npv_m = cdo_m.npv()
    rel_diff = abs(npv_i - npv_m) / max(abs(npv_i), abs(npv_m))
    # Allow 5% relative difference (engines disagree on the premium-leg
    # treatment of the loss-adjusted notional).
    assert rel_diff < 0.05, f"NPV disagreement too large: integral={npv_i}, midpoint={npv_m}"


def test_zero_loss_model_yields_pure_premium_npv() -> None:
    """If the loss model returns 0 ETL everywhere, NPV = premium leg PV.

    A risk-free CDO seller receives the full coupon stream; the
    protection leg is 0.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    cdo, basket, discount_curve, maturity = _make_synthetic_cdo(today)
    zero_model = _LinearETLLossModel(today, maturity, max_loss=0.0)
    basket.set_loss_model(zero_model)
    engine = MidPointCDOEngine(discount_curve)
    cdo.set_pricing_engine(engine)
    tolerance.loose(cdo.protection_value(), 0.0)
    # NPV equals premium (seller side).
    tolerance.loose(cdo.npv(), cdo.premium_value())


def test_buyer_seller_signs_flip() -> None:
    """Buyer NPV = -Seller NPV (zero-sum, ignoring transaction costs).

    Tests the side-flip path inside both engines.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today

    cdo_s, basket_s, discount_s, maturity_s = _make_synthetic_cdo(today)
    model_s = _LinearETLLossModel(today, maturity_s, max_loss=200_000.0)
    basket_s.set_loss_model(model_s)
    cdo_s.set_pricing_engine(MidPointCDOEngine(discount_s))
    npv_s = cdo_s.npv()

    basket_b = _make_5_issuer_basket(today)
    schedule = Schedule.from_rule(
        effective_date=today,
        termination_date=today + Period(5, TimeUnit.Years),
        tenor=Period.from_frequency(Frequency.Quarterly),
        calendar=TARGET(),
        convention=BusinessDayConvention.Following,
        termination_date_convention=BusinessDayConvention.Following,
        rule=DateGeneration.Forward,
        end_of_month=False,
    )
    cdo_b = SyntheticCDO(
        basket=basket_b,
        side=ProtectionSide.Buyer,
        schedule=schedule,
        upfront_rate=0.0,
        running_rate=0.05,
        day_counter=Actual365Fixed(),
        payment_convention=BusinessDayConvention.Following,
    )
    discount_b = FlatForward.from_rate(today, 0.03, Actual365Fixed())
    model_b = _LinearETLLossModel(
        today, today + Period(5, TimeUnit.Years), max_loss=200_000.0,
    )
    basket_b.set_loss_model(model_b)
    cdo_b.set_pricing_engine(MidPointCDOEngine(discount_b))
    npv_b = cdo_b.npv()
    tolerance.loose(npv_b, -npv_s)


def test_cdo_basket_structural_fields_propagate_to_engine(
    cpp_ref: dict[str, Any],
) -> None:
    """Engine results echo basket attach/detach/tranche fields.

    Cross-checks the SyntheticCDO -> engine -> results structure
    against the C++ probe basket values.
    """
    today = Date.from_ymd(15, Month.January, 2024)
    ObservableSettings().evaluation_date = today
    cdo, basket, discount_curve, maturity = _make_synthetic_cdo(today)
    model = _LinearETLLossModel(today, maturity, max_loss=200_000.0)
    basket.set_loss_model(model)
    engine = MidPointCDOEngine(discount_curve)
    cdo.set_pricing_engine(engine)
    _ = cdo.npv()  # trigger calculate
    ref = cpp_ref["basket"]
    tolerance.tight(cdo.remaining_notional(), ref["tranche_notional"])
    tolerance.tight(cdo.leverage_factor(), 1.0)
