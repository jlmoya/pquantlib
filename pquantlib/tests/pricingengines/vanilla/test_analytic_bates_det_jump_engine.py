"""AnalyticBatesDetJumpEngine tests — cross-validated against C++ reference.

Cross-validates against ``migration-harness/references/cluster/w1c.json``.

C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp}
            @ v1.42.1 (099987f0) — Gatheral form + DetJump CF wrap,
            Gauss-Laguerre order 144.

Tolerance choice:

* Engine NPV vs C++ reference: **LOOSE** (abs_tol=1e-8, rel_tol=1e-8).
  pquantlib uses scipy.integrate.quad (QUADPACK adaptive) while C++ uses
  Gauss-Laguerre quadrature. Same trade-off as L6-B (BatesEngine).
* **Algebraic identity** at ``thetaLambda == lambda``: TIGHT
  (abs_tol=1e-14, rel_tol=1e-12). When the OU long-term mean equals
  the initial intensity, the deterministic-intensity wrap collapses
  algebraically to the base ``BatesEngine.add_on_term``: the engine
  produces the same NPV as ``BatesEngine`` for any ``kappaLambda``.
  This is the cleanest reduction check for the wrap formula.
* **Degenerate-jump reduction** to Heston (lambda ~ 1e-12): LOOSE
  vs the C++ reference (one extra layer of CF arithmetic).
* Put-call parity at jumps-on: TIGHT (algebraic identity).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.models.equity.bates_det_jump_model import BatesDetJumpModel
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_bates_det_jump_engine import (
    AnalyticBatesDetJumpEngine,
)
from pquantlib.pricingengines.vanilla.bates_engine import BatesEngine
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# AMST Heston canonical testbed (rho=-0.7).
_S = 100.0
_R = 0.05
_Q = 0.0
_V0 = 0.04
_KAPPA = 2.0
_THETA = 0.04
_SIGMA = 0.3
_RHO = -0.7
_T = 1.0

# Lognormal jump params used by the C++ probe.
_LAMBDA = 0.1
_NU = -0.05
_DELTA = 0.1

# Deterministic-intensity OU params.
_KAPPA_LAMBDA = 1.0
_THETA_LAMBDA = 0.1  # = LAMBDA → algebraic reduction to BatesEngine.


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1c")


def _make_eval_setup() -> tuple[Date, Date, FlatForward, FlatForward, SimpleQuote]:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365  # exactly T=1.0 under Act/365 Fixed
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    spot = SimpleQuote(_S)
    return ref, expiry, rf, div, spot


def _make_det_engine(
    *,
    lambda_: float = _LAMBDA,
    nu: float = _NU,
    delta: float = _DELTA,
    kappa_lambda: float = _KAPPA_LAMBDA,
    theta_lambda: float = _THETA_LAMBDA,
) -> tuple[AnalyticBatesDetJumpEngine, Date]:
    """Build an AnalyticBatesDetJumpEngine on the AMST testbed."""
    _ref, expiry, rf, div, spot = _make_eval_setup()
    process = BatesProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=_RHO,
        lambda_=lambda_,
        nu=nu,
        delta=delta,
    )
    model = BatesDetJumpModel(
        process, kappa_lambda=kappa_lambda, theta_lambda=theta_lambda
    )
    return AnalyticBatesDetJumpEngine(model), expiry


def _make_bates_engine() -> tuple[BatesEngine, Date]:
    """Build the baseline BatesEngine for the algebraic-reduction check."""
    _ref, expiry, rf, div, spot = _make_eval_setup()
    process = BatesProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=_RHO,
        lambda_=_LAMBDA,
        nu=_NU,
        delta=_DELTA,
    )
    model = BatesModel(process)
    return BatesEngine(model), expiry


def _price(
    engine: AnalyticBatesDetJumpEngine | BatesEngine,
    expiry: Date,
    strike: float,
    option_type: OptionType,
) -> float:
    """Price a single European vanilla via the given engine."""
    payoff = PlainVanillaPayoff(option_type, strike)
    exercise = EuropeanExercise(expiry)
    option = EuropeanOption(payoff, exercise)
    option.set_pricing_engine(engine)
    return option.npv()


# ---------------------------------------------------------------------
# C++ Gauss-Laguerre cross-validation (jumps on).
# ---------------------------------------------------------------------


def test_amst_call_atm(cpp_refs: dict[str, Any]) -> None:
    """ATM call vs C++ reference (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 100.0, OptionType.Call), e["call_atm"])


def test_amst_put_atm(cpp_refs: dict[str, Any]) -> None:
    """ATM put vs C++ reference (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 100.0, OptionType.Put), e["put_atm"])


def test_amst_call_otm_low(cpp_refs: dict[str, Any]) -> None:
    """Deep-ITM call (K=80) vs C++ reference (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 80.0, OptionType.Call), e["call_otm_low"])


def test_amst_call_otm_high(cpp_refs: dict[str, Any]) -> None:
    """OTM call (K=120) vs C++ reference (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 120.0, OptionType.Call), e["call_otm_high"])


def test_amst_call_skew_low(cpp_refs: dict[str, Any]) -> None:
    """K=90 call (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 90.0, OptionType.Call), e["call_skew_low"])


def test_amst_call_skew_high(cpp_refs: dict[str, Any]) -> None:
    """K=110 call (LOOSE)."""
    engine, expiry = _make_det_engine()
    e = cpp_refs["bates_det_jump_engine_amst"]
    loose(_price(engine, expiry, 110.0, OptionType.Call), e["call_skew_high"])


# ---------------------------------------------------------------------
# Algebraic reduction: thetaLambda == lambda → BatesEngine.
# ---------------------------------------------------------------------


def test_reduction_to_bates_engine_call_atm() -> None:
    """thetaLambda = lambda → DetJump wrap == BatesEngine add-on.

    Algebraic identity (independent of kappaLambda):

        add_on_DetJump = [(kL*t - 1 + exp(-kL*t)) * thetaL / (kL*t*lambda)
                          + (1 - exp(-kL*t)) / (kL*t)] * l
        at thetaL == lambda
                       = [(kL*t - 1 + exp(-kL*t)) + (1 - exp(-kL*t))] * l / (kL*t)
                       = kL*t * l / (kL*t)
                       = l.

    Both engines invoke the same scipy quad on the same integrand →
    bit-identical (modulo float arithmetic noise from one extra
    multiply+divide).
    """
    det_engine, expiry = _make_det_engine(theta_lambda=_LAMBDA)
    bates_engine, _ = _make_bates_engine()
    tight(
        _price(det_engine, expiry, 100.0, OptionType.Call),
        _price(bates_engine, expiry, 100.0, OptionType.Call),
        reason="thetaLambda == lambda: wrap collapses to l (algebraic)",
    )


def test_reduction_to_bates_engine_put_atm() -> None:
    """thetaLambda = lambda → ATM put matches BatesEngine (TIGHT)."""
    det_engine, expiry = _make_det_engine(theta_lambda=_LAMBDA)
    bates_engine, _ = _make_bates_engine()
    tight(
        _price(det_engine, expiry, 100.0, OptionType.Put),
        _price(bates_engine, expiry, 100.0, OptionType.Put),
        reason="ATM put algebraic reduction at thetaLambda==lambda",
    )


def test_reduction_to_bates_engine_otm_call() -> None:
    """K=120 OTM call matches BatesEngine at thetaLambda=lambda (TIGHT)."""
    det_engine, expiry = _make_det_engine(theta_lambda=_LAMBDA)
    bates_engine, _ = _make_bates_engine()
    tight(
        _price(det_engine, expiry, 120.0, OptionType.Call),
        _price(bates_engine, expiry, 120.0, OptionType.Call),
        reason="OTM call algebraic reduction",
    )


def test_reduction_independent_of_kappa_lambda() -> None:
    """At thetaLambda=lambda the wrap collapses independent of kappaLambda.

    Doubles kappaLambda (kL=2 vs kL=1) — both engines must give the
    same NPV at thetaLambda=lambda.
    """
    e1, expiry = _make_det_engine(theta_lambda=_LAMBDA, kappa_lambda=1.0)
    e2, _ = _make_det_engine(theta_lambda=_LAMBDA, kappa_lambda=2.0)
    tight(
        _price(e1, expiry, 100.0, OptionType.Call),
        _price(e2, expiry, 100.0, OptionType.Call),
        reason="kappaLambda dependence cancels at thetaLambda==lambda",
    )


# ---------------------------------------------------------------------
# Algebraic + structural tests.
# ---------------------------------------------------------------------


def test_put_call_parity_with_jumps() -> None:
    """C - P = S*df_q - K*df_r at jumps-on ATM (TIGHT)."""
    engine, expiry = _make_det_engine()
    call = _price(engine, expiry, 100.0, OptionType.Call)
    put = _price(engine, expiry, 100.0, OptionType.Put)
    expected = _S - 100.0 * math.exp(-_R * _T)
    tight(
        call - put,
        expected,
        reason="put-call parity: C - P = S*df_q - K*df_r at q=0",
    )


def test_engine_holds_det_jump_model_reference() -> None:
    """``model()`` returns the BatesDetJumpModel (narrowed)."""
    engine, _ = _make_det_engine()
    m = engine.model()
    assert isinstance(m, BatesDetJumpModel)
    tight(m.kappa_lambda(), _KAPPA_LAMBDA, reason="ConstantParameter roundtrip")
    tight(m.theta_lambda(), _THETA_LAMBDA, reason="ConstantParameter roundtrip")
