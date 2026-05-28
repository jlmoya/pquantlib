"""BatesEngine tests — cross-validated against C++ Gauss-Laguerre reference.

Cross-validates against ``migration-harness/references/cluster/l6b.json``.

C++ parity: ql/pricingengines/vanilla/batesengine.{hpp,cpp}
            @ v1.42.1 (099987f0) — Gatheral form + jump add-on, Gauss-
            Laguerre order 144.

Tolerance choice:

* Engine NPV vs C++ reference (jumps on): **LOOSE** (abs_tol=1e-8,
  rel_tol=1e-8). pquantlib uses scipy.integrate.quad (QUADPACK
  adaptive) while C++ uses Gauss-Laguerre quadrature. Both converge
  to ~1e-8 absolute accuracy on the standard Bates regime but on
  different node grids — diverging at the 7th-8th significant
  figure. Same trade-off as L4-C's AnalyticHestonEngine.
* **Zero-jump reduction** to AnalyticHestonEngine: TIGHT
  (abs_tol=1e-14, rel_tol=1e-12). Algebraic reduction — at
  ``lambda=0`` the C++ add-on is identically zero, so the Bates and
  Heston engines invoke the same Fj integrand via the same scipy
  quad call. The Python BatesEngine's add-on term at ``lambda=0``
  evaluates to a complex ``0 + 0i`` exactly (modulo float
  multiplication noise) and the resulting NPV is exactly the
  AnalyticHestonEngine NPV. We compare against
  AnalyticHestonEngine() rather than the C++ value because at
  lambda=0 both engines should be **bit-identical** in pquantlib
  (same engine code path).
* **C++ probe reduction** check: LOOSE — even at lambda=1e-12 the C++
  uses Gauss-Laguerre, so we get the standard Heston-LOOSE
  divergence.
* Put-call parity at jumps-on: TIGHT (algebraic identity).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.models.equity.bates_model import BatesModel
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)
from pquantlib.pricingengines.vanilla.bates_engine import BatesEngine
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Albrecher-Mayer-Schoutens-Tistaert testbed (rho=-0.7).
_S = 100.0
_R = 0.05
_Q = 0.0
_V0 = 0.04
_KAPPA = 2.0
_THETA = 0.04
_SIGMA = 0.3
_RHO = -0.7
_T = 1.0

# Bates 1996 paper testbed (rho=-0.5).
_RHO_B96 = -0.5

# Jump parameters used by the C++ probe (AMST + jumps).
_LAMBDA = 0.1
_NU = -0.05
_DELTA = 0.1

# Near-zero jump parameters (PositiveConstraint rejects 0).
_LAMBDA_ZERO = 1e-12
_NU_ZERO = 0.0
_DELTA_ZERO = 1e-12


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/l6b")


def _make_eval_setup() -> tuple[Date, Date, FlatForward, FlatForward, SimpleQuote]:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365  # exactly T=1.0 under Act/365 Fixed
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    spot = SimpleQuote(_S)
    return ref, expiry, rf, div, spot


def _make_bates_engine(
    *,
    rho: float = _RHO,
    lambda_: float = _LAMBDA,
    nu: float = _NU,
    delta: float = _DELTA,
) -> tuple[BatesEngine, Date]:
    """Build a BatesEngine with the AMST testbed Heston params + given jumps."""
    _ref, expiry, rf, div, spot = _make_eval_setup()
    process = BatesProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=rho,
        lambda_=lambda_,
        nu=nu,
        delta=delta,
    )
    model = BatesModel(process)
    return BatesEngine(model), expiry


def _make_heston_engine() -> tuple[AnalyticHestonEngine, Date]:
    """Build the canonical AMST AnalyticHestonEngine (for reduction check)."""
    _ref, expiry, rf, div, spot = _make_eval_setup()
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=spot,
        v0=_V0,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=_RHO,
    )
    model = HestonModel(process)
    return AnalyticHestonEngine(model), expiry


def _price(
    engine: AnalyticHestonEngine, expiry: Date, strike: float, option_type: OptionType
) -> float:
    """Price a single European vanilla via the given engine."""
    payoff = PlainVanillaPayoff(option_type, strike)
    exercise = EuropeanExercise(expiry)
    option = EuropeanOption(payoff, exercise)
    option.set_pricing_engine(engine)
    return option.npv()


# ---------------------------------------------------------------------
# Zero-jump reduction — must match AnalyticHestonEngine exactly.
# ---------------------------------------------------------------------


def test_zero_jump_reduces_to_heston_atm_call() -> None:
    """BatesEngine at lambda=delta=0 equals AnalyticHestonEngine.

    At lambda=0 the add_on_term is identically 0+0j; the engine
    invokes the same Fj integrand via the same scipy quad call as
    AnalyticHestonEngine. NPVs should agree to TIGHT (the only
    arithmetic difference is the redundant ``0.0 + ...`` complex add
    inside the integrand, which is bit-identical under IEEE 754).

    Tighter than LOOSE because we are comparing two Python engines
    that share the same integrand code path, not Python vs C++.
    """
    # BatesModel enforces PositiveConstraint on the lambda/delta
    # ConstantParameter slots — exact 0.0 is rejected at model
    # construction time. We use 1e-30 as a tractable proxy: at this
    # magnitude the add_on_term evaluates to 1e-30 * (small complex)
    # which is far below the LOOSE comparison floor.
    bates_engine, expiry = _make_bates_engine(
        lambda_=1e-30, nu=0.0, delta=1e-30
    )
    heston_engine, _heston_expiry = _make_heston_engine()
    bates_price = _price(bates_engine, expiry, 100.0, OptionType.Call)
    heston_price = _price(heston_engine, expiry, 100.0, OptionType.Call)
    tight(
        bates_price,
        heston_price,
        reason="lambda~0 add_on ~ 0; same integrand → same NPV (Py vs Py)",
    )


def test_zero_jump_reduces_to_heston_atm_put() -> None:
    """ATM put under near-zero jumps matches Heston engine (TIGHT)."""
    bates_engine, expiry = _make_bates_engine(
        lambda_=1e-30, nu=0.0, delta=1e-30
    )
    heston_engine, _ = _make_heston_engine()
    tight(
        _price(bates_engine, expiry, 100.0, OptionType.Put),
        _price(heston_engine, expiry, 100.0, OptionType.Put),
        reason="ATM put zero-jump reduction",
    )


def test_zero_jump_reduces_to_heston_otm_call() -> None:
    """OTM (K=120) call reduces to Heston (TIGHT)."""
    bates_engine, expiry = _make_bates_engine(
        lambda_=1e-30, nu=0.0, delta=1e-30
    )
    heston_engine, _ = _make_heston_engine()
    tight(
        _price(bates_engine, expiry, 120.0, OptionType.Call),
        _price(heston_engine, expiry, 120.0, OptionType.Call),
        reason="OTM call zero-jump reduction",
    )


# ---------------------------------------------------------------------
# C++ Gauss-Laguerre cross-validation (jumps on).
# ---------------------------------------------------------------------


def test_amst_call_atm_with_jumps(cpp_refs: dict[str, Any]) -> None:
    """ATM call vs C++ Gauss-Laguerre Bates reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 100.0, OptionType.Call), e["call_atm"])


def test_amst_put_atm_with_jumps(cpp_refs: dict[str, Any]) -> None:
    """ATM put vs C++ Bates reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 100.0, OptionType.Put), e["put_atm"])


def test_amst_call_otm_low_with_jumps(cpp_refs: dict[str, Any]) -> None:
    """Deep-ITM call (K=80) vs C++ reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 80.0, OptionType.Call), e["call_otm_low"])


def test_amst_call_otm_high_with_jumps(cpp_refs: dict[str, Any]) -> None:
    """OTM call (K=120) vs C++ reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 120.0, OptionType.Call), e["call_otm_high"])


def test_amst_call_skew_low(cpp_refs: dict[str, Any]) -> None:
    """K=90 call captures left-tail jump-induced skew (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 90.0, OptionType.Call), e["call_skew_low"])


def test_amst_call_skew_high(cpp_refs: dict[str, Any]) -> None:
    """K=110 call captures right-tail jump skew (LOOSE)."""
    bates_engine, expiry = _make_bates_engine()
    e = cpp_refs["bates_engine_amst"]
    loose(_price(bates_engine, expiry, 110.0, OptionType.Call), e["call_skew_high"])


# ---------------------------------------------------------------------
# Bates 1996 reference (rho=-0.5).
# ---------------------------------------------------------------------


def test_bates_1996_call_atm(cpp_refs: dict[str, Any]) -> None:
    """ATM call under the Bates 1996 scenario vs C++ reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine(rho=_RHO_B96)
    e = cpp_refs["bates_engine_b96"]
    loose(_price(bates_engine, expiry, 100.0, OptionType.Call), e["call_atm"])


def test_bates_1996_put_atm(cpp_refs: dict[str, Any]) -> None:
    """ATM put under the Bates 1996 scenario vs C++ reference (LOOSE)."""
    bates_engine, expiry = _make_bates_engine(rho=_RHO_B96)
    e = cpp_refs["bates_engine_b96"]
    loose(_price(bates_engine, expiry, 100.0, OptionType.Put), e["put_atm"])


# ---------------------------------------------------------------------
# Algebraic + structural tests.
# ---------------------------------------------------------------------


def test_put_call_parity_with_jumps() -> None:
    """C - P = S*df_q - K*df_r at jumps-on ATM.

    Algebraic identity that holds for the Bates dynamics (the
    martingale jump compensator was designed precisely so that the
    discounted spot stays a martingale). Two engine integrations
    times ~1e-10 quad noise → TIGHT.
    """
    bates_engine, expiry = _make_bates_engine()
    call = _price(bates_engine, expiry, 100.0, OptionType.Call)
    put = _price(bates_engine, expiry, 100.0, OptionType.Put)
    expected = _S - 100.0 * math.exp(-_R * _T)
    tight(
        call - put,
        expected,
        reason="put-call parity: C - P = S*df_q - K*df_r at q=0",
    )


def test_engine_holds_bates_model_reference() -> None:
    """``model()`` returns the BatesModel (narrowed) passed to the ctor."""
    bates_engine, _ = _make_bates_engine()
    m = bates_engine.model()
    assert isinstance(m, BatesModel)
    # nu/delta/lambda accessors should be reachable without an
    # explicit cast. ConstantParameter evaluation is exact at t=0.
    tight(m.lambda_(), _LAMBDA, reason="ConstantParameter exact roundtrip")
    tight(m.nu(), _NU, reason="ConstantParameter exact roundtrip")
    tight(m.delta(), _DELTA, reason="ConstantParameter exact roundtrip")


def test_jumps_change_atm_npv() -> None:
    """Switching jumps on shifts the ATM call price by O(lambda).

    Sanity check: at lambda=0.1 the BatesEngine ATM call must differ
    from the AnalyticHestonEngine ATM call by a non-negligible
    amount (the C++ probe shows ~0.11 difference: 10.50 vs 10.39).
    """
    bates_engine, expiry = _make_bates_engine()
    heston_engine, _ = _make_heston_engine()
    bates = _price(bates_engine, expiry, 100.0, OptionType.Call)
    heston = _price(heston_engine, expiry, 100.0, OptionType.Call)
    # Sign: with nu=-0.05 < 0 (downside jumps) the put-call parity
    # makes call MORE expensive when jumps are added (more left tail,
    # less right tail). Actually both wings inflate; net call is up.
    assert bates - heston > 0.05, (
        f"jumps should shift ATM call up by >0.05; got bates={bates} heston={heston}"
    )
