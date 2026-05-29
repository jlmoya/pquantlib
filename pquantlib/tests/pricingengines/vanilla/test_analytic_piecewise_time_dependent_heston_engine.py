"""AnalyticPiecewiseTimeDependentHestonEngine tests.

Cross-validates against ``migration-harness/references/cluster/w1d.json``.

C++ parity: ql/pricingengines/vanilla/analyticptdhestonengine.{hpp,cpp}
            @ v1.42.1 (099987f0) — Gatheral form only.

Tolerance choice:

* Degenerate case (PTD vs plain Heston): TIGHT — at all-equal piecewise
  params the PTD engine must reduce algebraically to plain Heston. The
  C++ reference confirms ~12-digit agreement; we accept tight.
* 2-segment NPV vs C++: LOOSE (abs_tol=1e-8) — same reason as
  AnalyticHestonEngine: scipy.quad on (0, +inf) diverges from C++
  Gauss-Laguerre at ~1e-8 absolute on multi-segment integrands.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.equity.piecewise_time_dependent_heston_model import (
    PiecewiseTimeDependentHestonModel,
)
from pquantlib.models.parameter import PiecewiseConstantParameter
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)
from pquantlib.pricingengines.vanilla.analytic_piecewise_time_dependent_heston_engine import (
    AnalyticPiecewiseTimeDependentHestonEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

_S = 100.0
_R = 0.05
_Q = 0.0
_V0 = 0.04


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1d")


def _build_degenerate_model() -> PiecewiseTimeDependentHestonModel:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)

    breaks = [0.5]
    theta = PiecewiseConstantParameter(breaks, PositiveConstraint())
    theta.set_param(0, 0.04)
    theta.set_param(1, 0.04)
    kappa = PiecewiseConstantParameter(breaks, PositiveConstraint())
    kappa.set_param(0, 2.0)
    kappa.set_param(1, 2.0)
    sigma = PiecewiseConstantParameter(breaks, PositiveConstraint())
    sigma.set_param(0, 0.3)
    sigma.set_param(1, 0.3)
    rho = PiecewiseConstantParameter(breaks, BoundaryConstraint(-1.0, 1.0))
    rho.set_param(0, -0.7)
    rho.set_param(1, -0.7)

    return PiecewiseTimeDependentHestonModel(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=_V0,
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        time_grid=TimeGrid.with_mandatory([0.5, 1.0]),
    )


def _build_two_segment_model() -> PiecewiseTimeDependentHestonModel:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)

    breaks = [0.5]
    theta = PiecewiseConstantParameter(breaks, PositiveConstraint())
    theta.set_param(0, 0.06)
    theta.set_param(1, 0.04)
    kappa = PiecewiseConstantParameter(breaks, PositiveConstraint())
    kappa.set_param(0, 2.5)
    kappa.set_param(1, 1.5)
    sigma = PiecewiseConstantParameter(breaks, PositiveConstraint())
    sigma.set_param(0, 0.4)
    sigma.set_param(1, 0.2)
    rho = PiecewiseConstantParameter(breaks, BoundaryConstraint(-1.0, 1.0))
    rho.set_param(0, -0.8)
    rho.set_param(1, -0.5)

    return PiecewiseTimeDependentHestonModel(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=_V0,
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        time_grid=TimeGrid.with_mandatory([0.5, 1.0]),
    )


def _plain_heston_engine(
    kappa: float, theta: float, sigma: float, rho: float, v0: float
) -> AnalyticHestonEngine:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=_R, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=_Q, day_counter=dc)
    proc = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(_S),
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
    )
    return AnalyticHestonEngine(HestonModel(proc), 144)


def _expiry() -> Date:
    return Date.from_ymd(15, Month.June, 2026) + 365


def test_degenerate_call_matches_plain_heston(
    cpp_refs: dict[str, Any],
) -> None:
    """At all-equal piecewise params, PTD engine NPV == plain Heston NPV.

    # C++ parity: analyticptdhestonengine.cpp Gatheral branch reduces
    # to analytichestonengine.cpp Gatheral branch.
    """
    ptd_model = _build_degenerate_model()
    ptd_engine = AnalyticPiecewiseTimeDependentHestonEngine(ptd_model, 144)
    plain_engine = _plain_heston_engine(2.0, 0.04, 0.3, -0.7, _V0)

    payoff = PlainVanillaPayoff(OptionType.Call, _S)
    exercise = EuropeanExercise(_expiry())

    ptd_opt = VanillaOption(payoff, exercise)
    ptd_opt.set_pricing_engine(ptd_engine)
    plain_opt = VanillaOption(payoff, exercise)
    plain_opt.set_pricing_engine(plain_engine)

    tight(
        ptd_opt.npv(),
        plain_opt.npv(),
        reason=(
            "all-equal piecewise params: PTD Riccati accumulation algebraically "
            "matches plain Heston"
        ),
    )

    # Cross-validate against C++ reference for the degenerate case too.
    loose(
        ptd_opt.npv(),
        cpp_refs["plain_heston_reference"]["call_atm_1y"],
        reason="scipy.quad vs C++ Gauss-Laguerre diverges at ~1e-8",
    )


def test_degenerate_put_matches_plain_heston() -> None:
    """Put NPV — same logic as call."""
    ptd_model = _build_degenerate_model()
    ptd_engine = AnalyticPiecewiseTimeDependentHestonEngine(ptd_model, 144)
    plain_engine = _plain_heston_engine(2.0, 0.04, 0.3, -0.7, _V0)

    payoff = PlainVanillaPayoff(OptionType.Put, _S)
    exercise = EuropeanExercise(_expiry())

    ptd_opt = VanillaOption(payoff, exercise)
    ptd_opt.set_pricing_engine(ptd_engine)
    plain_opt = VanillaOption(payoff, exercise)
    plain_opt.set_pricing_engine(plain_engine)

    tight(ptd_opt.npv(), plain_opt.npv(), reason="degenerate put reduces to Heston put")


def test_two_segment_call_matches_cpp(cpp_refs: dict[str, Any]) -> None:
    """2-segment piecewise NPV matches C++ AnalyticPTDHestonEngine."""
    model = _build_two_segment_model()
    engine = AnalyticPiecewiseTimeDependentHestonEngine(model, 144)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, _S), EuropeanExercise(_expiry())
    )
    option.set_pricing_engine(engine)
    expected = cpp_refs["ptd_heston_2segment"]["call_atm_1y"]
    loose(option.npv(), expected, reason="scipy.quad vs C++ Gauss-Laguerre")


def test_two_segment_put_matches_cpp(cpp_refs: dict[str, Any]) -> None:
    """2-segment piecewise put NPV matches C++."""
    model = _build_two_segment_model()
    engine = AnalyticPiecewiseTimeDependentHestonEngine(model, 144)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Put, _S), EuropeanExercise(_expiry())
    )
    option.set_pricing_engine(engine)
    expected = cpp_refs["ptd_heston_2segment"]["put_atm_1y"]
    loose(option.npv(), expected, reason="scipy.quad vs C++ Gauss-Laguerre")


def test_engine_inspector_returns_model() -> None:
    """``engine.model()`` returns the supplied PTD model."""
    model = _build_two_segment_model()
    engine = AnalyticPiecewiseTimeDependentHestonEngine(model, 144)
    assert engine.model() is model
