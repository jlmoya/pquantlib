"""Cross-validate CEVCalculator and AnalyticCEVEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``cev_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

What these tests are actually defending
---------------------------------------
1. **Both CEV regimes.** ``delta = (1-2 beta)/(1-beta)`` splits at ``delta = 2``,
   i.e. at ``beta = 1``, and the two sides are different formulae — the
   ``beta > 1`` call branch carries a ``gamma_p`` factor the put branch does not.
   Nine betas from -2.0 to 2.0 x four strikes x both types are pinned.
2. **The economics of each regime.** For ``beta < 1`` the CEV process is a true
   martingale and put-call parity holds to machine precision; for ``beta > 1`` it
   is a strict *local* martingale and parity is violated by a computable amount
   (-0.228 on the reference market). A port that "fixed" the asymmetric branch
   would restore parity and be wrong.
3. **The ``beta -> 1`` lognormal limit**, bracketed at 0.99 and 1.01 around the
   Black-76 value.
4. **Which distribution routine is in play.** C++ uses Boost, not QuantLib's own
   ``NonCentralCumulativeChiSquareDistribution`` (an absolute-error series that
   disagrees with Boost by orders of magnitude in the tail). Five
   ``cev_calc_far_tail_call_*`` points sit where the two part company.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) throughout. The calculator is two distribution
evaluations and a subtraction; the question is only whether SciPy's ``ncx2.cdf``
and ``gammainc`` are the same computation as Boost's. Measured over all 77 pinned
calculator values plus the engine and parity cases, the worst relative deviation
is 4.4e-15 — including the deliberately deep-tail points, which is what makes the
delegation cross-validated rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_cev_engine import (
    AnalyticCEVEngine,
    CEVCalculator,
)
from pquantlib.testing import reference_reader, tolerance

from ._hybrid_v143 import REFERENCE_KEY, TODAY, flat_curve, maturity_date, option_type

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)


def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF, used only to build an independent Black reference."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

CALCULATOR_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("cev_calc_") and "value" in case["expected"]
)
INSPECTOR_CASES = sorted(
    name for name in CPP if name.startswith("cev_calc_inspectors_")
)
ENGINE_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("cev_engine_") and "npv" in case["expected"]
)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:1365 — Settings::instance().evaluationDate() = kToday,
    # kToday = Date(1, March, 2025) at probe.cpp:225.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


@pytest.mark.parametrize("case_name", CALCULATOR_CASES)
def test_calculator_value_matches_cpp(case_name: str) -> None:
    """``CEVCalculator::value`` reproduces C++ across both regimes and both types."""
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    calc = CEVCalculator(inputs["f0"], inputs["alpha"], inputs["beta"])
    tolerance.tight(
        calc.value(option_type(inputs["optionType"]), inputs["strike"], inputs["t"]),
        expected["value"],
    )


@pytest.mark.parametrize("case_name", INSPECTOR_CASES)
def test_calculator_inspectors(case_name: str) -> None:
    """``f0()`` / ``alpha()`` / ``beta()`` are the whole public read surface.

    ``X(f)`` and the cached ``delta_`` are private in C++
    (analyticcevengine.hpp:58-60) and are private here too.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    calc = CEVCalculator(inputs["f0"], inputs["alpha"], inputs["beta"])
    tolerance.exact(calc.f0(), expected["f0"])
    tolerance.exact(calc.alpha(), expected["alpha"])
    tolerance.exact(calc.beta(), expected["beta"])
    assert not hasattr(calc, "X")


def test_both_regimes_are_exercised() -> None:
    """Guard the guard: the pinned betas really do straddle ``delta = 2``."""
    betas = {CPP[name]["inputs"]["beta"] for name in CALCULATOR_CASES}
    deltas = [(1.0 - 2.0 * b) / (1.0 - b) for b in betas]
    assert any(d < 2.0 for d in deltas), "no beta < 1 case"
    assert any(d >= 2.0 for d in deltas), "no beta > 1 case"
    # And the probe's own `delta` annotations agree with that arithmetic.
    for name in CALCULATOR_CASES:
        inputs = CPP[name]["inputs"]
        if "delta" in inputs:
            beta = inputs["beta"]
            tolerance.tight((1.0 - 2.0 * beta) / (1.0 - beta), inputs["delta"])


def _engine_option(inputs: dict[str, Any]) -> VanillaOption:
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(
        AnalyticCEVEngine(
            inputs["f0"],
            inputs["alpha"],
            inputs["beta"],
            flat_curve(inputs["discountRate"]),
        )
    )
    return option


@pytest.mark.parametrize("case_name", ENGINE_CASES)
def test_engine_npv_matches_cpp(case_name: str) -> None:
    """The engine discounts the calculator value with ``discountCurve->discount``."""
    case = CPP[case_name]
    tolerance.tight(_engine_option(case["inputs"]).npv(), case["expected"]["npv"])


@pytest.mark.parametrize("case_name", ENGINE_CASES)
def test_engine_npv_is_the_discounted_calculator_value(case_name: str) -> None:
    """The engine adds exactly one thing to the calculator: the discount factor.

    # C++ parity: analyticcevengine.cpp:108-112.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    curve = flat_curve(inputs["discountRate"])
    calc = CEVCalculator(inputs["f0"], inputs["alpha"], inputs["beta"])
    undiscounted = calc.value(
        option_type(inputs["optionType"]),
        inputs["strike"],
        curve.time_from_reference(maturity_date(inputs)),
    )
    tolerance.tight(
        undiscounted * curve.discount(maturity_date(inputs)), expected["npv"]
    )
    tolerance.tight(curve.discount(maturity_date(inputs)), expected["discount"])


def test_put_call_parity_holds_below_beta_one() -> None:
    """``beta < 1``: the CEV process is a true martingale, so parity is exact."""
    case = CPP["cev_parity_beta045"]
    inputs, expected = case["inputs"], case["expected"]
    assert inputs["beta"] < 1.0
    curve = flat_curve(0.15)
    calc = CEVCalculator(inputs["f0"], inputs["alpha"], inputs["beta"])
    t = curve.time_from_reference(TODAY + 365)
    df = curve.discount(TODAY + 365)
    call = calc.value(OptionType.Call, inputs["strike"], t) * df
    put = calc.value(OptionType.Put, inputs["strike"], t) * df
    tolerance.tight(call, expected["call_npv"])
    tolerance.tight(put, expected["put_npv"])
    residual = call - put - (inputs["f0"] - inputs["strike"]) * df
    tolerance.tight(residual, expected["parity_residual"])
    # C++'s own residual is -2.2e-16; anything at that scale is round-off.
    assert abs(residual) < 1e-14


def test_put_call_parity_fails_above_beta_one() -> None:
    """``beta > 1``: a strict local martingale, so parity is genuinely violated.

    This is not a defect to be fixed. ``E[F_t] < F_0`` for ``beta > 1``, and the
    call branch's extra ``gamma_p(delta/2 - 1, x0/(2t))`` factor is exactly the
    mass that has leaked to the absorbing boundary. The residual is a real,
    reproducible -0.228 on this market.
    """
    case = CPP["cev_parity_beta145"]
    inputs, expected = case["inputs"], case["expected"]
    assert inputs["beta"] > 1.0
    curve = flat_curve(0.15)
    calc = CEVCalculator(inputs["f0"], inputs["alpha"], inputs["beta"])
    t = curve.time_from_reference(TODAY + 365)
    df = curve.discount(TODAY + 365)
    call = calc.value(OptionType.Call, inputs["strike"], t) * df
    put = calc.value(OptionType.Put, inputs["strike"], t) * df
    tolerance.tight(call, expected["call_npv"])
    tolerance.tight(put, expected["put_npv"])
    residual = call - put - (inputs["f0"] - inputs["strike"]) * df
    tolerance.tight(residual, expected["parity_residual"])
    assert residual < -0.2, "the local-martingale defect must survive the port"


def test_beta_approaching_one_is_the_lognormal_limit() -> None:
    """``beta -> 1`` turns CEV into Black-76 with volatility ``alpha``.

    The 0.99 and 1.01 cases must bracket the Black value — they sit on opposite
    sides of the ``delta = 2`` branch, so this also checks the two formulae agree
    across the split rather than merely each being self-consistent.
    """
    f0, alpha, strike, t = 2.1, 0.75, 2.3, 1.0
    below = CEVCalculator(f0, alpha, 0.99).value(OptionType.Call, strike, t)
    above = CEVCalculator(f0, alpha, 1.01).value(OptionType.Call, strike, t)
    tolerance.tight(below, CPP["cev_calc_b0_99_k2_3_call"]["expected"]["value"])
    tolerance.tight(above, CPP["cev_calc_b1_01_k2_3_call"]["expected"]["value"])

    std_dev = alpha * math.sqrt(t)
    d1 = (math.log(f0 / strike) + 0.5 * std_dev * std_dev) / std_dev
    d2 = d1 - std_dev
    black = f0 * _std_normal_cdf(d1) - strike * _std_normal_cdf(d2)
    assert below < black < above
    # Each side is within ~1% of the lognormal limit at |beta - 1| = 0.01.
    assert abs(below - black) / black < 0.01
    assert abs(above - black) / black < 0.01


def test_american_exercise_throws() -> None:
    """# C++ parity: analyticcevengine.cpp:98-99."""
    assert CPP["cev_engine_american_exercise_throws"]["expected"]["throws"] is True
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 2.3),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(AnalyticCEVEngine(2.1, 0.75, 0.45, flat_curve(0.15)))
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()


def test_no_greeks_provided() -> None:
    """# C++ parity: calculate() assigns only ``results_.value``."""
    expected = CPP["cev_engine_no_greeks"]["expected"]
    assert expected == {"delta_throws": True, "vega_throws": True}
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 2.3), EuropeanExercise(TODAY + 365)
    )
    option.set_pricing_engine(AnalyticCEVEngine(2.1, 0.75, 0.45, flat_curve(0.15)))
    option.npv()
    with pytest.raises(LibraryException, match="delta not provided"):
        option.delta()
    with pytest.raises(LibraryException, match="vega not provided"):
        option.vega()
