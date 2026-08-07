"""Cross-validate AnalyticHestonHullWhiteEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``hhw_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

The single best correctness check available for this engine is the
deterministic-rate limit: with the Hull-White volatility driven to zero the
add-on term vanishes and the price must collapse *exactly* onto the plain
``AnalyticHestonEngine``. A port that got the sign of the add-on term wrong, or
applied it to only one of the two characteristic functions, or computed ``m`` for
the wrong maturity, fails that immediately.
:func:`test_zero_hull_white_vol_collapses_to_heston` asserts it against C++'s own
bit-identical pair.

Also pinned: that the Hull-White volatility moves the price at all (five
volatilities, strictly increasing), the ``a -> 0`` algebraic branch straddled to
one ULP of the ``a*t > 2**-13`` threshold, and the absence of Greeks.

Both C++ constructors are exercised: ``__init__`` (Gauss-Laguerre, order 144) and
``with_lobatto`` (adaptive Gauss-Lobatto). The probe's primary values are taken
at ``gaussLobatto(1e-13, 1e6)`` — the converged configuration — and
``hhw_gausslaguerre144_vs_lobatto`` additionally pins the 144-node value, so both
Python code paths are compared against the C++ path they correspond to rather
than against a single "converged" number that would hide a wrong node set.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) for every NPV. The two libraries now run the *same*
quadrature — PQuantLib ports ``AnalyticHestonEngine::Integration``, including
``GaussLobattoIntegral`` and ``GaussLaguerreIntegration`` — so agreement is
limited by floating-point round-off through an identical node set rather than by
two integrators independently converging. Measured over the 17 pinned values the
worst relative deviation is 2.0e-14, two decades inside the tier.

The one exception is :func:`test_branch_threshold_gap_reproduces_cpp`, which
compares a *difference* of two nearly equal NPVs and therefore has its own
derived tolerance; the derivation is in that test's docstring.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import AnalyticHestonEngine
from pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine import (
    AnalyticHestonHullWhiteEngine,
)
from pquantlib.testing import reference_reader, tolerance

from ._hybrid_v143 import (
    NULL_REAL,
    REFERENCE_KEY,
    TODAY,
    flat_curve,
    heston_model,
    maturity_date,
    option_type,
)

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)

HHW_NPV_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("hhw_") and "npv" in case["expected"]
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


def _build(inputs: dict[str, Any]) -> VanillaOption:
    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]),
        a=inputs["hwA"],
        sigma=inputs["hwSigma"],
    )
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(
        AnalyticHestonHullWhiteEngine.with_lobatto(
            model, hull_white, inputs["relTolerance"], int(inputs["maxEvaluations"])
        )
    )
    return option


@pytest.mark.parametrize("case_name", HHW_NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """NPV reproduces C++ for every pinned market, both option types."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


def test_hull_white_vol_changes_the_price() -> None:
    """The five-volatility sweep must be strictly increasing.

    ``m`` is proportional to ``sigma^2`` and enters the characteristic function
    as ``exp(-m u^2 + i u m (3 - 2j))``, i.e. as extra variance in the forward
    measure. More short-rate volatility means more total variance, so a call is
    worth more. An engine that ignored ``sigma`` returns the Heston price five
    times.
    """
    sweep = sorted(
        (CPP[name]["inputs"]["hwSigma"], name)
        for name in HHW_NPV_CASES
        if name.startswith("hhw_hwvol_")
    )
    assert len(sweep) == 5
    prices = [_build(CPP[name]["inputs"]).npv() for _, name in sweep]
    assert len(set(prices)) == len(prices), "the Hull-White vol is being discarded"
    assert prices == sorted(prices)
    for (_, name), got in zip(sweep, prices, strict=True):
        tolerance.tight(got, CPP[name]["expected"]["npv"], reason=name)


def test_zero_hull_white_vol_collapses_to_heston() -> None:
    """The deterministic-rate limit — the strongest check this engine has.

    C++'s own two numbers are bit-identical here (``m`` is ~1e-24 at
    ``sigma = 1e-12``), so the Python pair must be too, up to the difference
    between two independent quadrature runs of the same integrand.
    """
    case = CPP["hhw_zero_hwvol_collapses_to_heston"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["npv"] == expected["analytic_heston_npv"], (
        "the probe's own pair should be bit-identical"
    )

    hybrid = _build(inputs)
    model = heston_model(inputs)
    plain = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    plain.set_pricing_engine(
        AnalyticHestonEngine.with_integration(
            model,
            AnalyticHestonEngine.Gatheral,
            AnalyticHestonEngine.Integration.gauss_lobatto(
                # NULL_REAL, not None: C++'s Null<Real>() for the absolute
                # tolerance is std::numeric_limits<float>::max(), and
                # Integrator's ctor requires absAccuracy > QL_EPSILON.
                inputs["relTolerance"],
                NULL_REAL,
                int(inputs["maxEvaluations"]),
            ),
        )
    )

    tolerance.tight(hybrid.npv(), expected["npv"])
    tolerance.tight(plain.npv(), expected["analytic_heston_npv"])
    tolerance.tight(hybrid.npv(), plain.npv())


@pytest.mark.parametrize(
    "case_name", ["hhw_low_a_branch", "hhw_a_at_threshold", "hhw_a_just_above_threshold"]
)
def test_low_a_branch(case_name: str) -> None:
    """``a*t > pow(QL_EPSILON, 0.25)`` is a strict ``>`` against exactly ``2**-13``."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


def test_branch_threshold_gap_reproduces_cpp() -> None:
    """One ULP of ``a`` flips the branch, and the resulting jump is C++'s jump.

    As in the BSM/Hull-White engine, the general branch evaluates a difference of
    O(1/a) terms whose true value is O(a^2 t^3), so ~12 digits cancel at the
    threshold and the two branches cannot agree to better than ~1e-7 relative.
    Comparing the *gap* to C++'s gap is what proves the port lands on the same
    side of the strict ``>`` and evaluates the cancelling sum in the same order.

    Tolerance derivation. The gap is 1.29e-6 formed by subtracting two NPVs of
    10.02, so it is itself a cancellation of seven decimal digits. Each NPV comes
    from an adaptive quadrature and matches C++ to ~1e-14 relative — 1e-13
    absolute — so the difference inherits ~2e-13 of absolute error, i.e. 1.6e-7
    relative. TIGHT (1e-12 rel) is unreachable here for arithmetic reasons and
    would be a false pass if it happened to hold; 1e-6 relative is the honest
    bound with half a decade of headroom, and it is still four orders of
    magnitude smaller than what taking the wrong branch does (the gap would
    change sign or vanish entirely).
    """
    below = _build(CPP["hhw_a_at_threshold"]["inputs"]).npv()
    above = _build(CPP["hhw_a_just_above_threshold"]["inputs"]).npv()
    cpp_gap = (
        CPP["hhw_a_at_threshold"]["expected"]["npv"]
        - CPP["hhw_a_just_above_threshold"]["expected"]["npv"]
    )
    assert below != above
    tolerance.custom(
        below - above,
        cpp_gap,
        abs_tol=1e-12,
        rel_tol=1e-6,
        reason="difference of two ~10.02 NPVs; 7 digits cancel, leaving ~1.6e-7",
    )


def test_both_constructors_match_their_own_cpp_configuration() -> None:
    """Gauss-Laguerre(144) and adaptive Gauss-Lobatto are compared separately.

    C++'s two constructors differ only in the ``Integration`` policy they hand to
    the base engine, and the probe records both answers for one market. Checking
    each Python constructor against its own C++ number — rather than both against
    the converged one — is what proves the 144-node Gauss-Laguerre path uses the
    right nodes and weights; the two C++ answers differ by only 2e-15 relative, so
    a shared assertion would not distinguish them.
    """
    case = CPP["hhw_gausslaguerre144_vs_lobatto"]
    inputs, expected = case["inputs"], case["expected"]
    laguerre = expected["gauss_laguerre_144_npv"]
    lobatto = expected["gauss_lobatto_converged_npv"]

    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    payoff = PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"])

    by_order = VanillaOption(payoff, EuropeanExercise(maturity_date(inputs)))
    by_order.set_pricing_engine(
        AnalyticHestonHullWhiteEngine(
            model, hull_white, int(inputs["integrationOrder"])
        )
    )
    tolerance.tight(by_order.npv(), laguerre)
    tolerance.tight(_build(inputs).npv(), lobatto)

    # And the quadrature choice itself is not the limiting term at these settings.
    assert abs(laguerre - lobatto) / lobatto < 1e-12


def test_integration_order_above_192_throws() -> None:
    """# C++ parity: ``Integration::gaussLaguerre`` QL_REQUIREs intOrder <= 192."""
    assert CPP["hhw_integration_order_over_192_throws"]["expected"]["throws"] is True
    inputs = CPP["hhw_call_k100_5y"]["inputs"]
    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    with pytest.raises(LibraryException, match="maximum integraton order"):
        AnalyticHestonHullWhiteEngine(model, hull_white, 1024)


def test_no_greeks_provided() -> None:
    """# C++ parity: inherits AnalyticHestonEngine::calculate — value only."""
    expected = CPP["hhw_no_greeks"]["expected"]
    assert expected == {"delta_throws": True, "vega_throws": True}
    option = _build(CPP["hhw_call_k100_5y"]["inputs"])
    option.npv()
    with pytest.raises(LibraryException, match="delta not provided"):
        option.delta()
    with pytest.raises(LibraryException, match="vega not provided"):
        option.vega()


def test_american_exercise_throws() -> None:
    """# C++ parity: the inherited AnalyticHestonEngine::calculate rejects it."""
    assert CPP["hhw_american_exercise_throws"]["expected"]["throws"] is True
    inputs = CPP["hhw_call_k100_5y"]["inputs"]
    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(AnalyticHestonHullWhiteEngine(model, hull_white))
    with pytest.raises(LibraryException, match="not a European option"):
        option.npv()


def test_add_on_term_signs() -> None:
    """``addOnTerm(u, t, j) = complex(-m u^2, u (m - 2 m (j-1)))``.

    # C++ parity: analytichestonhullwhiteengine.hpp:95-100. The imaginary part
    # flips sign between the two characteristic functions; a port that used the
    # same sign for both still prices *something*, and the collapse test above
    # would not necessarily catch it because both add-ons vanish there.
    """
    inputs = CPP["hhw_call_k100_5y"]["inputs"]
    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    engine = AnalyticHestonHullWhiteEngine.with_lobatto(
        model, hull_white, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(engine)
    option.npv()  # populates m

    u, t = 0.75, 5.0
    term1 = engine.add_on_term(u, t, 1)
    term2 = engine.add_on_term(u, t, 2)
    assert term1.real == term2.real
    assert term1.real < 0.0
    tolerance.exact(term1.imag, -term2.imag)
    assert term1.imag > 0.0
    assert engine.hull_white_model() is hull_white


def test_add_on_term_vanishes_when_hull_white_vol_is_zero() -> None:
    """With ``m == 0`` the engine is exactly ``AnalyticHestonEngine``."""
    inputs = CPP["hhw_zero_hwvol_collapses_to_heston"]["inputs"]
    model = heston_model(inputs)
    hull_white = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    engine = AnalyticHestonHullWhiteEngine.with_lobatto(
        model, hull_white, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(engine)
    option.npv()
    for j in (1, 2):
        assert abs(engine.add_on_term(1.0, 5.0, j)) < 1e-20
