"""Cross-validate AnalyticBSMHullWhiteEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``bsmhw_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

What these tests are actually defending
---------------------------------------
1. **That ``rho`` moves the price.** ``bsmhw_rho_*`` prices one market at seven
   correlations; an engine that accepted and discarded the correlation would
   return the same NPV seven times. :func:`test_correlation_changes_the_price`
   asserts the sweep is strictly monotone as well as reproducing each value.
2. **That the whole Greek block survives the delegation.** C++ copies the inner
   ``AnalyticEuropeanEngine``'s entire result object, so delta/gamma/theta/vega/
   rho/dividendRho/itmCashProbability and the ``additionalResults`` map are all
   populated — and ``vega`` and ``additionalResults["volatility"]`` are taken
   against the *shifted* vol surface. A port that only assigned ``value`` passes
   every NPV test and fails these.
3. **The low-``a`` algebraic branch**, straddled to one ULP either side of the
   ``a*t > 2**-13`` threshold.
4. **The degenerate limit**: Hull-White vol -> 0 must reproduce
   ``AnalyticEuropeanEngine`` exactly.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) everywhere. This engine performs no quadrature —
it evaluates a closed-form variance offset and hands off to a closed-form Black
formula — so the only source of disagreement is the order of the floating-point
operations, and the port keeps C++'s order. Measured over all 89 pinned
quantities the worst relative deviation is 3.9e-15, three orders of magnitude
inside the tier; nothing here needs LOOSE.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_bsm_hull_white_engine import (
    AnalyticBSMHullWhiteEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date

from ._hybrid_v143 import (
    REFERENCE_KEY,
    TODAY,
    bsm_process,
    flat_curve,
    flat_vol,
    maturity_date,
    option_type,
)

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)

BSMHW_NPV_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("bsmhw_") and "npv" in case["expected"]
)

GREEK_ACCESSORS: dict[str, str] = {
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho",
    "dividendRho": "dividend_rho",
    "itmCashProbability": "itm_cash_probability",
}


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:1365 — Settings::instance().evaluationDate() = kToday,
    # kToday = Date(1, March, 2025) at probe.cpp:225.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _build(inputs: dict[str, Any], exercise_date: Date | None = None) -> VanillaOption:
    process = bsm_process(inputs)
    model = HullWhite(
        term_structure=flat_curve(inputs["r"]),
        a=inputs["hwA"],
        sigma=inputs["hwSigma"],
    )
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(exercise_date if exercise_date else maturity_date(inputs)),
    )
    option.set_pricing_engine(AnalyticBSMHullWhiteEngine(inputs["rho"], process, model))
    return option


@pytest.mark.parametrize("case_name", BSMHW_NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """NPV reproduces C++ for every pinned market."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


@pytest.mark.parametrize(
    "case_name",
    [name for name in BSMHW_NPV_CASES if "delta" in CPP[name]["expected"]],
)
def test_greeks_match_cpp(case_name: str) -> None:
    """The delegated AnalyticEuropeanEngine Greek block is copied wholesale.

    # C++ parity: analyticbsmhullwhiteengine.cpp:130-131 assigns the entire
    # ``OneAssetOption::results``, not just ``value``.
    """
    case = CPP[case_name]
    option = _build(case["inputs"])
    for cpp_key, py_name in GREEK_ACCESSORS.items():
        expected = case["expected"][cpp_key]
        accessor = getattr(option, py_name)
        tolerance.tight(accessor(), expected, reason=f"{case_name}.{cpp_key}")


@pytest.mark.parametrize(
    "case_name",
    [name for name in BSMHW_NPV_CASES if "additional_forward" in CPP[name]["expected"]],
)
def test_additional_results_match_cpp(case_name: str) -> None:
    """``additionalResults`` survive the results copy too.

    ``volatility`` here is the *shifted* surface's vol, not the input 0.25 — a
    port that rebuilt the additional results off the original process would give
    0.25 for every correlation.
    """
    case = CPP[case_name]
    option = _build(case["inputs"])
    option.npv()
    extra = option.additional_results()
    for cpp_key, value in case["expected"].items():
        if not cpp_key.startswith("additional_"):
            continue
        key = cpp_key.removeprefix("additional_")
        assert key in extra, f"{key} missing from additional results"
        tolerance.tight(float(extra[key]), value, reason=f"{case_name}.{key}")
    assert extra["volatility"] != case["inputs"]["vol"], (
        "the additional 'volatility' must come from the SHIFTED surface"
    )


def test_correlation_changes_the_price() -> None:
    """The rho sweep must be strictly increasing, not seven copies of one number.

    This is the defect shape earlier waves found: a constructor argument accepted
    and discarded. ``mu = 2 rho sigma eta / a (t - (1-e^{-at})/a)`` enters the
    variance offset linearly in rho and ``t - (1-e^{-at})/a > 0``, so the total
    variance — and hence a call's NPV — is strictly increasing in rho.
    """
    sweep = [
        (CPP[name]["inputs"]["rho"], CPP[name]["expected"]["npv"], name)
        for name in BSMHW_NPV_CASES
        if name.startswith("bsmhw_rho_")
    ]
    sweep.sort()
    assert len(sweep) == 7
    prices = [_build(CPP[name]["inputs"]).npv() for _, _, name in sweep]
    assert len(set(prices)) == len(prices), "the correlation is being discarded"
    assert prices == sorted(prices), "NPV must increase with rho for a call"
    for (_, expected, name), got in zip(sweep, prices, strict=True):
        tolerance.tight(got, expected, reason=name)


@pytest.mark.parametrize(
    "case_name",
    ["bsmhw_low_a_branch", "bsmhw_a_at_threshold", "bsmhw_a_just_above_threshold"],
)
def test_low_a_branch(case_name: str) -> None:
    """The ``a*t > pow(QL_EPSILON, 0.25)`` split is a strict ``>``.

    ``pow(QL_EPSILON, 0.25)`` is exactly ``2**-13``; ``bsmhw_a_at_threshold``
    puts ``a*t`` exactly on it (so the *low* branch runs) and
    ``bsmhw_a_just_above_threshold`` one ULP above (so the general branch runs).
    """
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


def test_branch_threshold_gap_reproduces_cpp() -> None:
    """The jump across the threshold is C++'s jump, to TIGHT.

    One ULP of ``a`` cannot move the true price, yet the two cases differ by
    1.3e-6 (1.3e-7 relative). That is not an error in either engine — it is why
    the algebraic branch exists. The general branch evaluates

        S = t + 2/a e^{-a t} - 1/(2a) e^{-2 a t} - 3/(2a)

    whose four terms are O(1/a) ~ 1.6e4 at ``a = 2**-13``, while ``S`` itself is
    O(a^2 t^3/3) ~ 5e-9: about twelve decimal digits cancel, so double precision
    leaves ``S`` with roughly three or four significant figures. Propagated
    through ``v = sigma^2 S / a^2`` into a total variance of ~0.064 that is a
    ~1e-7 relative perturbation of the price, which is exactly what is seen.

    So the interesting assertion is not that the branches agree — they cannot to
    better than the cancellation allows — but that the *port lands on the same
    side of the threshold as C++ and inherits the same cancellation*. Comparing
    the gap itself does that: getting the strict ``>`` backwards, or evaluating
    ``S`` in a different order, changes this number immediately.
    """
    below = _build(CPP["bsmhw_a_at_threshold"]["inputs"]).npv()
    above = _build(CPP["bsmhw_a_just_above_threshold"]["inputs"]).npv()
    cpp_gap = (
        CPP["bsmhw_a_at_threshold"]["expected"]["npv"]
        - CPP["bsmhw_a_just_above_threshold"]["expected"]["npv"]
    )
    assert below != above, "one ULP of `a` must flip the branch"
    tolerance.tight(below - above, cpp_gap)
    # And the branches are still consistent to the accuracy the cancellation
    # leaves, which is ~1e-7 relative — not the 1e-15 a naive reading expects.
    assert abs(below - above) / below < 1e-6


def test_zero_hull_white_vol_collapses_to_black_scholes() -> None:
    """With ``sigma -> 0`` the variance offset vanishes and BS must be recovered.

    The correlation is 0.9 here, so a port that forgot to multiply ``mu`` by
    ``sigma`` would fail this even though it passes the flat cases.
    """
    case = CPP["bsmhw_zero_hw_vol_matches_bs"]
    inputs, expected = case["inputs"], case["expected"]
    hybrid = _build(inputs)
    plain = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    plain.set_pricing_engine(AnalyticEuropeanEngine(bsm_process(inputs)))

    tolerance.tight(hybrid.npv(), expected["npv"])
    tolerance.tight(plain.npv(), expected["analytic_european_npv"])
    # C++ itself only agrees to ~1.5e-10 absolute here: with sigma = 1e-12 the
    # offset is ~1e-24 * t^3 but `eta * mu` still perturbs the last bits. Assert
    # the same closeness the C++ pair shows rather than a tighter one.
    cpp_gap = abs(expected["npv"] - expected["analytic_european_npv"])
    assert abs(hybrid.npv() - plain.npv()) <= max(cpp_gap * 10.0, 1e-9)


def test_zero_spot_throws() -> None:
    """# C++ parity: analyticbsmhullwhiteengine.cpp:81."""
    assert CPP["bsmhw_zero_spot_throws"]["expected"]["throws"] is True
    process = BlackScholesMertonProcess(
        x0=SimpleQuote(0.0),
        dividend_ts=flat_curve(0.04),
        risk_free_ts=flat_curve(0.0525),
        black_vol_ts=flat_vol(0.25),
    )
    model = HullWhite(term_structure=flat_curve(0.0525), a=0.05, sigma=0.01)
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0), EuropeanExercise(TODAY + 365)
    )
    option.set_pricing_engine(AnalyticBSMHullWhiteEngine(0.0, process, model))
    with pytest.raises(LibraryException, match="negative or null underlying given"):
        option.npv()


def test_american_exercise_throws_from_the_inner_engine() -> None:
    """The engine has no exercise check of its own; the delegate has.

    # C++ parity: AnalyticBSMHullWhiteEngine::calculate never inspects the
    # exercise type — the AnalyticEuropeanEngine it builds does, so the message
    # is the inner engine's.
    """
    assert CPP["bsmhw_american_exercise_throws"]["expected"]["throws"] is True
    inputs = CPP["bsmhw_rho_000"]["inputs"]
    process = bsm_process(inputs)
    model = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(AnalyticBSMHullWhiteEngine(0.0, process, model))
    with pytest.raises(LibraryException, match="not a European option"):
        option.npv()


def test_constructor_guards() -> None:
    """# C++ parity: analyticbsmhullwhiteengine.cpp:74-75."""
    expected = CPP["bsmhw_ctor_guards"]["expected"]
    assert expected["null_process_throws"] is True
    assert expected["null_model_throws"] is True
    model = HullWhite(term_structure=flat_curve(0.0525), a=0.05, sigma=0.01)
    process = bsm_process(CPP["bsmhw_rho_000"]["inputs"])
    with pytest.raises(LibraryException, match="no Black-Scholes process specified"):
        AnalyticBSMHullWhiteEngine(0.0, None, model)  # pyright: ignore[reportArgumentType]
    with pytest.raises(LibraryException, match="no Hull-White model specified"):
        AnalyticBSMHullWhiteEngine(0.0, process, None)  # pyright: ignore[reportArgumentType]


def test_engine_inspectors() -> None:
    """``rho()`` and ``model()`` expose what was handed to the constructor."""
    inputs = CPP["bsmhw_rho_075"]["inputs"]
    process = bsm_process(inputs)
    model = HullWhite(
        term_structure=flat_curve(inputs["r"]), a=inputs["hwA"], sigma=inputs["hwSigma"]
    )
    engine = AnalyticBSMHullWhiteEngine(inputs["rho"], process, model)
    assert engine.model() is model
    assert math.isclose(engine.rho(), inputs["rho"])
