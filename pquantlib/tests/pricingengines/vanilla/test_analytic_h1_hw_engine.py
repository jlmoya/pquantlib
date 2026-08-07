"""Cross-validate AnalyticH1HWEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``h1hw_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

What these tests are actually defending
---------------------------------------
1. **That ``rho_sr`` moves the price**, over a four-point sweep, and that
   ``rho_sr = 0`` collapses *exactly* onto the
   ``AnalyticHestonHullWhiteEngine`` parent — which it must, because the whole
   H1-HW contribution is ``eta * rhoSr * I4``.
2. **Both branches of the ``E[sqrt(v_t)]`` fit.** ``8 kappa theta / gamma^2`` is
   1.333 for ``sigma_v = 0.3`` (closed-form ``a``, ``b``, ``c`` via
   ``LambdaApprox(1)``) and 0.333 for ``sigma_v = 0.6``, which instead runs the
   truncated hypergeometric series ``Lambda(1/kappa)`` with its 1000-term cap and
   its ``do { } while (s > eps && ++i < maxIter)`` off-by-one subtlety.
3. **The constructor asymmetry**: the Gauss-Laguerre constructor rejects a
   negative ``rho_sr``, the Gauss-Lobatto one does not — and what the second one
   then returns is not an option price.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel). Both libraries run the same Gauss-Lobatto
integration of the same integrand, and the H1-HW term itself is closed form apart
from the series in the ``sigma_v = 0.6`` branch, which is transcribed
term-for-term including its ``float``-epsilon stopping rule. Measured over the 17
pinned values the worst relative deviation is 5.2e-14 — the largest of the six
engines in this cluster, because ``Fj_Helper`` is rebuilt per integrand
evaluation and its ``I4`` involves four exponentials, but still 20x inside the
tier.

Sanity check beyond C++: the implied volatilities these NPVs generate reproduce
the Grzelak thesis table (0.267503, 0.235742, 0.228223, 0.223461, 0.217855 for
``sigma_v = 0.3``) to about 2e-5, the residual being that the probe uses a clean
3650-day maturity rather than the thesis's 15-Jul-2012 -> 13-Jul-2022.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_h1_hw_engine import AnalyticH1HWEngine
from pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine import (
    AnalyticHestonHullWhiteEngine,
)
from pquantlib.testing import reference_reader, tolerance

from ._hybrid_v143 import (
    REFERENCE_KEY,
    TODAY,
    flat_curve,
    heston_model,
    maturity_date,
    option_type,
)

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)


def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF, used only to build an independent Black reference."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

H1HW_NPV_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("h1hw_") and "npv" in case["expected"]
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


def _hull_white(inputs: dict[str, Any]) -> HullWhite:
    return HullWhite(
        term_structure=flat_curve(inputs["r"]),
        a=inputs["hwA"],
        sigma=inputs["hwSigma"],
    )


def _build(inputs: dict[str, Any]) -> VanillaOption:
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(
        AnalyticH1HWEngine.with_lobatto(
            heston_model(inputs),
            _hull_white(inputs),
            inputs["rhoSr"],
            inputs["relTolerance"],
            int(inputs["maxEvaluations"]),
        )
    )
    return option


@pytest.mark.parametrize("case_name", H1HW_NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """NPV reproduces C++ across both Fj_Helper branches and all strikes."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


def test_both_fj_helper_branches_are_exercised() -> None:
    """Guard the guard: the two vol-of-vols really do straddle the Feller ratio."""
    ratios = {
        CPP[name]["inputs"]["fellerRatio"]
        for name in H1HW_NPV_CASES
        if name.startswith(("h1hw_sv030_", "h1hw_sv060_"))
    }
    assert any(r > 1.0 for r in ratios), "no closed-form branch case"
    assert any(r <= 1.0 for r in ratios), "no Lambda() series branch case"
    for name in H1HW_NPV_CASES:
        inputs = CPP[name]["inputs"]
        if "fellerRatio" not in inputs:
            continue
        expected = (
            8.0
            * inputs["hestonKappa"]
            * inputs["hestonTheta"]
            / (inputs["hestonSigma"] ** 2)
        )
        tolerance.tight(expected, inputs["fellerRatio"])


def test_rho_sr_changes_the_price() -> None:
    """The rho_sr sweep must be strictly increasing, not four copies of one number.

    ``Fj_Helper`` returns ``eta * rhoSr * I4``, so the correlation scales the whole
    H1-HW term linearly. An engine that accepted and discarded ``rho_sr`` would
    return the Heston/Hull-White price four times.
    """
    sweep = sorted(
        (CPP[name]["inputs"]["rhoSr"], name)
        for name in H1HW_NPV_CASES
        if name.startswith("h1hw_rhosr_") and name[-3:].isdigit()
    )
    assert len(sweep) == 4
    prices = [_build(CPP[name]["inputs"]).npv() for _, name in sweep]
    assert len(set(prices)) == len(prices), "rho_sr is being discarded"
    assert prices == sorted(prices)
    for (_, name), got in zip(sweep, prices, strict=True):
        tolerance.tight(got, CPP[name]["expected"]["npv"], reason=name)


def test_zero_rho_sr_collapses_onto_the_parent_engine() -> None:
    """``rho_sr = 0`` kills the whole H1-HW term, so the parent price must return.

    C++'s own two numbers are bit-identical here. This is the H1-HW analogue of
    the Hull-White-vol-to-zero check: it isolates the added term from everything
    the parent already does.
    """
    case = CPP["h1hw_rhosr_zero_matches_hhw"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["npv"] == expected["heston_hull_white_npv"], (
        "the probe's own pair should be bit-identical"
    )

    h1hw = _build(inputs)
    parent = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    parent.set_pricing_engine(
        AnalyticHestonHullWhiteEngine.with_lobatto(
            heston_model(inputs),
            _hull_white(inputs),
            inputs["relTolerance"],
            int(inputs["maxEvaluations"]),
        )
    )

    tolerance.tight(h1hw.npv(), expected["npv"])
    tolerance.tight(parent.npv(), expected["heston_hull_white_npv"])
    tolerance.tight(h1hw.npv(), parent.npv())


def test_add_on_term_is_the_parent_term_plus_the_h1hw_term() -> None:
    """``addOnTerm = AnalyticHestonHullWhiteEngine::addOnTerm + Fj_Helper(...)(u)``.

    # C++ parity: analytich1hwengine.cpp:158-163. At ``rho_sr = 0`` the second
    # summand is identically zero, so the two engines' hooks must agree exactly;
    # at ``rho_sr > 0`` they must not.
    """
    inputs = CPP["h1hw_rhosr_060"]["inputs"]
    model = heston_model(inputs)
    hull_white = _hull_white(inputs)
    parent = AnalyticHestonHullWhiteEngine.with_lobatto(
        model, hull_white, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    zero_corr = AnalyticH1HWEngine.with_lobatto(
        model, hull_white, 0.0, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    positive = AnalyticH1HWEngine.with_lobatto(
        model, hull_white, 0.6, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    for engine in (parent, zero_corr, positive):
        option = VanillaOption(
            PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
            EuropeanExercise(maturity_date(inputs)),
        )
        option.set_pricing_engine(engine)
        option.npv()

    u, t = 0.6, 10.0
    for j in (1, 2):
        tolerance.exact(zero_corr.add_on_term(u, t, j).real, parent.add_on_term(u, t, j).real)
        tolerance.exact(zero_corr.add_on_term(u, t, j).imag, parent.add_on_term(u, t, j).imag)
        assert positive.add_on_term(u, t, j) != parent.add_on_term(u, t, j)
    assert math.isclose(positive.rho_sr(), 0.6)


def test_negative_rho_sr_constructor_asymmetry() -> None:
    """One C++ constructor rejects ``rho_sr < 0``; the other does not.

    # C++ parity: analytich1hwengine.cpp:146-147 sits on the integrationOrder
    # ctor only. The non-throwing path then returns roughly -3e16 on a spot of
    # 100 — the instability the other ctor is warning about. Only its absurdity
    # is asserted, never its digits: it is a divergent Fourier integral and is not
    # reproducible across integrators.
    """
    expected = CPP["h1hw_negative_rhosr_ctor_asymmetry"]["expected"]
    assert expected["order_ctor_throws"] is True
    assert expected["tolerance_ctor_throws"] is False
    assert expected["tolerance_ctor_npv_exceeds_spot"] is True

    inputs = CPP["h1hw_rhosr_060"]["inputs"]
    model = heston_model(inputs)
    hull_white = _hull_white(inputs)

    with pytest.raises(LibraryException, match="Fourier integration is not stable"):
        AnalyticH1HWEngine(model, hull_white, -0.5)

    engine = AnalyticH1HWEngine.with_lobatto(
        model, hull_white, -0.5, inputs["relTolerance"], int(inputs["maxEvaluations"])
    )
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(engine)
    assert abs(option.npv()) > inputs["spot"]


def test_implied_vols_reproduce_the_grzelak_table() -> None:
    """Independent evidence: the published H1-HW smile comes out.

    Not a C++ comparison — the C++ agreement is asserted above to 1e-14. This
    checks the *model* is the one Grzelak published, which no amount of matching
    a probe would establish on its own. The 1e-4 band is the thesis's own quoted
    tolerance plus the maturity difference (3650 days versus 15-Jul-2012 ->
    13-Jul-2022).
    """
    thesis = {
        "h1hw_sv030": [0.267503, 0.235742, 0.228223, 0.223461, 0.217855],
        "h1hw_sv060": [0.263626, 0.211625, 0.199907, 0.193502, 0.190025],
    }
    strikes = [40, 80, 100, 120, 180]
    for prefix, expected_vols in thesis.items():
        for strike, expected_vol in zip(strikes, expected_vols, strict=True):
            case = CPP[f"{prefix}_k{strike}"]
            inputs = case["inputs"]
            npv = _build(inputs).npv()
            t = 10.0
            spot, rate = inputs["spot"], inputs["r"]
            lo, hi = 1e-6, 3.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                sqrt_t = mid * math.sqrt(t)
                d1 = (math.log(spot / strike) + (rate + 0.5 * mid * mid) * t) / sqrt_t
                d2 = d1 - sqrt_t
                black = spot * _std_normal_cdf(d1) - strike * math.exp(-rate * t) * _std_normal_cdf(d2)
                if black < npv:
                    lo = mid
                else:
                    hi = mid
            assert abs(0.5 * (lo + hi) - expected_vol) < 1e-4, (
                f"{prefix} K={strike}: implied {0.5 * (lo + hi)} vs thesis {expected_vol}"
            )
