"""Cross-validate AnalyticBlackVasicekEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``vasicek_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

What these tests are actually defending
---------------------------------------
1. **That ``correlation`` moves the price.** ``vasicek_rho_*`` prices one market
   at five correlations, calls and puts.
2. **That the equity vol is read at ``t = 0``, not at maturity.** C++ line 79 is
   ``blackVol(t, K)`` with ``t`` still 0. ``vasicek_vol_read_at_time_zero_{1y,5y}``
   uses a variance curve whose 1y and 5y vols are 0.15 and 0.35, so a port that
   sampled at maturity is off by tens of percent.
3. **That only ``results.value`` is filled** — no Greeks.
4. **The degenerate limit**: with ``sigma_r -> 0`` and ``b = r0``, the Vasicek
   zero-coupon bond is exactly ``exp(-r0 T)``, so the engine reduces to
   Black-Scholes with zero dividend and must reproduce
   ``AnalyticEuropeanEngine``.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel). The only quadrature is a
``SimpsonIntegral(1e-5, 1000)`` over a smooth quadratic-in-``g`` integrand, and
the port reproduces C++'s Romberg refinement step for step — same node counts,
same ``i > 5`` guard — so the two integrators stop on the same iteration with the
same partial sums rather than merely converging to the same answer. Measured over
all 19 pinned values the worst relative deviation is 7.2e-15.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_vasicek_engine import (
    AnalyticBlackVasicekEngine,
)
from pquantlib.testing import reference_reader, tolerance

from ._hybrid_v143 import (
    REFERENCE_KEY,
    TODAY,
    bsm_process,
    flat_vol,
    maturity_date,
    option_type,
    term_vol,
)

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)

VASICEK_NPV_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("vasicek_") and "npv" in case["expected"]
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


def _vasicek(inputs: dict[str, Any]) -> Vasicek:
    return Vasicek(
        r0=inputs["vasicekR0"],
        a=inputs["vasicekA"],
        b=inputs["vasicekB"],
        sigma=inputs["vasicekSigma"],
        lambda_=inputs["vasicekLambda"],
    )


def _build(inputs: dict[str, Any]) -> VanillaOption:
    use_term_vol = inputs.get("volTermStructure") == "BlackVarianceCurve"
    process = bsm_process(inputs, vol_ts=term_vol() if use_term_vol else None)
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(
        AnalyticBlackVasicekEngine(process, _vasicek(inputs), inputs["correlation"])
    )
    return option


@pytest.mark.parametrize("case_name", VASICEK_NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """NPV reproduces C++ for every pinned market."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


@pytest.mark.parametrize("option_kind", ["call", "put"])
def test_correlation_changes_the_price(option_kind: str) -> None:
    """The correlation sweep must be monotone, not five copies of one number.

    ``upsilon`` picks up ``2 rho sigma_s sigma_r g(T-u)`` with ``g > 0``, so the
    total variance rises with rho; a call's value rises with it, and so does a
    put's (the bond price and forward are untouched by rho).
    """
    prefix = "vasicek_rho_" if option_kind == "call" else "vasicek_put_rho_"
    sweep = sorted(
        (CPP[name]["inputs"]["correlation"], name)
        for name in VASICEK_NPV_CASES
        if name.startswith(prefix)
    )
    assert len(sweep) == 5
    prices = [_build(CPP[name]["inputs"]).npv() for _, name in sweep]
    assert len(set(prices)) == len(prices), "the correlation is being discarded"
    assert prices == sorted(prices)
    for (_, name), got in zip(sweep, prices, strict=True):
        tolerance.tight(got, CPP[name]["expected"]["npv"], reason=name)


def test_vol_is_sampled_at_time_zero_not_at_maturity() -> None:
    """# C++ parity: analyticeuropeanvasicekengine.cpp:79 — ``blackVol(t, K)``, t == 0.

    Both cases share one ``BlackVarianceCurve`` (1y vol 0.15, 5y vol 0.35) and
    differ only in maturity. C++ uses the curve's short end for both — via
    ``BlackVarianceTermStructure::blackVolImpl``'s ``t == 0 -> 1e-5``
    substitution — so the 5y price is the 5y one computed with a ~0.15 equity
    vol, not a 0.35 one. A port that read the vol at maturity produces a
    materially larger 5y number and still passes every flat-vol case.
    """
    one_year = CPP["vasicek_vol_read_at_time_zero_1y"]
    five_year = CPP["vasicek_vol_read_at_time_zero_5y"]
    tolerance.tight(_build(one_year["inputs"]).npv(), one_year["expected"]["npv"])
    tolerance.tight(_build(five_year["inputs"]).npv(), five_year["expected"]["npv"])

    # Independent evidence that the short end is what was used: pricing the 5y
    # case against a *flat* 0.15 surface moves the answer far less than against a
    # flat 0.35 one.
    inputs = dict(five_year["inputs"])
    short_end = bsm_process(inputs, vol_ts=flat_vol(0.15))
    long_end = bsm_process(inputs, vol_ts=flat_vol(0.35))
    prices: list[float] = []
    for process in (short_end, long_end):
        option = VanillaOption(
            PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
            EuropeanExercise(maturity_date(inputs)),
        )
        option.set_pricing_engine(
            AnalyticBlackVasicekEngine(process, _vasicek(inputs), inputs["correlation"])
        )
        prices.append(option.npv())
    actual = five_year["expected"]["npv"]
    assert abs(actual - prices[0]) < abs(actual - prices[1])


def test_zero_rate_vol_collapses_to_black_scholes() -> None:
    """``sigma_r -> 0`` with ``b == r0``: the Vasicek bond is exactly ``exp(-r0 T)``.

    The engine then reduces to Black-Scholes with zero dividend yield, so
    ``AnalyticEuropeanEngine`` on the same process must agree. Correlation is 0.9,
    so a port that dropped the ``sigma_r`` factor from the cross term fails here.
    """
    case = CPP["vasicek_zero_rate_vol_matches_bs"]
    inputs, expected = case["inputs"], case["expected"]
    hybrid = _build(inputs)
    plain = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    plain.set_pricing_engine(AnalyticEuropeanEngine(bsm_process(inputs)))

    tolerance.tight(hybrid.npv(), expected["npv"])
    tolerance.tight(plain.npv(), expected["analytic_european_npv"])
    # C++'s own two numbers differ by 1.3e-10 (the Simpson integral is only
    # requested to 1e-5 absolute, and sigma_r = 1e-12 rather than 0). Require the
    # Python pair to be at least as close as the C++ pair, with a decade of slack.
    cpp_gap = abs(expected["npv"] - expected["analytic_european_npv"])
    assert abs(hybrid.npv() - plain.npv()) <= max(cpp_gap * 10.0, 1e-9)


def test_no_greeks_provided() -> None:
    """# C++ parity: calculate() assigns only ``results_.value``."""
    expected = CPP["vasicek_no_greeks"]["expected"]
    assert expected == {"delta_throws": True, "gamma_throws": True, "vega_throws": True}
    option = _build(CPP["vasicek_rho_000"]["inputs"])
    option.npv()
    with pytest.raises(LibraryException, match="delta not provided"):
        option.delta()
    with pytest.raises(LibraryException, match="gamma not provided"):
        option.gamma()
    with pytest.raises(LibraryException, match="vega not provided"):
        option.vega()


def test_american_exercise_throws() -> None:
    """# C++ parity: analyticeuropeanvasicekengine.cpp:62-63."""
    assert CPP["vasicek_american_exercise_throws"]["expected"]["throws"] is True
    inputs = CPP["vasicek_rho_000"]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(
        AnalyticBlackVasicekEngine(
            bsm_process(inputs), _vasicek(inputs), inputs["correlation"]
        )
    )
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()


def test_engine_inspector() -> None:
    """``correlation()`` returns what the constructor was handed."""
    inputs = CPP["vasicek_rho_050"]["inputs"]
    engine = AnalyticBlackVasicekEngine(
        bsm_process(inputs), _vasicek(inputs), inputs["correlation"]
    )
    assert math.isclose(engine.correlation(), inputs["correlation"])
