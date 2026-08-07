"""Cross-validate AnalyticGJRGARCHEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/hybrid`` (``gjrgarch_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_hybrid/probe.cpp``.

C++ parity: ql/pricingengines/vanilla/analyticgjrgarchengine.{hpp,cpp} @ v1.43 —
the Edgeworth expansion of Duan, Gauthier, Simonato & Sasseville (2006).

This file replaces the earlier one, which loaded ``cluster/w1d`` and pinned two
values (an ATM 1y call and put) at the v1.42.1 pin. Both are still covered —
``gjrgarch_atm_1y_{call,put}`` reproduces that market bit-for-bit, which is how
the migration was checked — but the coverage is now the whole 72-point Duan grid:
three risk-premium parameters ``lambda`` x two maturities x six strikes x both
option types. That matters because the Edgeworth correction is
``k3 * A3 + (k4 - 3) * A4``, and the skewness ``k3`` is what ``lambda`` controls;
a single ATM point barely feels it.

The earlier file also had no evaluation-date fixture. The curves carried their
own reference dates so the *arithmetic* was pinned, but ``Option.is_expired``
reads the global evaluation date, so the tests would have started failing on
16 June 2027. Fixed here.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel). The engine is closed form, but it accumulates a
triple loop over ``T`` days (up to 252 here) of mixed products and quotients, so
the round-off is genuinely larger than a one-line formula's: measured over the 74
pinned values the worst relative deviation is 3.3e-12, on a deep-OTM put whose
NPV is small enough that TIGHT's 1e-14 absolute leg is what carries it. That is
the honest reading of ``math.isclose(rel_tol=1e-12, abs_tol=1e-14)`` — the
absolute leg is not a fudge, it is the right criterion for a value near zero —
and no case needs LOOSE.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.equity.gjr_garch_model import GJRGARCHModel
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_gjr_garch_engine import (
    AnalyticGJRGARCHEngine,
)
from pquantlib.processes.gjr_garch_process import GJRGARCHProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance

from ._hybrid_v143 import REFERENCE_KEY, TODAY, flat_curve, maturity_date, option_type

CPP: dict[str, Any] = reference_reader.load(REFERENCE_KEY)

GJR_NPV_CASES = sorted(
    name
    for name, case in CPP.items()
    if name.startswith("gjrgarch_") and "npv" in case["expected"]
)

# The Duan et al. (2006) reference table, reproduced in the v1.43 test-suite as
# ``analytic[lambda][maturity][strike]`` in gjrgarchmodel.cpp. Call prices.
DUAN_TABLE: dict[tuple[int, int], list[float]] = {
    (0, 90): [15.4315, 10.5552, 5.9625, 2.3282, 0.5408, 0.0835],
    (0, 180): [15.8969, 11.2173, 6.9112, 3.4788, 1.3769, 0.4357],
    (10, 90): [15.4556, 10.6929, 6.2381, 2.6831, 0.7822, 0.1738],
    (10, 180): [16.0587, 11.5338, 7.3170, 3.9074, 1.7279, 0.6568],
    (20, 90): [15.8000, 11.2734, 7.0376, 3.6767, 1.5871, 0.5934],
    (20, 180): [16.9286, 12.3170, 8.0405, 4.6348, 2.3429, 1.0590],
}
DUAN_STRIKES = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0]


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:1365 — Settings::instance().evaluationDate() = kToday,
    # kToday = Date(1, March, 2025) at probe.cpp:225.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _model(inputs: dict[str, Any]) -> GJRGARCHModel:
    return GJRGARCHModel(
        GJRGARCHProcess(
            risk_free_rate=flat_curve(inputs["r"]),
            dividend_yield=flat_curve(inputs["q"]),
            s0=SimpleQuote(inputs["spot"]),
            v0=inputs["v0"],
            omega=inputs["omega"],
            alpha=inputs["alpha"],
            beta=inputs["beta"],
            gamma=inputs["gamma"],
            lambda_=inputs["lambda"],
            days_per_year=inputs["daysPerYear"],
        )
    )


def _build(inputs: dict[str, Any]) -> VanillaOption:
    option = VanillaOption(
        PlainVanillaPayoff(option_type(inputs["optionType"]), inputs["strike"]),
        EuropeanExercise(maturity_date(inputs)),
    )
    option.set_pricing_engine(AnalyticGJRGARCHEngine(_model(inputs)))
    return option


@pytest.mark.parametrize("case_name", GJR_NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """NPV reproduces C++ across the whole Duan grid."""
    case = CPP[case_name]
    tolerance.tight(_build(case["inputs"]).npv(), case["expected"]["npv"])


def test_grid_covers_every_lambda_maturity_and_strike() -> None:
    """Guard the guard: the parametrization really is the full 3x2x6x2 grid."""
    grid = {
        (
            CPP[name]["inputs"]["lambda"],
            CPP[name]["inputs"]["maturityDays"],
            CPP[name]["inputs"]["strike"],
            CPP[name]["inputs"]["optionType"],
        )
        for name in GJR_NPV_CASES
        if CPP[name]["inputs"]["daysPerYear"] == 365.0
    }
    assert len(grid) == 3 * 2 * 6 * 2


def test_lambda_changes_the_price() -> None:
    """The risk-premium parameter must move the price at every strike.

    ``lambda`` enters every recursion constant (``m1``, ``m2``, ``v1``, ``z1``,
    ``x1``, ...) through ``N(lambda)`` and ``phi(lambda)``, and thence the
    skewness ``k3`` that drives the ``A3`` correction. Three values, six strikes:
    eighteen distinct prices.
    """
    for strike in DUAN_STRIKES:
        prices = [
            _build(CPP[name]["inputs"]).npv()
            for name in GJR_NPV_CASES
            if CPP[name]["inputs"].get("daysPerYear") == 365.0
            and CPP[name]["inputs"]["maturityDays"] == 90
            and CPP[name]["inputs"]["strike"] == strike
            and CPP[name]["inputs"]["optionType"] == "Call"
        ]
        assert len(prices) == 3
        assert len(set(prices)) == 3, f"lambda is being discarded at K={strike}"


@pytest.mark.parametrize(("lambda_pct", "days"), sorted(DUAN_TABLE))
def test_reproduces_the_duan_2006_table(lambda_pct: int, days: int) -> None:
    """Independent evidence: the published Edgeworth prices come out.

    Not a C++ comparison — that is asserted above to 1e-12. This checks the
    *model* is the one Duan et al. published, which matching a probe alone would
    not establish. The band is 7.5e-2, the upstream C++ test's own tolerance for
    this comparison (gjrgarchmodel.cpp uses ``2.0 * tolerance`` with
    ``tolerance = 7.5e-2``); the table itself is quoted to four decimals.
    """
    expected = DUAN_TABLE[lambda_pct, days]
    for strike, want in zip(DUAN_STRIKES, expected, strict=True):
        name = f"gjrgarch_lam{lambda_pct}_d{days}_k{int(strike)}_call"
        got = _build(CPP[name]["inputs"]).npv()
        assert abs(got - want) < 2.0 * 7.5e-2, f"{name}: {got} vs Duan {want}"


def test_put_call_parity_is_the_engines_own_identity() -> None:
    """The put branch is literally ``C + K Df_rf / Df_div - S``.

    # C++ parity: analyticgjrgarchengine.cpp:288-290. Worth pinning separately
    # from the NPVs because it is the only place the engine touches the dividend
    # discount after the drift calculation.
    """
    call_inputs = CPP["gjrgarch_atm_1y_call"]["inputs"]
    put_inputs = CPP["gjrgarch_atm_1y_put"]["inputs"]
    call = _build(call_inputs).npv()
    put = _build(put_inputs).npv()
    tolerance.tight(call, CPP["gjrgarch_atm_1y_call"]["expected"]["npv"])
    tolerance.tight(put, CPP["gjrgarch_atm_1y_put"]["expected"]["npv"])

    process = _model(call_inputs).process()
    expiry = maturity_date(call_inputs)
    df_rf = process.risk_free_rate().discount(expiry)
    df_div = process.dividend_yield().discount(expiry)
    strike = call_inputs["strike"]
    spot = call_inputs["spot"]
    tolerance.tight(call + strike * df_rf / df_div - spot, put)


def test_migrated_reference_matches_the_retired_cluster_w1d_values() -> None:
    """The two values this file used to assert survive the reference migration.

    ``cluster/w1d`` pinned the ATM 1y call at 9.95764608376202 and the put at
    5.0805885338334065 under the phase-11 (v1.42.1) probe. The v1.43 probe
    re-anchors the same flat-curve, 365-day market onto 1 March 2025 and produces
    the same two doubles, so the migration is a re-labelling and not a silent
    re-baseline. Asserting the literals here is what makes that checkable.
    """
    tolerance.exact(CPP["gjrgarch_atm_1y_call"]["expected"]["npv"], 9.95764608376202)
    tolerance.exact(
        CPP["gjrgarch_atm_1y_put"]["expected"]["npv"], 5.0805885338334065
    )
    tolerance.tight(_build(CPP["gjrgarch_atm_1y_call"]["inputs"]).npv(), 9.95764608376202)
    tolerance.tight(
        _build(CPP["gjrgarch_atm_1y_put"]["inputs"]).npv(), 5.0805885338334065
    )


def test_american_exercise_throws() -> None:
    """# C++ parity: analyticgjrgarchengine.cpp:46-47."""
    assert CPP["gjrgarch_american_exercise_throws"]["expected"]["throws"] is True
    inputs = CPP["gjrgarch_atm_1y_call"]["inputs"]
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, inputs["strike"]),
        AmericanExercise(TODAY, TODAY + 90),
    )
    option.set_pricing_engine(AnalyticGJRGARCHEngine(_model(inputs)))
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()


def test_zero_spot_throws() -> None:
    """# C++ parity: analyticgjrgarchengine.cpp:59."""
    case = CPP["gjrgarch_zero_spot_throws"]
    assert case["expected"]["throws"] is True
    inputs = dict(CPP["gjrgarch_atm_1y_call"]["inputs"])
    inputs["spot"] = 0.0
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 50.0), EuropeanExercise(TODAY + 90)
    )
    option.set_pricing_engine(AnalyticGJRGARCHEngine(_model(inputs)))
    with pytest.raises(LibraryException, match="negative or null underlying given"):
        option.npv()


def test_no_greeks_provided() -> None:
    """# C++ parity: calculate() assigns only ``results_.value``."""
    expected = CPP["gjrgarch_no_greeks"]["expected"]
    assert expected == {"delta_throws": True, "vega_throws": True}
    option = _build(CPP["gjrgarch_atm_1y_call"]["inputs"])
    option.npv()
    with pytest.raises(LibraryException, match="delta not provided"):
        option.delta()
    with pytest.raises(LibraryException, match="vega not provided"):
        option.vega()


def test_engine_takes_no_integration_knob() -> None:
    """The C++ constructor has exactly one argument.

    # C++ parity: ``AnalyticGJRGARCHEngine(const ext::shared_ptr<GJRGARCHModel>&)``
    # — analyticgjrgarchengine.hpp:59. The Python port used to accept an
    # ``integration_order`` kwarg "for API parity with sibling engines" and
    # discard it; the engine is closed form and never integrates anything, so
    # there was nothing for it to parity with. Removed, and asserted removed.
    """
    model = _model(CPP["gjrgarch_atm_1y_call"]["inputs"])
    engine = AnalyticGJRGARCHEngine(model)
    assert engine.model() is model
    with pytest.raises(TypeError):
        AnalyticGJRGARCHEngine(model, 144)  # pyright: ignore[reportCallIssue]
