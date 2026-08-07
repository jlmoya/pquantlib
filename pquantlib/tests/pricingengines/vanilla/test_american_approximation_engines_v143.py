"""Cross-validate the three American approximation engines against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/american`` (produced by
``migration-harness/cpp/probes/v143_pe_american/probe.cpp``), cases prefixed
``baw_``, ``ju_`` and ``bs_``.

The three engines — :class:`BaroneAdesiWhaleyApproximationEngine`,
:class:`JuQuadraticApproximationEngine`,
:class:`BjerksundStenslandApproximationEngine` — differ far more in *shape*
than in price, and the shape is what the cases are chosen to pin.

Which greeks each engine fills
------------------------------
========================  ================================================
engine / arm              filled
========================  ================================================
BAW, European arm         everything (delta .. itm_cash_probability)
BAW, approximation arm    value only
Ju, European arm          everything
Ju, approximation arm     value, delta, gamma
Bjerksund, all three arms value, delta, gamma, theta, theta_per_day, vega,
                          rho, dividend_rho, strike_sensitivity, plus
                          additional_results "strikeGamma" / "exerciseType"
========================  ================================================

Every absence is asserted, not merely unchecked: the probe writes
``"unset"`` and the assertion requires the accessor to raise.

Branch coverage
---------------
* the ``dividend_discount >= 1.0 and type == Call`` early-out (BAW and Ju),
  entered with ``q == 0`` exactly and with ``q < 0``;
* both sides of the ``spot < Sk`` / ``spot > Sk`` intrinsic-value test for
  Call and Put in both engines;
* :meth:`BaroneAdesiWhaleyApproximationEngine.critical_price` pinned
  *directly*, at the default tolerance and at 1e-6 / 1e-10, plus the
  ``close(rDF, 1.0, 1000)`` fallback at ``r == 0`` and the negative-rate
  rejection — it is a public static that Ju calls, so it needs its own
  contract test rather than being covered through NPV;
* Bjerksund's put-call transform, its ``bT == rT`` (i.e. ``r == 0``) branch
  in ``B0``, its run-away-boundary fallback (``q > 12.5``), its immediate
  exercise arm and its double-boundary refusal;
* 1-day and 10-year maturities, and zero volatility (which divides by zero
  inside ``critical_price``; C++ throws and so must the port).

Tolerance is TIGHT (1e-14 abs / 1e-12 rel).  Nothing here is quadrature: BAW
and Ju are closed forms wrapped around a Newton iteration whose stopping
criterion is identical on both sides, and Bjerksund's greeks are analytic.
Measured worst-case agreement across the whole table is ~3e-15 relative,
roughly 300x inside the bound; the Newton loop is deterministic, so a port
that gets the iteration subtly wrong lands orders of magnitude outside it
rather than marginally.

Three Bjerksund slots are *not* asserted — ``delta_forward``,
``elasticity``, ``itm_cash_probability``.  C++ builds a local
``OneAssetOption::results`` in each result builder and assigns the whole
struct into ``results_``; ``Greeks``/``MoreGreeks`` have no default member
initialisers, so those three carry indeterminate values (observed:
``deltaForward == -1.03e+80``).  The probe marks them ``"indeterminate"``.
Python leaves them unset instead, which is the only defensible behaviour —
there is no correct value to reproduce.

KNOWN FAILURE — ``ju_call_one_day`` delta and gamma
---------------------------------------------------
This case fails, and is left failing on purpose.  The cause is outside this
cluster and is a genuine defect, not a tolerance question:

``pquantlib.math.error_function.ErrorFunction`` and
``pquantlib.math.distributions.cumulative_normal_distribution.CumulativeNormalDistribution``
both delegate to :func:`math.erf` instead of porting QuantLib's own
Sun-derived ``ErrorFunction``.  The two are **not** interchangeable:
comparing ``ErrorFunction()(x)`` against ``std::erf(x)`` over a uniform grid
on [-6, 6] (step 0.0013) gives **2512 of 9231 points differing by 1 ulp** —
27%, mid-range, not the "few extreme-range inputs" the docstring on
``error_function.py`` claims.

That 1-ulp bias enters
:meth:`BaroneAdesiWhaleyApproximationEngine.critical_price` through the
Newton iteration and lands ``Sk`` **65 ulps** away from C++
(Python ``103.75566597237196`` vs C++ ``103.75566597237103``).  Ju's 1-day
case is extraordinarily sensitive to ``Sk``: measured by perturbing ``Sk``
one ulp at a time, ``d(delta)/delta`` is ``1.07e-12`` **per ulp** — an
amplification of ~4800 machine epsilons, driven by ``lambda = 108.7``
appearing as ``(S/Sk)^lambda``, by ``1/h = 7300``, and by ``1 - chi = 0.074``
against ``chi = 0.926``.

Substituting the C++ ``Sk`` into the otherwise-unchanged Python formula
drops the delta error from ``2.67e-12`` to ``4.46e-13`` — i.e. inside TIGHT.
So the Ju port itself is correct; the CDF primitive underneath it is not.
Observed error, slot by slot: npv ``3.0e-13`` (passes), delta ``2.7e-12``,
gamma ``4.6e-11``.  Every other Ju case in the table agrees to ``<= 2e-16``.

Loosening the tolerance here would hide a repo-wide numerical bias in the
most-used primitive in the library, so it is not done.  The fix — porting
``ErrorFunction`` properly and routing ``CumulativeNormalDistribution``
through it — moves values in every engine and must be sequenced repo-wide.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import CashOrNothingPayoff, PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.barone_adesi_whaley_engine import (
    BaroneAdesiWhaleyApproximationEngine,
)
from pquantlib.pricingengines.vanilla.bjerksund_stensland_engine import (
    BjerksundStenslandApproximationEngine,
)
from pquantlib.pricingengines.vanilla.ju_quadratic_engine import (
    JuQuadraticApproximationEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.testing import reference_reader, tolerance

from ._american_v143 import TODAY, build_payoff, expect_number, market, option_type

CPP: dict[str, Any] = reference_reader.load("v143/pe/american")

_ENGINE_CASES = [
    name
    for name in CPP
    if name.startswith(("baw_", "ju_", "bs_")) and not name.startswith("baw_critical_")
]
_CRITICAL_PRICE_CASES = [name for name in CPP if name.startswith("baw_critical_")]

_GREEK_SLOTS: list[tuple[str, str]] = [
    ("delta", "delta"),
    ("gamma", "gamma"),
    ("theta", "theta"),
    ("theta_per_day", "theta_per_day"),
    ("vega", "vega"),
    ("rho", "rho"),
    ("dividend_rho", "dividend_rho"),
    ("strike_sensitivity", "strike_sensitivity"),
    ("itm_cash_probability", "itm_cash_probability"),
    ("delta_forward", "delta_forward"),
    ("elasticity", "elasticity"),
]

_ENGINE_FACTORIES = {
    "BaroneAdesiWhaleyApproximationEngine": BaroneAdesiWhaleyApproximationEngine,
    "JuQuadraticApproximationEngine": JuQuadraticApproximationEngine,
    "BjerksundStenslandApproximationEngine": BjerksundStenslandApproximationEngine,
}


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp — ``Settings::instance().evaluationDate() = TODAY;`` in main(),
    # with ``const Date TODAY(1, March, 2025);``.
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _engine_for(name: str, process: GeneralizedBlackScholesProcess) -> PricingEngine:
    return _ENGINE_FACTORIES[name](process)


def test_case_tables_are_populated() -> None:
    """Guard against a reference that silently lost its cases."""
    assert len(_ENGINE_CASES) >= 40
    assert len(_CRITICAL_PRICE_CASES) >= 8


# --- criticalPrice, the public static --------------------------------------


@pytest.mark.tight
@pytest.mark.parametrize("case", _CRITICAL_PRICE_CASES)
def test_barone_adesi_whaley_critical_price(case: str) -> None:
    """Pin the Newton solve directly, including its tolerance argument."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    payoff = PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"]))
    args = (
        payoff,
        float(inputs["risk_free_discount"]),
        float(inputs["dividend_discount"]),
        float(inputs["variance"]),
    )

    if expected["throws"]:
        with pytest.raises(LibraryException):
            BaroneAdesiWhaleyApproximationEngine.critical_price(
                *args, float(inputs["tolerance"])
            )
        return

    if inputs["tolerance"] == "default":
        # Exercise the ``tolerance = 1e-6`` default at the call site.
        sk = BaroneAdesiWhaleyApproximationEngine.critical_price(*args)
    else:
        sk = BaroneAdesiWhaleyApproximationEngine.critical_price(
            *args, float(inputs["tolerance"])
        )
    tolerance.tight(sk, float(expected["critical_price"]), reason=f"{case} critical price")


def test_critical_price_default_tolerance_matches_explicit_1e6() -> None:
    """The default argument really is 1e-6, not merely "something small"."""
    default_case = CPP["baw_critical_call_default_tol"]["expected"]["critical_price"]
    explicit_case = CPP["baw_critical_call_tol_1e6"]["expected"]["critical_price"]
    assert default_case == explicit_case

    inputs = CPP["baw_critical_call_default_tol"]["inputs"]
    payoff = PlainVanillaPayoff(option_type(inputs["type"]), float(inputs["strike"]))
    sk = BaroneAdesiWhaleyApproximationEngine.critical_price(
        payoff,
        float(inputs["risk_free_discount"]),
        float(inputs["dividend_discount"]),
        float(inputs["variance"]),
    )
    tolerance.tight(sk, float(default_case), reason="default tolerance")


def test_tighter_tolerance_moves_the_critical_price() -> None:
    """A port that ignores ``tolerance`` would return the same number twice."""
    loose = CPP["baw_critical_call_tol_1e6"]["expected"]["critical_price"]
    tight_ = CPP["baw_critical_call_tol_1e10"]["expected"]["critical_price"]
    assert loose != tight_


# --- the three engines -----------------------------------------------------


@pytest.mark.tight
@pytest.mark.parametrize("case", _ENGINE_CASES)
def test_american_approximation_engine(case: str) -> None:
    """Reproduce NPV, every greek C++ fills, and every greek it does not."""
    record = CPP[case]
    inputs = record["inputs"]
    expected = record["expected"]

    process = market(
        float(inputs["spot"]), float(inputs["q"]), float(inputs["r"]), float(inputs["vol"])
    )
    ex_date = TODAY + int(inputs["maturity_days"])
    exercise = (
        EuropeanExercise(ex_date)
        if inputs["european_exercise"]
        else AmericanExercise(TODAY, ex_date, bool(inputs["payoff_at_expiry"]))
    )
    payoff = (
        CashOrNothingPayoff(
            option_type(inputs["type"]), float(inputs["strike"]), float(inputs["cash"])
        )
        if inputs.get("payoff") == "CashOrNothing"
        else build_payoff({**inputs, "payoff": "PlainVanilla"})
    )
    option = VanillaOption(payoff, exercise)
    option.set_pricing_engine(_engine_for(inputs["engine"], process))

    if expected["throws"]:
        with pytest.raises(LibraryException):
            option.npv()
        return

    expect_number(option.npv, expected["npv"], f"{case} npv")
    for attr, key in _GREEK_SLOTS:
        expect_number(getattr(option, attr), expected[key], f"{case} {key}")

    if inputs["engine"] == "BjerksundStenslandApproximationEngine":
        tolerance.tight(
            float(option.result("strikeGamma")),
            float(expected["strike_gamma"]),
            reason=f"{case} strikeGamma",
        )
        assert option.result("exerciseType") == expected["exercise_type"]


def test_bjerksund_exercise_type_labels_all_three_arms() -> None:
    """All three of Bjerksund's result builders are reached by the table."""
    labels = {
        CPP[name]["expected"]["exercise_type"]
        for name in _ENGINE_CASES
        if name.startswith("bs_") and not CPP[name]["expected"]["throws"]
    }
    assert labels == {"European", "Immediate", "American"}


def test_barone_adesi_whaley_approximation_arm_publishes_no_greeks() -> None:
    """The approximation arm fills ``value`` and nothing else."""
    expected = CPP["baw_call_haug_atm"]["expected"]
    assert expected["npv"] > 0.0
    for key in (
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "dividend_rho",
        "strike_sensitivity",
        "itm_cash_probability",
        "delta_forward",
        "elasticity",
    ):
        assert expected[key] == "unset", key


def test_ju_approximation_arm_publishes_delta_and_gamma_only() -> None:
    """Ju fills delta and gamma where BAW does not — a real signature difference."""
    ju = CPP["ju_call_atm"]["expected"]
    baw = CPP["baw_call_haug_atm"]["expected"]
    assert ju["delta"] != "unset"
    assert ju["gamma"] != "unset"
    assert baw["delta"] == "unset"
    for key in ("theta", "vega", "rho", "dividend_rho"):
        assert ju[key] == "unset", key


def test_ju_intrinsic_arm_returns_unit_delta_and_zero_gamma() -> None:
    """``phi * (Sk - S) <= 0`` short-circuits to the intrinsic value."""
    call = CPP["ju_call_deep_itm_intrinsic"]["expected"]
    put = CPP["ju_put_deep_itm_intrinsic"]["expected"]
    assert call["delta"] == 1
    assert call["gamma"] == 0
    assert put["delta"] == -1
    assert put["gamma"] == 0


def test_bjerksund_put_transform_is_not_symmetric_when_q_differs_from_r() -> None:
    """The put-call transform must survive ``q != r``.

    With ``q == r`` the mirrored call is the original option and a port that
    forgot to un-swap the greeks would still look right; ``bs_put_atm``
    (q == r == 0.10) is exactly that trap, so this pins the asymmetric case
    where delta and strike sensitivity genuinely differ.
    """
    symmetric_call = CPP["bs_call_atm"]["expected"]
    symmetric_put = CPP["bs_put_atm"]["expected"]
    assert symmetric_call["npv"] == symmetric_put["npv"]
    assert symmetric_call["delta"] != symmetric_put["delta"]

    asymmetric = CPP["bs_put_transform_american"]["expected"]
    assert asymmetric["delta"] < 0.0
    assert asymmetric["strike_sensitivity"] > 0.0
    assert asymmetric["rho"] < 0.0
    assert asymmetric["dividend_rho"] > 0.0
