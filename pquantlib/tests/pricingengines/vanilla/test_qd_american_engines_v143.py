"""Cross-validate the QD+ / QD-FP American engines against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/qdfp`` (probe source
``migration-harness/cpp/probes/v143_pe_qdfp/probe.cpp``).

Covers the whole Andersen-Lake(-Offengenden) cluster --
:class:`QdPutCallParityEngine`, :class:`QdPlusBoundaryEvaluator`,
:class:`QdPlusAddOnValue`, :class:`QdPlusAmericanEngine`,
:class:`QdFpIterationScheme` and its three implementations, and
:class:`QdFpAmericanEngine` -- at four levels, deliberately, because the price
alone does not discriminate: the boundary at several ``tau`` (with the engine's
own evaluation count), the whole Chebyshev boundary curve, the add-on integrand
at several ``z``, and only then the NPV. A boundary that is wrong in the middle
of ``[0, T]`` still prices nearly right at one strike.

Every probe case carries its full market description, so the sweeps below
reconstruct each case rather than restating constants.

Tolerances
----------
TIGHT is used where the quantity is algebraic and the two implementations
evaluate the same closed form: :meth:`QdPlusAmericanEngine.x_max`, the
Chebyshev node positions, and the iteration schemes' integrator fingerprints.

LOOSE (1e-8 relative) is the tier for everything the engines *compute*: the
boundary is the root of a transcendental equation solved only to the engine's
own ``eps`` (1e-8 or 1e-10 in these cases), and the price is a quadrature of an
integrand built on top of that root. Agreement is therefore governed by those
tolerances, not by the last bits of the arithmetic. In practice the port does
much better -- 347 of the 363 pinned numbers land inside TIGHT and the worst
non-degenerate case is 1.3e-11 -- but LOOSE is what the algorithm entitles us
to claim.

One case needs more, and it is the tiny-volatility one; the derivation is in
:func:`test_tiny_volatility_is_quadrature_limited`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.interpolations.chebyshev_interpolation import ChebyshevInterpolation
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import (
    CashOrNothingPayoff,
    NullPayoff,
    OptionType,
    PlainVanillaPayoff,
)
from pquantlib.pricingengines.vanilla.qd_fp_american_engine import (
    QdFpAmericanEngine,
    QdFpIterationScheme,
    QdFpLegendreScheme,
    QdFpLegendreTanhSinhScheme,
    QdFpTanhSinhIterationScheme,
)
from pquantlib.pricingengines.vanilla.qd_plus_american_engine import (
    QdPlusAddOnValue,
    QdPlusAmericanEngine,
    QdPutCallParityEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

CPP: Final[dict[str, Any]] = reference_reader.load("v143/pe/qdfp")

# probe.cpp -- Settings::instance().evaluationDate() = Date(1, June, 2022)
TODAY: Final[Date] = Date.from_ymd(1, Month.June, 2022)

SOLVERS: Final[dict[str, QdPlusAmericanEngine.SolverType]] = {
    s.name: s for s in QdPlusAmericanEngine.SolverType
}
FP_EQUATIONS: Final[dict[str, QdFpAmericanEngine.FixedPointEquation]] = {
    e.name: e for e in QdFpAmericanEngine.FixedPointEquation
}

# The one case whose disagreement exceeds LOOSE; see the dedicated test.
TINY_VOL_CASE: Final[str] = "qdplus_tiny_vol_put_extremT_inside"

# Cases that are not a plain "build the engine, price the option" sweep.
_SPECIAL: Final[frozenset[str]] = frozenset(
    {"qdplus_cash_or_nothing_payoff_is_accepted", "greeks_not_provided", TINY_VOL_CASE}
)


def _names(prefix: str) -> list[str]:
    return sorted(k for k in CPP if k.startswith(prefix))


XMAX_CASES: Final[list[str]] = _names("xmax_")
BOUNDARY_CASES: Final[list[str]] = _names("boundary_")
CURVE_CASES: Final[list[str]] = _names("put_exercise_boundary_")
ADD_ON_CASES: Final[list[str]] = _names("add_on_value_")
SCHEME_CASES: Final[list[str]] = _names("scheme_")
ENGINE_CASES: Final[list[str]] = sorted(
    k
    for k, v in CPP.items()
    if "engine" in v["inputs"] and not k.startswith("throws_") and k not in _SPECIAL
)
THROW_CASES: Final[list[str]] = _names("throws_")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp main() -- Settings::instance().evaluationDate() = kToday,
    # kToday = Date(1, June, 2022).
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


# --- market reconstruction --------------------------------------------------


def _day_counter() -> Actual365Fixed:
    return Actual365Fixed()


def _process(
    spot: float, r: float, q: float, vol: float
) -> GeneralizedBlackScholesProcess:
    dc = _day_counter()
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=q, day_counter=dc
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=r, day_counter=dc
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=dc,
            volatility=vol,
        ),
    )


def _process_from(inputs: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    return _process(
        float(inputs["spot"]),
        float(inputs["r"]),
        float(inputs["q"]),
        float(inputs["volatility"]),
    )


def _american_option(inputs: dict[str, Any]) -> VanillaOption:
    option_type = OptionType.Call if inputs["option_type"] == "Call" else OptionType.Put
    maturity = TODAY + int(inputs["maturity_days"])
    return VanillaOption(
        PlainVanillaPayoff(option_type, float(inputs["strike"])),
        AmericanExercise(TODAY, maturity),
    )


def _scheme_from(inputs: dict[str, Any]) -> QdFpIterationScheme:
    kind = inputs["scheme_kind"]
    if kind == "legendre":
        return QdFpLegendreScheme(
            int(inputs["l"]), int(inputs["m"]), int(inputs["n"]), int(inputs["p"])
        )
    if kind == "legendre_tanh_sinh":
        return QdFpLegendreTanhSinhScheme(
            int(inputs["l"]), int(inputs["m"]), int(inputs["n"]), float(inputs["eps"])
        )
    if kind == "tanh_sinh":
        return QdFpTanhSinhIterationScheme(
            int(inputs["m"]), int(inputs["n"]), float(inputs["eps"])
        )
    if kind == "fast":
        return QdFpAmericanEngine.fast_scheme()
    if kind == "accurate":
        return QdFpAmericanEngine.accurate_scheme()
    if kind == "high_precision":
        return QdFpAmericanEngine.high_precision_scheme()
    raise AssertionError(f"unknown scheme kind {kind!r}")


def _engine_from(inputs: dict[str, Any]) -> QdPutCallParityEngine:
    process = _process_from(inputs)
    if inputs["engine"] == "QdPlusAmericanEngine":
        return QdPlusAmericanEngine(
            process,
            int(inputs["interpolation_points"]),
            SOLVERS[inputs["solver"]],
            float(inputs["eps"]),
        )
    return QdFpAmericanEngine(
        process, _scheme_from(inputs), FP_EQUATIONS[inputs["fp_equation"]]
    )


def _boundary_engine(inputs: dict[str, Any]) -> QdPlusAmericanEngine:
    # These cases drive putExerciseBoundaryAtTau with no market at all, which is
    # legal in C++ (the process is a possibly-null shared_ptr) and is how the
    # upstream test-suite exercises the boundary.
    return QdPlusAmericanEngine(
        None,
        int(inputs["interpolation_points"]),
        SOLVERS[inputs["solver"]],
        float(inputs["eps"]),
    )


def _boundary_curve(inputs: dict[str, Any]) -> ChebyshevInterpolation:
    return _boundary_engine(inputs).get_put_exercise_boundary(
        float(inputs["spot"]),
        float(inputs["strike"]),
        float(inputs["r"]),
        float(inputs["q"]),
        float(inputs["volatility"]),
        float(inputs["T"]),
    )


# --- QdPlusAmericanEngine::xMax --------------------------------------------


@pytest.mark.parametrize("name", XMAX_CASES)
def test_x_max(name: str) -> None:
    """All eight branches of Andersen/Lake (2021) Table 2, at TIGHT.

    ``xMax`` is a pure sign-case table over ``(r, q)`` returning ``K``,
    ``K r/q`` or ``0``; nothing is iterated, so anything looser than TIGHT
    would be hiding a wrong branch rather than accommodating arithmetic.
    """
    case = CPP[name]
    got = QdPlusAmericanEngine.x_max(
        float(case["inputs"]["strike"]),
        float(case["inputs"]["r"]),
        float(case["inputs"]["q"]),
    )
    tolerance.tight(got, float(case["expected"]["x_max"]))


# --- putExerciseBoundaryAtTau ----------------------------------------------


@pytest.mark.parametrize("name", BOUNDARY_CASES)
def test_put_exercise_boundary_at_tau(name: str) -> None:
    """Boundary value *and* the engine's own evaluation count, per solver.

    The evaluation count is the sharpest structural check available: it is the
    number of ``QdPlusBoundaryEvaluator.__call__`` invocations, so it pins the
    bracket-widening loop, the per-solver iteration and the Halley/Brent
    fallback all at once. It is asserted exactly for the three well-conditioned
    markets.

    Market ``m4`` (sigma = 1e-6 over 40 years) is the exception, and only by
    one: the residual at the returned root is at the 1e-16 level -- e.g. C++
    reports 4.4e-16 where this port reaches exactly 0.0 -- and Brent's
    ``close(f_root, 0)`` test therefore costs or saves exactly one extra
    iteration depending on which side of that the last bit lands. The value is
    unaffected. A tolerance of one evaluation is the honest statement; a wrong
    algorithm would be out by far more.
    """
    case = CPP[name]
    inputs, expected = case["inputs"], case["expected"]
    evaluations, boundary = _boundary_engine(inputs).put_exercise_boundary_at_tau(
        float(inputs["spot"]),
        float(inputs["strike"]),
        float(inputs["r"]),
        float(inputs["q"]),
        float(inputs["volatility"]),
        float(inputs["T"]),
        float(inputs["tau"]),
    )
    tolerance.loose(boundary, float(expected["boundary"]))

    slack = 1 if inputs["market"] == "m4" else 0
    assert abs(evaluations - int(expected["evaluations"])) <= slack, (
        f"{name}: {evaluations} evaluations, C++ reports {expected['evaluations']}"
    )


@pytest.mark.parametrize("name", CURVE_CASES)
def test_put_exercise_boundary_curve(name: str) -> None:
    """The whole Chebyshev boundary interpolation, nodes and values.

    ``nodes()`` is ``-cos(i pi / (n-1))``, evaluated identically on both sides,
    so it is asserted at TIGHT; the interpolated values ride on the solved
    boundary and get LOOSE.
    """
    case = CPP[name]
    inputs, expected = case["inputs"], case["expected"]
    interp = _boundary_curve(inputs)

    nodes = interp.nodes()
    assert len(nodes) == len(expected["nodes"])
    for i, z in enumerate(nodes):
        tolerance.tight(float(z), float(expected["nodes"][i]))
        tolerance.loose(
            interp(float(z), allow_extrapolation=True),
            float(expected["node_values"][i]),
        )

    for i, z in enumerate(inputs["z"]):
        tolerance.loose(
            interp(float(z), allow_extrapolation=True), float(expected["values"][i])
        )


@pytest.mark.parametrize("name", ADD_ON_CASES)
def test_add_on_value(name: str) -> None:
    """``QdPlusAddOnValue`` on top of the boundary the same engine produced.

    ``z = 0`` and ``z = 1e-8`` drive ``v = vol sqrt(z^2)`` below ``QL_EPSILON``
    and so exercise the three degenerate branches C++ keeps for that case,
    which the smooth two-``Phi`` expression would otherwise hide.
    """
    case = CPP[name]
    inputs, expected = case["inputs"], case["expected"]
    interp = _boundary_curve(inputs)
    add_on = QdPlusAddOnValue(
        float(inputs["T"]),
        float(inputs["spot"]),
        float(inputs["strike"]),
        float(inputs["r"]),
        float(inputs["q"]),
        float(inputs["volatility"]),
        float(inputs["x_max"]),
        interp,
    )
    for i, z in enumerate(inputs["z"]):
        tolerance.loose(add_on(float(z)), float(expected["values"][i]))


# --- iteration schemes ------------------------------------------------------


def _lorentz(x: float) -> float:
    return 1.0 / (1.0 + x * x)


def _integrator_fingerprints(integrator: Integrator) -> tuple[float, float, float]:
    """The three probe integrands, in the probe's order."""
    return (
        integrator(NormalDistribution(), -10.0, 10.0),
        integrator(_lorentz, -1.0, 1.0),
        integrator(math.sqrt, 0.0, 1.0),
    )


def _scheme_from_probe(inputs: dict[str, Any]) -> QdFpIterationScheme:
    kind = inputs["scheme"]
    if kind == "QdFpLegendreScheme":
        return QdFpLegendreScheme(
            int(inputs["l"]), int(inputs["m"]), int(inputs["n"]), int(inputs["p"])
        )
    if kind == "QdFpLegendreTanhSinhScheme":
        return QdFpLegendreTanhSinhScheme(
            int(inputs["l"]), int(inputs["m"]), int(inputs["n"]), float(inputs["eps"])
        )
    if kind == "QdFpTanhSinhIterationScheme":
        return QdFpTanhSinhIterationScheme(
            int(inputs["m"]), int(inputs["n"]), float(inputs["eps"])
        )
    if kind == "QdFpAmericanEngine::fastScheme":
        return QdFpAmericanEngine.fast_scheme()
    if kind == "QdFpAmericanEngine::accurateScheme":
        return QdFpAmericanEngine.accurate_scheme()
    if kind == "QdFpAmericanEngine::highPrecisionScheme":
        return QdFpAmericanEngine.high_precision_scheme()
    raise AssertionError(f"unknown scheme {kind!r}")


@pytest.mark.parametrize("name", SCHEME_CASES)
def test_iteration_scheme(name: str) -> None:
    """Node/step counts and both integrators' fingerprints, at TIGHT.

    The three integrands fingerprint the quadrature *order*: an ``l``-point
    Gauss-Legendre rule is exact for polynomials of degree < 2l and therefore
    lands on a specific wrong answer for ``1/(1+x^2)`` and ``sqrt(x)``, while a
    tanh-sinh rule gets both essentially exactly. Pinning them stops a port
    substituting a different order, or wiring one integrator where C++ builds
    two -- ``QdFpLegendreScheme`` uses order ``l`` inside the fixed-point step
    and order ``p`` for the boundary-to-price conversion, and those are
    different rules.

    TIGHT is right here because both sides evaluate a fixed deterministic rule
    on the same smooth integrands; the worst observed difference is 1.3e-15
    relative.
    """
    case = CPP[name]
    scheme = _scheme_from_probe(case["inputs"])
    expected = case["expected"]

    assert scheme.get_number_of_chebyshev_interpolation_nodes() == int(
        expected["chebyshev_nodes"]
    )
    assert scheme.get_number_of_naive_fixed_point_steps() == int(
        expected["naive_fixed_point_steps"]
    )
    assert scheme.get_number_of_jacobi_newton_fixed_point_steps() == int(
        expected["jacobi_newton_fixed_point_steps"]
    )

    fp_normal, fp_lorentz, fp_sqrt = _integrator_fingerprints(
        scheme.get_fixed_point_integrator()
    )
    tolerance.tight(fp_normal, float(expected["fp_normal_pdf_m10_10"]))
    tolerance.tight(fp_lorentz, float(expected["fp_lorentz_m1_1"]))
    tolerance.tight(fp_sqrt, float(expected["fp_sqrt_0_1"]))

    eb_normal, eb_lorentz, eb_sqrt = _integrator_fingerprints(
        scheme.get_exercise_boundary_to_price_integrator()
    )
    tolerance.tight(eb_normal, float(expected["eb_normal_pdf_m10_10"]))
    tolerance.tight(eb_lorentz, float(expected["eb_lorentz_m1_1"]))
    tolerance.tight(eb_sqrt, float(expected["eb_sqrt_0_1"]))


def test_legendre_tanh_sinh_returns_a_fresh_boundary_integrator() -> None:
    """``QdFpLegendreTanhSinhScheme`` builds a new tanh-sinh on every call.

    # C++ parity: qdfpamericanengine.cpp:108-116 -- the override constructs
    # ``ext::make_shared<TanhSinhIntegral>(eps_)`` each time rather than
    # caching, unlike the Legendre integrators the base holds. Caching it would
    # share the mutable evaluation counter between calls.
    """
    scheme = QdFpLegendreTanhSinhScheme(25, 5, 13, 1e-8)
    assert (
        scheme.get_exercise_boundary_to_price_integrator()
        is not scheme.get_exercise_boundary_to_price_integrator()
    )
    # ... while the fixed-point Legendre integrator is the cached one.
    assert scheme.get_fixed_point_integrator() is scheme.get_fixed_point_integrator()


def test_static_schemes_are_shared_instances() -> None:
    """# C++ parity: cpp:411-427 -- function-local ``static`` in all three."""
    assert QdFpAmericanEngine.fast_scheme() is QdFpAmericanEngine.fast_scheme()
    assert QdFpAmericanEngine.accurate_scheme() is QdFpAmericanEngine.accurate_scheme()
    assert (
        QdFpAmericanEngine.high_precision_scheme()
        is QdFpAmericanEngine.high_precision_scheme()
    )


# --- prices -----------------------------------------------------------------


@pytest.mark.parametrize("name", ENGINE_CASES)
def test_engine_npv(name: str) -> None:
    """Every priced probe case, through the instrument, at LOOSE.

    The sweep covers both engines over the shared degenerate-input battery
    (zero/tiny strike, zero spot with ``r`` of either sign, the ``r <= 0 and
    r <= q`` European short circuit, the zero-volatility intrinsic-maximum with
    its interior optimum inside and outside ``(0, T)``, and the ``r == q``
    branch), the five QD+ solvers, the ``interpolationPoints`` and ``eps``
    sweeps, all three QD-FP fixed-point equations, the three static schemes,
    the upstream Andersen-Lake high-precision grid, and the two double-boundary
    inputs that must raise.
    """
    case = CPP[name]
    inputs, expected = case["inputs"], case["expected"]
    option = _american_option(inputs)
    option.set_pricing_engine(_engine_from(inputs))

    if expected.get("throws"):
        with pytest.raises(LibraryException, match="double-boundary case"):
            option.npv()
    else:
        tolerance.loose(option.npv(), float(expected["npv"]))


def test_solver_types_agree_with_each_other() -> None:
    """The five QD+ solvers must reach the same price, not merely each their own.

    They are pinned individually against C++ by :func:`test_engine_npv`; this
    adds the cross-check that the *port's* five agree, which is what fails if a
    port quietly routes every ``SolverType`` to one implementation and then
    drifts. ``eps = 1e-10`` is the requested root accuracy and the boundary
    enters the price roughly linearly, so 1e-8 absolute on a ~23 NPV is a
    generous but non-vacuous band -- the solvers actually agree to 1.3e-13.
    """
    prices = [
        float(CPP[f"qdplus_standard_put_{s}"]["expected"]["npv"])
        for s in ("Brent", "Newton", "Ridder", "Halley", "SuperHalley")
    ]
    inputs = CPP["qdplus_standard_put_Halley"]["inputs"]
    got: list[float] = []
    for solver in SOLVERS.values():
        option = _american_option(inputs)
        option.set_pricing_engine(
            QdPlusAmericanEngine(
                _process_from(inputs),
                int(inputs["interpolation_points"]),
                solver,
                float(inputs["eps"]),
            )
        )
        got.append(option.npv())

    assert max(got) - min(got) < 1e-8
    for value in got:
        tolerance.loose(value, prices[0])


@pytest.mark.parametrize(
    "family",
    ["qdfp_standard_example", "qdfp_auto_picks_a", "qdfp_auto_picks_b"],
)
def test_auto_selects_the_documented_fixed_point_equation(family: str) -> None:
    """``Auto`` is ``|r - q| < 0.001 ? FP_A : FP_B`` -- pinned on both sides.

    ``qdfp_auto_picks_a`` has ``|r - q| = 5e-4`` and ``qdfp_auto_picks_b`` has
    ``2e-3``; the standard example has ``2.5e-2``. FP-A and FP-B differ in the
    5th significant figure here, so the equality below is a real discriminator
    rather than a tautology -- which the ``!=`` assertion makes explicit.
    """
    a = float(CPP[f"{family}_FP_A"]["expected"]["npv"])
    b = float(CPP[f"{family}_FP_B"]["expected"]["npv"])
    assert a != b

    inputs = CPP[f"{family}_Auto"]["inputs"]
    option = _american_option(inputs)
    option.set_pricing_engine(_engine_from(inputs))
    auto = option.npv()

    expected_auto = a if abs(float(inputs["r"]) - float(inputs["q"])) < 0.001 else b
    tolerance.loose(auto, expected_auto)
    tolerance.loose(auto, float(CPP[f"{family}_Auto"]["expected"]["npv"]))


def test_tiny_volatility_is_quadrature_limited() -> None:
    """sigma = 1e-6 over 40 years: agreement is capped by the quadrature.

    This is the only pinned number in the reference where the port and C++
    differ by more than LOOSE (observed: 1.60e-5 absolute on an NPV of
    ~1.0898e2, i.e. 1.46e-7 relative). The disagreement is localised, and
    localising it is why the probe pins market ``m4`` at three separate levels:

    * ``boundary_m4_*`` -- the solved boundary agrees to 5e-16 relative;
    * ``add_on_value_m4_n8`` -- the integrand agrees at TIGHT at every sampled
      ``z``;
    * only the *integral* of that integrand differs.

    The integrand is a near-step. ``v = vol sqrt(t)`` is about 6e-6, so
    ``dp = log(S dq / (b dr)) / v`` is amplified by 1e6 and ``Phi(-dp)`` climbs
    from 1e-301 to its plateau of 3.76 across ``z`` in [2.76, 3.05]. Neither
    implementation converges on it inside the 15 refinements QuantLib's
    ``TanhSinhIntegral`` defaults to: this port reports an absolute error
    estimate of 1.8e-7 and still moves by 1.6e-7 when allowed 20 refinements,
    and C++'s answer sits 1.6e-5 from the converged value -- boost stops
    earlier, so C++ is the *less* accurate of the two here.

    The bound is therefore taken from what upstream itself claims for exactly
    this parameter set: ``test-suite/americanoption.cpp`` lists
    ``{Option::Put, 100.0, 120.0, 4*3650, 1e-6, 0.01, 0.50,
    108.980920365700442, 1e-4}`` -- a precision of 1e-4, not 1e-8. The observed
    difference is a factor of 6 inside that, and a genuine porting error in
    this cluster moves the price by percent (FP-A vs FP-B differ in the 5th
    figure), so nothing is given up by using it.
    """
    case = CPP[TINY_VOL_CASE]
    inputs = case["inputs"]
    assert float(inputs["volatility"]) == 1e-6
    option = _american_option(inputs)
    option.set_pricing_engine(_engine_from(inputs))
    tolerance.custom(
        option.npv(),
        float(case["expected"]["npv"]),
        abs_tol=1e-4,
        rel_tol=0.0,
        reason=(
            "near-step add-on integrand (v = vol*sqrt(t) ~ 6e-6 amplifies dp by 1e6); "
            "neither boost's tanh-sinh nor this port converges within the 15-refinement "
            "budget. Bound is upstream's own precision for this exact parameter set "
            "(test-suite/americanoption.cpp, 'zero vol put 1': 1e-4)."
        ),
    )


# --- guards, and the results the engines do not fill ------------------------


@pytest.mark.parametrize("name", THROW_CASES)
def test_guards_raise(name: str) -> None:
    """Every ``QL_REQUIRE`` in ``QdPutCallParityEngine::calculate``.

    The non-striked-payoff guard is driven through the engine's own arguments
    because ``VanillaOption`` only accepts a ``StrikedTypePayoff`` -- which is
    also true in C++, where the probe has to reach for a bare
    ``OneAssetOption`` to get a ``NullPayoff`` past the constructor.
    """
    case = CPP[name]
    inputs = case["inputs"]
    assert case["expected"]["throws"] is True
    engine = (
        QdPlusAmericanEngine(_process_from(inputs))
        if inputs["engine"] == "QdPlusAmericanEngine"
        else QdFpAmericanEngine(_process_from(inputs))
    )
    maturity = TODAY + int(inputs["maturity_days"])

    if inputs.get("payoff") == "NullPayoff":
        arguments = engine.get_arguments()
        arguments.payoff = NullPayoff()
        arguments.exercise = AmericanExercise(TODAY, maturity)
        with pytest.raises(LibraryException, match="non-striked payoff given"):
            engine.calculate()
        return

    if inputs.get("exercise") == "European":
        option = VanillaOption(
            PlainVanillaPayoff(OptionType.Put, float(inputs["strike"])),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(engine)
        with pytest.raises(LibraryException, match="not an American option"):
            option.npv()
        return

    # Remaining guard: negative underlying.
    assert float(inputs["spot"]) < 0.0
    option = _american_option(inputs)
    option.set_pricing_engine(engine)
    with pytest.raises(LibraryException, match="negative underlying given"):
        option.npv()


def test_cash_or_nothing_payoff_is_accepted() -> None:
    """The guard is *striked*, not *plain vanilla*.

    ``QdPutCallParityEngine::calculate`` only ever reads ``strike()`` and
    ``optionType()`` off the payoff, so a ``CashOrNothingPayoff`` is accepted
    and priced exactly as the plain-vanilla payoff with the same strike -- the
    cash amount is ignored. Pinned so a port does not "helpfully" tighten the
    check to plain vanilla, which would turn a wrong-but-C++-faithful answer
    into an exception.
    """
    case = CPP["qdplus_cash_or_nothing_payoff_is_accepted"]
    inputs, expected = case["inputs"], case["expected"]
    maturity = TODAY + int(inputs["maturity_days"])

    cash = VanillaOption(
        CashOrNothingPayoff(
            OptionType.Put, float(inputs["strike"]), float(inputs["cash_payoff"])
        ),
        AmericanExercise(TODAY, maturity),
    )
    cash.set_pricing_engine(_engine_from(inputs))
    plain = _american_option(inputs)
    plain.set_pricing_engine(_engine_from(inputs))

    tolerance.loose(cash.npv(), float(expected["npv"]))
    tolerance.loose(plain.npv(), float(expected["plain_vanilla_npv"]))
    tolerance.exact(cash.npv(), plain.npv())


def test_no_greeks_are_provided() -> None:
    """``calculate()`` assigns ``results_.value`` and nothing else.

    ``GenericEngine.reset()`` nulls the whole Greeks block first, so every
    accessor must raise. Pinned explicitly because an engine that silently
    returned a stale or zero delta would still pass every price test.
    """
    case = CPP["greeks_not_provided"]
    inputs, expected = case["inputs"], case["expected"]
    option = _american_option(inputs)
    option.set_pricing_engine(
        QdFpAmericanEngine(_process_from(inputs), None, QdFpAmericanEngine.FixedPointEquation.Auto)
    )
    tolerance.loose(option.npv(), float(expected["npv"]))

    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert expected[f"{greek}_throws"] is True
        with pytest.raises(LibraryException, match=f"{greek} not provided"):
            getattr(option, greek)()
