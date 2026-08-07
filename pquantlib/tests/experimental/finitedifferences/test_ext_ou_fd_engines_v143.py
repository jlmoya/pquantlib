"""Cross-validate the extended-OU / Kluge FD engines against C++ v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_extoufd/probe.cpp
Reference:    migration-harness/references/v143/experimental/extoufd.json

C++ classes:
    ``FdExtOUJumpVanillaEngine``
        ql/experimental/finitedifferences/fdextoujumpvanillaengine.hpp:38
    ``FdKlugeExtOUSpreadEngine``
        ql/experimental/finitedifferences/fdklugeextouspreadengine.hpp:41

Both were absent from the port. The two solvers they need,
``FdmExtOUJumpSolver`` and ``FdmKlugeExtOUSolver``, existed but raised
``NotImplementedError`` behind a carve-out note saying the multi-D backward
FDM framework was "deferred to a follow-up Phase 11 cluster"; that framework
(``Fdm2DimSolver`` / ``FdmNdimSolver`` / the Hundsdorfer family) had in fact
already shipped, so the note was stale rather than true.

Section A first
---------------
Neither engine touches the payoff grid directly: ``FdmSimpleProcess1dMesher``
sizes direction 0 by calling ``evolve``, hence
``ExtendedOrnsteinUhlenbeckProcess.expectation_1d`` and
``.std_deviation_1d``. A wrong expectation produces a different MESH and then
every NPV disagrees for a reason unrelated to the engine. Section A pins the
moments directly, for all three ``Discretization`` modes, against a
non-constant ``b(t)`` — with constant ``b`` the three modes coincide, which is
exactly how a port ships the wrong one and passes.

Evaluation date
---------------
The probe sets ``Settings::instance().evaluationDate() = Date(15, January,
2024)`` in ``main`` (probe.cpp, first statement). Pinned and restored here.

Tolerance
---------
Process moments: TIGHT — closed forms, plus one adaptive Gauss-Lobatto
integral whose target accuracy (1e-4) both sides request identically and both
overshoot by orders of magnitude.

Engine NPVs: LOOSE. An FD rollback accumulates thousands of operator applies
and tridiagonal solves; the operations are the same but the summation order
inside numpy's vectorised kernels is not C++'s.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import BermudanExercise, EuropeanExercise, Exercise
from pquantlib.experimental.finitedifferences.fd_ext_ou_jump_vanilla_engine import (
    FdExtOUJumpVanillaEngine,
)
from pquantlib.experimental.finitedifferences.fd_kluge_ext_ou_spread_engine import (
    FdKlugeExtOUSpreadEngine,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    Discretization,
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.instruments.basket_option import AverageBasketPayoff, BasketOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/extoufd")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date(cpp_ref: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Probe: ``Settings::instance().evaluationDate() = TODAY`` (probe main)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["evaluationDate_serial"]))
    yield
    settings.evaluation_date = previous


# ---------------------------------------------------------------------------
# Section A — ExtendedOrnsteinUhlenbeckProcess moments
# ---------------------------------------------------------------------------


def _b_curve(t: float) -> float:
    """Probe: ``bCurve`` — ``0.75 + 0.4 sin(2t) + 0.1 t^2``."""
    return 0.75 + 0.4 * math.sin(2.0 * t) + 0.1 * t * t


_MODES = [
    ("midpoint", Discretization.MidPoint),
    ("trapezodial", Discretization.Trapezodial),
    ("gausslobatto", Discretization.GaussLobatto),
]


def test_b_curve_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The mean curve itself, before any process wraps it."""
    for t, want in zip(cpp_ref["proc_b_t"], cpp_ref["proc_b_value"], strict=True):
        tolerance.tight(_b_curve(t), want)


@pytest.mark.parametrize(("tag", "mode"), _MODES)
def test_process_moments_match_cpp(
    cpp_ref: dict[str, Any], tag: str, mode: Discretization
) -> None:
    """``ExtendedOrnsteinUhlenbeckProcess`` expectation / variance / stdDev.

    C++ extendedornsteinuhlenbeckprocess.cpp:66-111. ``variance`` and
    ``stdDeviation`` forward to the plain OU process; only ``expectation``
    depends on the mode.
    """
    p = ExtendedOrnsteinUhlenbeckProcess(
        cpp_ref["proc_speed"], cpp_ref["proc_vol"], cpp_ref["proc_x0"],
        _b_curve, mode,
    )
    x0 = cpp_ref["proc_x0"]
    for i, (t0, dt) in enumerate(
        zip(cpp_ref["proc_t0"], cpp_ref["proc_dt"], strict=True)
    ):
        tolerance.tight(
            p.expectation_1d(t0, x0, dt), cpp_ref[f"proc_{tag}_expectation"][i]
        )
        tolerance.tight(
            p.variance_1d(t0, x0, dt), cpp_ref[f"proc_{tag}_variance"][i]
        )
        tolerance.tight(
            p.std_deviation_1d(t0, x0, dt), cpp_ref[f"proc_{tag}_stdDeviation"][i]
        )
        tolerance.tight(p.drift_1d(t0, x0), cpp_ref[f"proc_{tag}_drift"][i])
        tolerance.tight(p.diffusion_1d(t0, x0), cpp_ref[f"proc_{tag}_diffusion"][i])


def test_modes_genuinely_differ(cpp_ref: dict[str, Any]) -> None:
    """The reference must actually discriminate the three modes.

    If ``b`` were constant or linear they would coincide and the test above
    would pass for a port that hardcoded one mode. Asserted on the C++ numbers
    so the fixture's discriminating power is itself pinned.
    """
    mid = cpp_ref["proc_midpoint_expectation"]
    trap = cpp_ref["proc_trapezodial_expectation"]
    lob = cpp_ref["proc_gausslobatto_expectation"]
    assert max(abs(a - b) for a, b in zip(mid, trap, strict=True)) > 1e-3
    assert max(abs(a - b) for a, b in zip(mid, lob, strict=True)) > 1e-3
    assert max(abs(a - b) for a, b in zip(trap, lob, strict=True)) > 1e-3


def test_constant_b_collapses_the_modes(cpp_ref: dict[str, Any]) -> None:
    """With constant ``b`` all three approximations are exact and equal."""
    for tag, mode in _MODES:
        p = ExtendedOrnsteinUhlenbeckProcess(
            cpp_ref["proc_speed"], cpp_ref["proc_vol"], cpp_ref["proc_x0"],
            lambda _t: 0.6, mode,
        )
        tolerance.tight(
            p.expectation_1d(0.3, cpp_ref["proc_x0"], 1.4),
            cpp_ref[f"proc_constb_{tag}"],
        )


# ---------------------------------------------------------------------------
# Section B — FdExtOUJumpVanillaEngine
# ---------------------------------------------------------------------------

_SCHEMES = {
    "hundsdorfer": FdmSchemeDesc.hundsdorfer,
    "douglas": FdmSchemeDesc.douglas,
    "craigsneyd": FdmSchemeDesc.craig_sneyd,
    "modcraigsneyd": FdmSchemeDesc.modified_craig_sneyd,
}


def _kluge_process() -> ExtOUWithJumpsProcess:
    """Probe: ``createKlugeProcess`` — the C++ test-suite numbers.

    swingoption.cpp:53-74.
    """
    x0 = 3.0
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 2.0, x0, lambda _t: x0)
    return ExtOUWithJumpsProcess(ou, 0.0, 5.0, 1.0, 2.0)


def _seasonal_shape(cpp_ref: dict[str, Any]) -> list[tuple[float, float]]:
    """The probe's own shape knots, read back rather than recomputed."""
    return [
        (t, v)
        for t, v in zip(cpp_ref["ext_shape_t"], cpp_ref["ext_shape_value"], strict=True)
    ]


def _price_jump_engine(
    cpp_ref: dict[str, Any],
    tag: str,
    exercise: Exercise,
    shape: list[tuple[float, float]] | None,
    scheme: FdmSchemeDesc,
) -> float:
    today = Date(int(cpp_ref["evaluationDate_serial"]))
    r_ts = FlatForward(
        today, SimpleQuote(cpp_ref[f"{tag}_irRate"]), Actual365Fixed()
    )
    option = VanillaOption(
        PlainVanillaPayoff(
            OptionType(int(cpp_ref[f"{tag}_type"])), cpp_ref[f"{tag}_strike"]
        ),
        exercise,
    )
    option.set_pricing_engine(
        FdExtOUJumpVanillaEngine(
            _kluge_process(),
            r_ts,
            int(cpp_ref[f"{tag}_tGrid"]),
            int(cpp_ref[f"{tag}_xGrid"]),
            int(cpp_ref[f"{tag}_yGrid"]),
            shape,
            scheme,
        )
    )
    return option.npv()


@pytest.mark.parametrize(
    "tag", ["ext_call_c0", "ext_call_c1", "ext_call_c2", "ext_put"]
)
def test_jump_engine_european_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``FdExtOUJumpVanillaEngine::calculate`` (fdextoujumpvanillaengine.cpp:54-98)."""
    exercise = EuropeanExercise(Date(int(cpp_ref["ext_maturity_serial"])))
    npv = _price_jump_engine(cpp_ref, tag, exercise, None, FdmSchemeDesc.hundsdorfer())
    tolerance.loose(npv, cpp_ref[f"{tag}_NPV"])


def test_jump_engine_refinement_moves_the_price(cpp_ref: dict[str, Any]) -> None:
    """The three grid sizes must give three different C++ answers.

    Otherwise ``test_jump_engine_european_matches_cpp`` would pass for a port
    whose grid resolution had no effect at all.
    """
    v0 = cpp_ref["ext_call_c0_NPV"]
    v1 = cpp_ref["ext_call_c1_NPV"]
    v2 = cpp_ref["ext_call_c2_NPV"]
    assert abs(v1 - v0) / abs(v0) > 1e-3
    assert abs(v2 - v1) / abs(v1) > 1e-3


def test_jump_engine_shape_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The seasonal shape enters the INNER VALUE, so it must move the NPV.

    A first draft of the probe used a shape whose value at maturity was
    -2.4e-16, and a European option samples the inner value only there, so the
    "shape" case reproduced the no-shape number exactly. The knots are also
    extended past maturity because C++
    ``FdmExtOUJumpModelInnerValue::innerValue`` dereferences
    ``lower_bound(...)`` without an ``end()`` check.
    """
    exercise = EuropeanExercise(Date(int(cpp_ref["ext_maturity_serial"])))
    npv = _price_jump_engine(
        cpp_ref, "ext_shape", exercise, _seasonal_shape(cpp_ref),
        FdmSchemeDesc.hundsdorfer(),
    )
    tolerance.loose(npv, cpp_ref["ext_shape_NPV"])
    assert (
        abs(cpp_ref["ext_shape_NPV"] - cpp_ref["ext_call_c1_NPV"])
        / cpp_ref["ext_call_c1_NPV"]
        > 0.1
    ), "the shape case no longer discriminates a shape-ignoring port"


@pytest.mark.parametrize(
    ("tag", "scheme_key"),
    [
        ("ext_douglas", "douglas"),
        ("ext_craigsneyd", "craigsneyd"),
        ("ext_modcraigsneyd", "modcraigsneyd"),
    ],
)
def test_jump_engine_scheme_argument_matches_cpp(
    cpp_ref: dict[str, Any], tag: str, scheme_key: str
) -> None:
    """The scheme is a constructor argument; a port that ignores it fails here."""
    exercise = EuropeanExercise(Date(int(cpp_ref["ext_maturity_serial"])))
    npv = _price_jump_engine(cpp_ref, tag, exercise, None, _SCHEMES[scheme_key]())
    tolerance.loose(npv, cpp_ref[f"{tag}_NPV"])


@pytest.mark.parametrize("tag", ["ext_bermudan", "ext_bermudan_shape"])
def test_jump_engine_bermudan_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """Reaches ``FdmStepConditionComposite.vanilla_composite``'s Bermudan branch.

    That branch contributes both a step condition and stopping times
    (fdmstepconditioncomposite.cpp:133-140); the shape variant additionally
    samples the shape at four different times, exercising its interpolation
    rather than only its endpoint.
    """
    dates = [Date(int(s)) for s in cpp_ref["ext_bermudan_dates"]]
    exercise = BermudanExercise(dates)
    shape = _seasonal_shape(cpp_ref) if tag.endswith("_shape") else None
    npv = _price_jump_engine(
        cpp_ref, tag, exercise, shape, FdmSchemeDesc.hundsdorfer()
    )
    tolerance.loose(npv, cpp_ref[f"{tag}_NPV"])


# ---------------------------------------------------------------------------
# Section C — FdKlugeExtOUSpreadEngine
# ---------------------------------------------------------------------------


def _kluge_ext_ou_process() -> KlugeExtOUProcess:
    """Probe: ``createKlugeExtOUProcess`` — vpp.cpp:205-234 numbers."""
    beta = 200.0
    eta = 1.0 / 0.2
    lambda_ = 4.0
    alpha = 7.0
    volatility_x = 1.4
    kappa = 4.45
    volatility_u = math.sqrt(1.3)
    rho = 0.7

    ou = ExtendedOrnsteinUhlenbeckProcess(alpha, volatility_x, 0.0, lambda _t: 0.0)
    ln_power = ExtOUWithJumpsProcess(ou, 0.0, beta, lambda_, eta)
    ln_gas = ExtendedOrnsteinUhlenbeckProcess(kappa, volatility_u, 0.0, lambda _t: 0.0)
    return KlugeExtOUProcess(rho, ln_power, ln_gas)


def _flat_shape(level: float) -> list[tuple[float, float]]:
    """Probe: ``flatShape`` — constant level on knots 0 .. 1.5."""
    return [(i / 4.0, level) for i in range(7)]


def _price_spread_engine(
    cpp_ref: dict[str, Any],
    tag: str,
    gas_shape: list[tuple[float, float]] | None,
    power_shape: list[tuple[float, float]] | None,
    scheme: FdmSchemeDesc,
) -> float:
    today = Date(int(cpp_ref["evaluationDate_serial"]))
    r_ts = FlatForward(
        today, SimpleQuote(cpp_ref[f"{tag}_irRate"]), Actual365Fixed()
    )
    payoff = AverageBasketPayoff(
        PlainVanillaPayoff(OptionType.Call, 0.0),
        [1.0, -cpp_ref[f"{tag}_heatRate"]],
    )
    option = BasketOption(
        payoff,
        EuropeanExercise(Date(int(cpp_ref[f"{tag}_maturity_serial"]))),
    )
    option.set_pricing_engine(
        FdKlugeExtOUSpreadEngine(
            _kluge_ext_ou_process(),
            r_ts,
            int(cpp_ref[f"{tag}_tGrid"]),
            int(cpp_ref[f"{tag}_xGrid"]),
            int(cpp_ref[f"{tag}_yGrid"]),
            int(cpp_ref[f"{tag}_uGrid"]),
            gas_shape,
            power_shape,
            scheme,
        )
    )
    return option.npv()


@pytest.mark.parametrize(
    "tag", ["spr_c0", "spr_c1", "spr_c2", "spr_neg", "spr_long"]
)
def test_spread_engine_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``FdKlugeExtOUSpreadEngine::calculate`` (fdklugeextouspreadengine.cpp:54-123).

    ``spr_neg`` reverses the basket weights, so the payoff's other branch runs.
    """
    npv = _price_spread_engine(
        cpp_ref, tag, None, None, FdmSchemeDesc.hundsdorfer()
    )
    tolerance.loose(npv, cpp_ref[f"{tag}_NPV"])


def test_spread_engine_shapes_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """Gas and power shapes are DIFFERENT curves fed to DIFFERENT calculators.

    Gas goes through ``FdmExpExtOUInnerValueCalculator`` on direction 2, power
    through ``FdmExtOUJumpModelInnerValue`` on directions 0 and 1. Swapping
    them changes the answer, which is why the two levels differ here.
    """
    npv = _price_spread_engine(
        cpp_ref, "spr_shapes", _flat_shape(0.3), _flat_shape(-0.2),
        FdmSchemeDesc.hundsdorfer(),
    )
    tolerance.loose(npv, cpp_ref["spr_shapes_NPV"])


@pytest.mark.parametrize(
    ("tag", "scheme_key"),
    [("spr_douglas", "douglas"), ("spr_craigsneyd", "craigsneyd")],
)
def test_spread_engine_scheme_argument_matches_cpp(
    cpp_ref: dict[str, Any], tag: str, scheme_key: str
) -> None:
    npv = _price_spread_engine(cpp_ref, tag, None, None, _SCHEMES[scheme_key]())
    tolerance.loose(npv, cpp_ref[f"{tag}_NPV"])


def test_spread_engine_refinement_moves_the_price(cpp_ref: dict[str, Any]) -> None:
    v0 = cpp_ref["spr_c0_NPV"]
    v1 = cpp_ref["spr_c1_NPV"]
    v2 = cpp_ref["spr_c2_NPV"]
    assert abs(v1 - v0) / abs(v0) > 1e-3
    assert abs(v2 - v1) / abs(v1) > 1e-3


def test_spread_engine_rejects_a_non_basket_payoff(cpp_ref: dict[str, Any]) -> None:
    """C++ ``QL_REQUIRE(basketPayoff, " basket payoff expected")``.

    fdklugeextouspreadengine.cpp:83, leading space and all.
    """
    today = Date(int(cpp_ref["evaluationDate_serial"]))
    r_ts = FlatForward(today, SimpleQuote(0.0), Actual365Fixed())
    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 1.0),
        EuropeanExercise(Date(int(cpp_ref["spr_c0_maturity_serial"]))),
    )
    option.set_pricing_engine(
        FdKlugeExtOUSpreadEngine(_kluge_ext_ou_process(), r_ts, 3, 8, 4, 4)
    )
    with pytest.raises(LibraryException, match="basket payoff expected"):
        option.npv()
