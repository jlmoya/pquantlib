"""Cross-validate the non-Gauss 1-D integrators against C++ QuantLib v1.43.

Probe source: ``migration-harness/cpp/probes/v143_math_integrals/probe.cpp``
Reference:    ``migration-harness/references/v143/math/integrals.json``

Covers ``discreteintegrals.{hpp,cpp}``, ``filonintegral.{hpp,cpp}``,
``kronrodintegral.cpp`` (``GaussKronrodNonAdaptive``),
``twodimensionalintegral.hpp``, the ``trapezoidintegral.hpp`` policies and the
two double-exponential integrators.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pquantlib.math.integrals.discrete_integrals import (
    DiscreteSimpsonIntegral,
    DiscreteSimpsonIntegrator,
    DiscreteTrapezoidIntegral,
    DiscreteTrapezoidIntegrator,
)
from pquantlib.math.integrals.exp_sinh_integral import ExpSinhIntegral
from pquantlib.math.integrals.filon import FilonIntegral
from pquantlib.math.integrals.gaussian_quadrature import GaussLegendreIntegrator
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.kronrod import GaussKronrodNonAdaptive
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.math.integrals.tanh_sinh_integral import TanhSinhIntegral
from pquantlib.math.integrals.trapezoid import Default, MidPoint, TrapezoidIntegral
from pquantlib.math.integrals.two_dimensional_integral import TwoDimensionalIntegral
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/integrals")


def _sqrt_abs(x: float) -> float:
    return math.sqrt(abs(x))


INTEGRANDS: dict[str, Callable[[float], float]] = {
    "one": lambda x: 1.0,
    "x": lambda x: x,
    "x2": lambda x: x * x,
    "x4": lambda x: x * x * x * x,
    "inv_1px2": lambda x: 1.0 / (1.0 + x * x),
    "sin_x": math.sin,
    "abs_x": abs,
    "sqrt_abs_x": _sqrt_abs,
}

# Integrands the probe defines locally, per section.
LOCAL_INTEGRANDS: dict[str, Callable[[float], float]] = {
    "x2": lambda x: x * x,
    "sin_x": math.sin,
    "sin_50x": lambda x: math.sin(50.0 * x),
    "sin_200x": lambda x: math.sin(200.0 * x),
    "inv_sqrt_x": lambda x: 1.0 / math.sqrt(x),
    "inv_1px2": lambda x: 1.0 / (1.0 + x * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "log_x": math.log,
    "sqrt_1mx2": lambda x: math.sqrt(1.0 - x * x),
    "gauss": lambda x: math.exp(-x * x),
    "exp_neg_x": lambda x: math.exp(-x),
    "exp_x": math.exp,
    "x_exp_neg_x2": lambda x: x * math.exp(-x * x),
    "inv_sqrt_exp": lambda x: math.exp(-x) / math.sqrt(x),
}


# --- discrete integrals -------------------------------------------------


@pytest.mark.parametrize("case", range(6))
def test_discrete_integrals(cpp: dict[str, Any], case: int) -> None:
    """Composite rules on sampled, non-uniform grids, at TIGHT.

    The grids alternate odd and even sample counts: with an even count
    ``DiscreteSimpsonIntegral`` adds the trailing trapezoid correction for the
    unpaired last panel, which is a branch a port can silently drop.
    """
    block = cpp["discrete_integrals"][case]
    x = np.asarray(block["x"], dtype=np.float64)
    f = np.asarray(block["f"], dtype=np.float64)

    tolerance.tight(DiscreteTrapezoidIntegral()(x, f), float(block["trapezoid"]))
    tolerance.tight(DiscreteSimpsonIntegral()(x, f), float(block["simpson"]))


@pytest.mark.parametrize("case", range(6))
def test_discrete_integrators(cpp: dict[str, Any], case: int) -> None:
    """The Integrator wrappers, value and evaluation count, at TIGHT.

    ``DiscreteSimpsonIntegrator`` is transcribed with its two successive
    ``sum *= 2`` and its asymmetric ``1.5 f(b) + 2.5 f(b-d)`` closing term for
    odd ``n``; both odd and even ``evaluations`` are exercised.
    """
    block = cpp["discrete_integrators"][case]
    f = INTEGRANDS[block["integrand"]]
    a, b = float(block["a"]), float(block["b"])

    trapezoid = DiscreteTrapezoidIntegrator(int(block["evaluations"]))
    tolerance.tight(trapezoid(f, a, b), float(block["trapezoid"]))
    assert trapezoid.number_of_evaluations() == int(block["trapezoid_evaluations"])

    simpson = DiscreteSimpsonIntegrator(int(block["evaluations"]))
    tolerance.tight(simpson(f, a, b), float(block["simpson"]))
    assert simpson.number_of_evaluations() == int(block["simpson_evaluations"])


# --- Filon --------------------------------------------------------------


@pytest.mark.parametrize("case", range(8))
def test_filon(cpp: dict[str, Any], case: int) -> None:
    """Filon sine/cosine quadrature at TIGHT.

    ``t = 0.1`` over 20 panels puts ``theta = t*h`` at 0.01, where alpha, beta
    and gamma each cancel catastrophically (``1/theta + sin(2 theta)/(2
    theta^2) - 2 sin^2(theta)/theta^3`` is a difference of terms up to 1e6
    producing an O(1) result). That is where an algebraically-equivalent but
    differently-associated rewrite of the coefficients would show up.
    """
    block = cpp["filon"][case]
    kind = (
        FilonIntegral.Type.Cosine
        if block["type"] == "Cosine"
        else FilonIntegral.Type.Sine
    )
    filon = FilonIntegral(kind, float(block["t"]), int(block["intervals"]))
    got = filon(INTEGRANDS[block["integrand"]], float(block["a"]), float(block["b"]))
    tolerance.tight(got, float(block["value"]))
    assert filon.max_evaluations() == int(block["max_evaluations"])


def test_filon_rejects_odd_interval_count() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    with pytest.raises(LibraryException, match="number of intervals must be even"):
        FilonIntegral(FilonIntegral.Type.Sine, 1.0, 7)


# --- GaussKronrodNonAdaptive -------------------------------------------


@pytest.mark.parametrize("case", range(8))
def test_gauss_kronrod_non_adaptive(cpp: dict[str, Any], case: int) -> None:
    """The 10/21/43/87 Patterson ladder, at TIGHT.

    Three things are pinned per case, and the evaluation count is the sharpest
    of them: it says *which* rung of the ladder returned, so a transcription
    error in any of the eleven abscissa/weight tables moves it. The cases are
    chosen to stop at 21 (smooth), and to run all the way to 87 without
    converging (``sin(50x)``, ``sin(200x)``, ``1/sqrt(x)``, Runge) — the
    non-convergent ones also pin ``integration_success() == False``.
    """
    block = cpp["kronrod_non_adaptive"][case]
    integrator = GaussKronrodNonAdaptive(
        float(block["abs_accuracy"]), 100, float(block["rel_accuracy"])
    )
    got = integrator(
        LOCAL_INTEGRANDS[block["integrand"]], float(block["a"]), float(block["b"])
    )
    tolerance.tight(got, float(block["value"]))
    tolerance.tight(integrator.absolute_error(), float(block["absolute_error"]))
    assert integrator.number_of_evaluations() == int(block["evaluations"])
    assert integrator.integration_success() is bool(block["success"])
    tolerance.exact(integrator.relative_accuracy(), float(block["rel_accuracy"]))


def test_gauss_kronrod_non_adaptive_relative_accuracy_is_mutable() -> None:
    integrator = GaussKronrodNonAdaptive(1e-8, 100, 1e-8)
    integrator.set_relative_accuracy(1e-3)
    tolerance.exact(integrator.relative_accuracy(), 1e-3)


# --- TwoDimensionalIntegral --------------------------------------------


def _f2d(x: float, y: float) -> float:
    return math.exp(-x * x) * math.sin(y) + x * y


def _make_integrator(name: str) -> Integrator:
    if name == "segment_50":
        return SegmentIntegral(50)
    if name == "legendre_16":
        return GaussLegendreIntegrator(16)
    if name == "simpson_1e-6":
        return SimpsonIntegral(1e-6, 40)
    raise AssertionError(f"unknown integrator {name}")


@pytest.mark.parametrize("case", range(5))
def test_two_dimensional_integral(cpp: dict[str, Any], case: int) -> None:
    """Nested integration, at TIGHT.

    Includes a degenerate box (``a_x == b_x``, which must short-circuit to 0
    through ``Integrator::operator()``) and a reversed one (``b_x < a_x``,
    which must flip the sign of the whole nested result, not of the inner
    one).
    """
    block = cpp["two_dimensional"][case]
    tdi = TwoDimensionalIntegral(
        _make_integrator(block["integrator_x"]), _make_integrator(block["integrator_y"])
    )
    a = (float(block["a"][0]), float(block["a"][1]))
    b = (float(block["b"][0]), float(block["b"][1]))
    tolerance.tight(tdi(_f2d, a, b), float(block["value"]))


# --- trapezoid policies -------------------------------------------------


@pytest.mark.parametrize("case", range(9))
def test_trapezoid_policies(cpp: dict[str, Any], case: int) -> None:
    """``Default`` and ``MidPoint`` refinement, value and evaluation count.

    The evaluation count is the discriminating quantity: ``Default`` adds
    ``N`` points and doubles ``N``, ``MidPoint`` adds ``2N`` and triples it, so
    the counts (131073 vs 59050, ...) fingerprint the policy independently of
    the value.
    """
    block = cpp["trapezoid_policies"][case]
    policy = Default if block["policy"] == "Default" else MidPoint
    integral = TrapezoidIntegral(
        float(block["accuracy"]), int(block["max_iterations"]), policy
    )
    got = integral(
        INTEGRANDS[block["integrand"]], float(block["a"]), float(block["b"])
    )
    tolerance.tight(got, float(block["value"]))
    assert integral.number_of_evaluations() == int(block["evaluations"])


def test_trapezoid_default_is_the_default_policy(cpp: dict[str, Any]) -> None:
    block = next(e for e in cpp["trapezoid_policies"] if e["name"] == "default_x2_0_1")
    integral = TrapezoidIntegral(float(block["accuracy"]), int(block["max_iterations"]))
    tolerance.tight(
        integral(INTEGRANDS["x2"], float(block["a"]), float(block["b"])),
        float(block["value"]),
    )


# --- double-exponential quadrature --------------------------------------


@pytest.mark.parametrize("case", range(10))
def test_tanh_sinh_integral(cpp: dict[str, Any], case: int) -> None:
    """Tanh-sinh quadrature against boost's, at TIGHT.

    C++ delegates to ``boost::math::quadrature::tanh_sinh``; the Python port
    implements the double-exponential rule directly. TIGHT is justified
    structurally rather than by the stopping tolerance: both sides sum the same
    node family ``t = j 2^-k`` with the same weights and the same level
    refinement, so once both have converged the sums agree to summation
    rounding. In practice most of these cases are bit-identical; the worst is
    3 ulp.

    The cases include the two things that break a naive DE implementation: an
    endpoint singularity (``1/sqrt(x)``, ``log(x)`` at ``a = 0``), which needs
    the abscissa to be built from its complement so it never lands exactly on
    the bound, and an oscillatory integrand that forces several extra levels.
    """
    block = cpp["tanh_sinh"][case]
    integral = TanhSinhIntegral(float(block["rel_tolerance"]))
    got = integral(
        LOCAL_INTEGRANDS[block["integrand"]], float(block["a"]), float(block["b"])
    )
    tolerance.tight(got, float(block["value"]))


@pytest.mark.parametrize("case", range(10))
def test_exp_sinh_integral(cpp: dict[str, Any], case: int) -> None:
    """Exp-sinh quadrature against boost's, at TIGHT.

    Same justification as the tanh-sinh case. Exercised through both entry
    points C++ exposes — the non-virtual ``integrate(f)`` for ``[0, inf)`` and
    ``operator()(f, a, b)`` with one infinite bound, on both sides.
    """
    block = cpp["exp_sinh"][case]
    integral = ExpSinhIntegral(float(block["rel_tolerance"]))
    f = LOCAL_INTEGRANDS[block["integrand"]]

    if block["half_infinite_overload"]:
        got = integral.integrate(f)
    elif block["lower_is_finite"]:
        got = integral(f, float(block["a"]), math.inf)
    else:
        got = integral(f, -math.inf, float(block["a"]))

    tolerance.tight(got, float(block["value"]))
    assert integral.number_of_evaluations() > 0


def test_exp_sinh_requires_exactly_one_infinite_bound() -> None:
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415

    integral = ExpSinhIntegral()
    with pytest.raises(LibraryException, match="exactly one"):
        integral(LOCAL_INTEGRANDS["exp_neg_x"], 0.0, 1.0)
