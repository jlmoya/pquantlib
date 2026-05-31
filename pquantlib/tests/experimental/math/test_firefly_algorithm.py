"""Cross-validate FireflyAlgorithm convergence (firefly + DE hybrid).

Probe source: migration-harness/cpp/probes/cluster_w6d/probe.cpp
Reference:    migration-harness/references/cluster/w6d.json

Firefly is a stochastic optimizer; with a fixed seed it is
deterministic. The contract is convergence into the global basin
(LOOSE). The probe emits the analytic global optima; these tests run
the optimizer with fixed seeds and assert the result lands in the basin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.firefly_algorithm import (
    ExponentialIntensity,
    FireflyAlgorithm,
    GaussianWalk,
    InverseLawSquareIntensity,
    LevyFlightWalk,
)
from pquantlib.math.optimization.constraint import BoundaryConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6d")


class _Sphere(CostFunction):
    """f(x) = sum x_i^2 — global min 0 at the origin."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()

    def value(self, x: npt.NDArray[np.float64]) -> float:
        return float(np.sum(x * x))


def _ec() -> EndCriteria:
    return EndCriteria(2000, 500, 1e-8, 1e-8, 1e-8)


def test_firefly_gaussian_walk_sphere(cpp_ref: dict[str, Any]) -> None:
    """Firefly with a Gaussian walk converges to the sphere minimum."""
    opt = cpp_ref["optimizers"]
    fa = FireflyAlgorithm(
        40, ExponentialIntensity(1.0, 0.1, 0.01), GaussianWalk(0.1, seed=3), seed=42
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -2.0]))
    ec_type = fa.minimize(problem, _ec())

    assert ec_type in (Type.StationaryPoint, Type.MaxIterations)
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-4,
        rel_tol=0.0,
        reason="firefly Gaussian-walk sphere convergence (seed 42)",
    )
    for j in range(2):
        tolerance.custom(
            float(problem.current_value[j]),
            float(opt["sphere_min_x"][j]),
            abs_tol=1e-2,
            rel_tol=0.0,
            reason="firefly sphere argmin near origin",
        )


def test_firefly_levy_walk_sphere(cpp_ref: dict[str, Any]) -> None:
    """Firefly with a Lévy-flight walk converges to the sphere minimum."""
    opt = cpp_ref["optimizers"]
    fa = FireflyAlgorithm(
        40, ExponentialIntensity(1.0, 0.1, 0.01), LevyFlightWalk(1.5, seed=7), seed=11
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([4.0, 4.0]))
    fa.minimize(problem, _ec())
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-4,
        rel_tol=0.0,
        reason="firefly Levy-walk sphere convergence (seed 11)",
    )


def test_firefly_inverse_square_intensity() -> None:
    """The inverse-square intensity kernel also converges."""
    fa = FireflyAlgorithm(
        40, InverseLawSquareIntensity(1.0, 0.1), GaussianWalk(0.1, seed=9), seed=21
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([2.0, -3.0]))
    fa.minimize(problem, _ec())
    assert problem.function_value < 1e-2


def test_firefly_hybrid_de() -> None:
    """Firefly with a DE subpopulation (Mde > 0) converges."""
    fa = FireflyAlgorithm(
        40,
        ExponentialIntensity(1.0, 0.1, 0.01),
        GaussianWalk(0.1, seed=3),
        m_de=10,
        seed=42,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -2.0]))
    fa.minimize(problem, _ec())
    assert problem.function_value < 1e-2


def test_firefly_pure_de(cpp_ref: dict[str, Any]) -> None:
    """Firefly with Mde == M is a pure differential-evolution optimizer."""
    opt = cpp_ref["optimizers"]
    fa = FireflyAlgorithm(
        30,
        ExponentialIntensity(1.0, 0.1, 0.01),
        GaussianWalk(0.1, seed=3),
        m_de=30,
        seed=42,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -2.0]))
    fa.minimize(problem, _ec())
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-4,
        rel_tol=0.0,
        reason="pure-DE firefly sphere convergence",
    )


def test_firefly_respects_bounds() -> None:
    """Firefly never returns a point outside the box constraint."""
    fa = FireflyAlgorithm(
        30, ExponentialIntensity(1.0, 0.1, 0.01), GaussianWalk(0.1, seed=1), seed=3
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-2.0, 2.0), np.array([1.0, 1.0]))
    fa.minimize(problem, _ec())
    assert bool(np.all(problem.current_value >= -2.0 - 1e-12))
    assert bool(np.all(problem.current_value <= 2.0 + 1e-12))


def test_firefly_invalid_mde_rejected() -> None:
    """Mde > M is rejected at construction."""
    with pytest.raises(LibraryException, match="cannot be larger than total population"):
        FireflyAlgorithm(
            10, ExponentialIntensity(1.0, 0.1, 0.01), GaussianWalk(0.1), m_de=20
        )


def test_firefly_determinism_same_seed() -> None:
    """Identical seeds give identical results."""
    results: list[float] = []
    for _ in range(2):
        fa = FireflyAlgorithm(
            30, ExponentialIntensity(1.0, 0.1, 0.01), GaussianWalk(0.1, seed=4), seed=99
        )
        problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -1.0]))
        fa.minimize(problem, _ec())
        results.append(problem.function_value)
    tolerance.exact(results[0], results[1])
