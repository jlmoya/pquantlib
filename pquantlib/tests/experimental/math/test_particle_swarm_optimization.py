"""Cross-validate ParticleSwarmOptimization convergence.

Probe source: migration-harness/cpp/probes/cluster_w6d/probe.cpp
Reference:    migration-harness/references/cluster/w6d.json

PSO is a stochastic optimizer; with a fixed seed it is deterministic,
but the contract is convergence into the global basin (LOOSE), not an
exact value. The probe emits only the analytic global optima of the
test functions; these tests run PSO with fixed seeds and assert the
returned point lands in the basin and the value is near the optimum.
The seed dependence is intrinsic — different seeds explore differently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.particle_swarm_optimization import (
    DecreasingInertia,
    GlobalTopology,
    KNeighbors,
    ParticleSwarmOptimization,
    SimpleRandomInertia,
    TrivialInertia,
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


class _Rosenbrock(CostFunction):
    """2-D Rosenbrock — global min 0 at (1, 1)."""

    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self._a = a
        self._b = b

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()

    def value(self, x: npt.NDArray[np.float64]) -> float:
        return float((self._a - x[0]) ** 2 + self._b * (x[1] - x[0] ** 2) ** 2)


def _end_criteria() -> EndCriteria:
    return EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8)


def test_pso_sphere_global_topology(cpp_ref: dict[str, Any]) -> None:
    """PSO converges to the sphere's global minimum (origin)."""
    opt = cpp_ref["optimizers"]
    pso = ParticleSwarmOptimization(40, GlobalTopology(), TrivialInertia(), seed=42)
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -2.0]))
    ec_type = pso.minimize(problem, _end_criteria())

    assert ec_type in (Type.StationaryPoint, Type.MaxIterations)
    # LOOSE: value at the global minimum is 0.
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-6,
        rel_tol=0.0,
        reason="PSO stochastic — converge to sphere min (seed 42)",
    )
    expected_x = opt["sphere_min_x"]
    for j in range(2):
        tolerance.custom(
            float(problem.current_value[j]),
            float(expected_x[j]),
            abs_tol=1e-3,
            rel_tol=0.0,
            reason="PSO sphere argmin near origin",
        )


def test_pso_rosenbrock_global_topology(cpp_ref: dict[str, Any]) -> None:
    """PSO lands in the Rosenbrock global basin near (1, 1)."""
    opt = cpp_ref["optimizers"]
    pso = ParticleSwarmOptimization(60, GlobalTopology(), TrivialInertia(), seed=7)
    problem = Problem(
        _Rosenbrock(float(opt["rosenbrock_a"]), float(opt["rosenbrock_b"])),
        BoundaryConstraint(-5.0, 10.0),
        np.array([0.0, 0.0]),
    )
    pso.minimize(problem, _end_criteria())

    # Rosenbrock is hard; LOOSE basin contract: value small, x near (1,1).
    assert problem.function_value < 0.5
    expected_x = opt["rosenbrock_min_x"]
    for j in range(2):
        tolerance.custom(
            float(problem.current_value[j]),
            float(expected_x[j]),
            abs_tol=0.1,
            rel_tol=0.0,
            reason="PSO Rosenbrock argmin in global basin (seed 7)",
        )


def test_pso_k_neighbors_topology(cpp_ref: dict[str, Any]) -> None:
    """K-neighbour topology also converges to the sphere minimum."""
    opt = cpp_ref["optimizers"]
    pso = ParticleSwarmOptimization(40, KNeighbors(2), TrivialInertia(), seed=123)
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([4.0, 4.0]))
    pso.minimize(problem, _end_criteria())
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-4,
        rel_tol=0.0,
        reason="PSO KNeighbors sphere convergence (seed 123)",
    )


def test_pso_in_inertia_variant() -> None:
    """The PSO-In (omega) inertia-factor variant converges on the sphere."""
    pso = ParticleSwarmOptimization(
        40, GlobalTopology(), TrivialInertia(), omega=0.7, c1=1.5, c2=1.5, seed=99
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, 3.0]))
    pso.minimize(problem, _end_criteria())
    tolerance.custom(
        problem.function_value, 0.0, abs_tol=1e-4, rel_tol=0.0, reason="PSO-In convergence"
    )


def test_pso_simple_random_inertia() -> None:
    """SimpleRandomInertia variant converges on the sphere."""
    pso = ParticleSwarmOptimization(
        40, GlobalTopology(), SimpleRandomInertia(0.5, seed=5), seed=11
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([2.0, -3.0]))
    pso.minimize(problem, _end_criteria())
    assert problem.function_value < 1e-3


def test_pso_decreasing_inertia() -> None:
    """DecreasingInertia variant converges on the sphere."""
    pso = ParticleSwarmOptimization(
        40, GlobalTopology(), DecreasingInertia(0.4), seed=17
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([-4.0, 1.0]))
    pso.minimize(problem, _end_criteria())
    assert problem.function_value < 1e-3


def test_pso_respects_bounds() -> None:
    """PSO never returns a point outside the box constraint."""
    pso = ParticleSwarmOptimization(30, GlobalTopology(), TrivialInertia(), seed=3)
    problem = Problem(_Sphere(), BoundaryConstraint(-2.0, 2.0), np.array([1.0, 1.0]))
    pso.minimize(problem, _end_criteria())
    assert bool(np.all(problem.current_value >= -2.0 - 1e-12))
    assert bool(np.all(problem.current_value <= 2.0 + 1e-12))


def test_pso_invalid_phi_rejected() -> None:
    """phi == 0 (c1 == c2 == 0) is rejected by the constriction factor."""
    with pytest.raises(LibraryException, match="Invalid phi"):
        ParticleSwarmOptimization(10, GlobalTopology(), TrivialInertia(), c1=0.0, c2=0.0)


def test_pso_k_neighbors_too_many_rejected() -> None:
    """KNeighbors with K >= M is rejected at start_state."""
    pso = ParticleSwarmOptimization(3, KNeighbors(5), TrivialInertia(), seed=1)
    problem = Problem(_Sphere(), BoundaryConstraint(-1.0, 1.0), np.array([0.5, 0.5]))
    with pytest.raises(LibraryException, match="smaller than total particles"):
        pso.minimize(problem, _end_criteria())


def test_pso_determinism_same_seed() -> None:
    """Identical seeds give identical results (deterministic stochasticity)."""
    results: list[tuple[float, npt.NDArray[np.float64]]] = []
    for _ in range(2):
        pso = ParticleSwarmOptimization(30, GlobalTopology(), TrivialInertia(), seed=555)
        problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, -1.0]))
        pso.minimize(problem, _end_criteria())
        results.append((problem.function_value, problem.current_value.copy()))
    tolerance.exact(results[0][0], results[1][0])
    for j in range(2):
        tolerance.exact(float(results[0][1][j]), float(results[1][1][j]))
