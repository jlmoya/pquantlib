"""Cross-validate HybridSimulatedAnnealing convergence + functor families.

Probe source: migration-harness/cpp/probes/cluster_w6d/probe.cpp
Reference:    migration-harness/references/cluster/w6d.json

HSA is a stochastic optimizer; with a fixed seed it is deterministic.
The contract is convergence into the global basin (LOOSE). The probe
emits the analytic global optima; these tests run HSA under various
sampler/probability/temperature/reannealing combinations and assert the
result lands in the basin. The functor families are unit-tested
directly for their closed-form schedules where applicable (TIGHT).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.experimental.math.hybrid_simulated_annealing import (
    HybridSimulatedAnnealing,
    LocalOptimizeScheme,
    ResetScheme,
)
from pquantlib.experimental.math.hybrid_simulated_annealing_functors import (
    ProbabilityAlwaysDownhill,
    ProbabilityBoltzmann,
    ProbabilityBoltzmannDownhill,
    ReannealingFiniteDifferences,
    ReannealingTrivial,
    SamplerCauchy,
    SamplerGaussian,
    SamplerLogNormal,
    SamplerMirrorGaussian,
    TemperatureBoltzmann,
    TemperatureCauchy,
    TemperatureExponential,
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


# --------------------------------------------------------------------------
# Driver convergence
# --------------------------------------------------------------------------


def test_hsa_gaussian_boltzmann_no_local(cpp_ref: dict[str, Any]) -> None:
    """Pure SA (Gaussian + Boltzmann-downhill, no local opt) converges."""
    opt = cpp_ref["optimizers"]
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=5),
        ProbabilityBoltzmannDownhill(seed=6),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
        local_optimizer=None,
        optimize_scheme=LocalOptimizeScheme.NoLocalOptimize,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([3.0, -3.0]))
    ec_type = hsa.minimize(problem, _ec())
    assert ec_type in (Type.StationaryPoint, Type.MaxIterations)
    tolerance.custom(
        problem.function_value,
        float(opt["sphere_min_value"]),
        abs_tol=1e-3,
        rel_tol=0.0,
        reason="pure-SA sphere convergence (seed 5/6)",
    )


def test_hsa_with_levenberg_marquardt_local() -> None:
    """The default local optimizer (LevenbergMarquardt) polishes to ~0."""
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=5),
        ProbabilityBoltzmannDownhill(seed=6),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([3.0, -3.0]))
    hsa.minimize(problem, _ec())
    tolerance.custom(
        problem.function_value, 0.0, abs_tol=1e-6, rel_tol=0.0, reason="HSA+LM polish"
    )


def test_hsa_cauchy_sampler() -> None:
    """Cauchy sampler + Cauchy temperature converges."""
    hsa = HybridSimulatedAnnealing(
        SamplerCauchy(seed=7),
        ProbabilityBoltzmannDownhill(seed=8),
        TemperatureCauchy(5.0, 2),
        start_temperature=5.0,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([2.0, -2.0]))
    hsa.minimize(problem, _ec())
    assert problem.function_value < 1e-2


def test_hsa_lognormal_sampler() -> None:
    """Lognormal sampler converges on a positive-support sphere."""
    hsa = HybridSimulatedAnnealing(
        SamplerLogNormal(seed=3),
        ProbabilityBoltzmannDownhill(seed=4),
        TemperatureExponential(2.0, 2),
        start_temperature=2.0,
        local_optimizer=None,
    )
    # Positive starting point; lognormal keeps it positive.
    problem = Problem(_Sphere(), BoundaryConstraint(0.01, 10.0), np.array([3.0, 4.0]))
    hsa.minimize(problem, _ec())
    # Min over the positive box is at the lower corner near 0.01.
    assert problem.function_value < 1.0


def test_hsa_mirror_gaussian_sampler() -> None:
    """Mirror-Gaussian sampler stays inside bounds and converges."""
    lower = np.array([-5.0, -5.0], dtype=np.float64)
    upper = np.array([5.0, 5.0], dtype=np.float64)
    hsa = HybridSimulatedAnnealing(
        SamplerMirrorGaussian(lower, upper, seed=2),
        ProbabilityBoltzmannDownhill(seed=3),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-5.0, 5.0), np.array([3.0, 3.0]))
    hsa.minimize(problem, _ec())
    assert problem.function_value < 1e-2


def test_hsa_boltzmann_probability() -> None:
    """The plain Boltzmann probability (no auto-downhill) converges."""
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=1),
        ProbabilityBoltzmann(seed=2),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([3.0, -3.0]))
    hsa.minimize(problem, _ec())
    assert problem.function_value < 0.5


def test_hsa_always_downhill_probability() -> None:
    """The always-downhill (greedy) probability converges (may local-trap)."""
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=1),
        ProbabilityAlwaysDownhill(),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([3.0, -3.0]))
    hsa.minimize(problem, _ec())
    # Sphere is unimodal, so greedy still reaches the global minimum.
    assert problem.function_value < 1e-2


def test_hsa_reannealing_finite_differences() -> None:
    """Finite-difference reannealing converges."""
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=1),
        ProbabilityBoltzmannDownhill(seed=2),
        TemperatureExponential(5.0, 2),
        reannealing=ReannealingFiniteDifferences(5.0, 2),
        start_temperature=5.0,
        re_anneal_steps=20,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([2.0, 2.0]))
    hsa.minimize(problem, _ec())
    assert problem.function_value < 0.1


def test_hsa_reset_to_origin() -> None:
    """ResetToOrigin reset scheme runs without error and converges."""
    hsa = HybridSimulatedAnnealing(
        SamplerGaussian(seed=1),
        ProbabilityBoltzmannDownhill(seed=2),
        TemperatureExponential(5.0, 2),
        start_temperature=5.0,
        reset_scheme=ResetScheme.ResetToOrigin,
        reset_steps=50,
        local_optimizer=None,
    )
    problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([2.0, -2.0]))
    hsa.minimize(problem, _ec())
    assert problem.function_value < 0.5


def test_hsa_determinism_same_seed() -> None:
    """Identical seeds give identical results."""
    results: list[float] = []
    for _ in range(2):
        hsa = HybridSimulatedAnnealing(
            SamplerGaussian(seed=5),
            ProbabilityBoltzmannDownhill(seed=6),
            TemperatureExponential(5.0, 2),
            start_temperature=5.0,
            local_optimizer=None,
        )
        problem = Problem(_Sphere(), BoundaryConstraint(-10.0, 10.0), np.array([3.0, -3.0]))
        hsa.minimize(problem, _ec())
        results.append(problem.function_value)
    tolerance.exact(results[0], results[1])


# --------------------------------------------------------------------------
# Functor closed-form schedules
# --------------------------------------------------------------------------


def test_temperature_exponential_schedule() -> None:
    """TemperatureExponential: T_i(k) = T0_i * power^k."""
    temp = TemperatureExponential(10.0, 2, power=0.9)
    new_t = np.zeros(2, dtype=np.float64)
    curr_t = np.full(2, 10.0, dtype=np.float64)
    steps = np.array([3.0, 5.0], dtype=np.float64)
    temp(new_t, curr_t, steps)
    tolerance.tight(float(new_t[0]), 10.0 * 0.9**3, reason="exp schedule dim 0")
    tolerance.tight(float(new_t[1]), 10.0 * 0.9**5, reason="exp schedule dim 1")


def test_temperature_boltzmann_schedule() -> None:
    """TemperatureBoltzmann: T_i(k) = T0_i / log(k)."""
    temp = TemperatureBoltzmann(8.0, 1)
    new_t = np.zeros(1, dtype=np.float64)
    curr_t = np.array([8.0], dtype=np.float64)
    steps = np.array([math.e**2], dtype=np.float64)
    temp(new_t, curr_t, steps)
    tolerance.tight(float(new_t[0]), 8.0 / 2.0, reason="boltzmann schedule")


def test_temperature_cauchy_schedule() -> None:
    """TemperatureCauchy: T_i(k) = T0_i / k."""
    temp = TemperatureCauchy(6.0, 2)
    new_t = np.zeros(2, dtype=np.float64)
    curr_t = np.full(2, 6.0, dtype=np.float64)
    steps = np.array([2.0, 3.0], dtype=np.float64)
    temp(new_t, curr_t, steps)
    tolerance.tight(float(new_t[0]), 3.0, reason="cauchy schedule dim 0")
    tolerance.tight(float(new_t[1]), 2.0, reason="cauchy schedule dim 1")


def test_probability_always_downhill_logic() -> None:
    """AlwaysDownhill accepts iff new < current."""
    p = ProbabilityAlwaysDownhill()
    temp = np.array([1.0], dtype=np.float64)
    assert p(5.0, 3.0, temp) is True
    assert p(3.0, 5.0, temp) is False


def test_probability_boltzmann_downhill_always_accepts_improvement() -> None:
    """BoltzmannDownhill always accepts a strictly improving point."""
    p = ProbabilityBoltzmannDownhill(seed=1)
    temp = np.array([1.0, 1.0], dtype=np.float64)
    # new < current must always return True regardless of the RNG.
    for _ in range(20):
        assert p(10.0, 1.0, temp) is True


def test_reannealing_trivial_is_noop() -> None:
    """ReannealingTrivial leaves the steps untouched."""
    r = ReannealingTrivial()
    steps = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    before = steps.copy()
    r(steps, np.zeros(3, dtype=np.float64), 0.0, np.ones(3, dtype=np.float64))
    for j in range(3):
        tolerance.exact(float(steps[j]), float(before[j]))
