"""Cross-validate ``SimulatedAnnealing`` against C++ v1.43.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block F.

The method is stochastic, so the whole point of these tests is that it is
also *reproducible*: every perturbation is ``-T * log(u)`` with ``u`` drawn
from QuantLib's own ``MersenneTwisterUniformRng``. If the port consumed a
different NUMBER of draws, or consumed them in a different ORDER, the streams
would desynchronise and nothing downstream would line up — so the exactly
matching function-evaluation count is the strongest single assertion here.

``EndCriteria.Type`` and the evaluation count are asserted exactly; the
converged point and objective at TIGHT, because the probe's ``value``
routines are compiled with FP contraction (``a - b*c`` -> fma, one rounding
instead of two) which leaves a last-bit difference that the annealing
iteration carries forward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.constraint import (
    Constraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.simulated_annealing import Scheme, SimulatedAnnealing
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import (
    RosenbrockAnalytic,
    WeightedQuadratic,
)


def _arr(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _case(
    name: str,
) -> tuple[
    CostFunction, Constraint, npt.NDArray[np.float64], SimulatedAnnealing, EndCriteria
]:
    if name == "sa_constantfactor_rosen2d":
        return (
            RosenbrockAnalytic(),
            NoConstraint(),
            _arr([-1.2, 1.0]),
            SimulatedAnnealing.constant_factor(
                0.1, 10.0, 0.99, 10, MersenneTwisterUniformRng(42)
            ),
            EndCriteria(500, 100, 1e-8, 1e-12, 1e-12),
        )
    if name == "sa_constantbudget_rosen2d":
        return (
            RosenbrockAnalytic(),
            NoConstraint(),
            _arr([-1.2, 1.0]),
            SimulatedAnnealing.constant_budget(
                0.1, 10.0, 200, 1.0, MersenneTwisterUniformRng(42)
            ),
            EndCriteria(500, 100, 1e-8, 1e-12, 1e-12),
        )
    if name == "sa_constantfactor_quad3d":
        return (
            WeightedQuadratic(np.array([3.0, -2.0, 0.5]), np.array([1.0, 4.0, 0.25])),
            NoConstraint(),
            _arr([0.0, 0.0, 0.0]),
            SimulatedAnnealing.constant_factor(
                1.0, 5.0, 0.95, 5, MersenneTwisterUniformRng(7)
            ),
            EndCriteria(400, 100, 1e-8, 1e-12, 1e-12),
        )
    if name == "sa_constantbudget_quadpos":
        return (
            WeightedQuadratic(np.array([3.0, 2.0]), np.array([1.0, 4.0])),
            PositiveConstraint(),
            _arr([0.5, 0.5]),
            SimulatedAnnealing.constant_budget(
                1.0, 5.0, 150, 2.0, MersenneTwisterUniformRng(3)
            ),
            EndCriteria(400, 100, 1e-8, 1e-12, 1e-12),
        )
    if name == "sa_zero_temperature":
        return (
            RosenbrockAnalytic(),
            NoConstraint(),
            _arr([-1.2, 1.0]),
            SimulatedAnnealing.constant_factor(
                0.1, 0.0, 0.9, 4, MersenneTwisterUniformRng(5)
            ),
            EndCriteria(300, 100, 1e-8, 1e-12, 1e-12),
        )
    raise KeyError(name)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def test_case_list_matches_reference(cpp: dict[str, Any]) -> None:
    assert cpp["block_f_cases"] == [
        "sa_constantfactor_rosen2d",
        "sa_constantbudget_rosen2d",
        "sa_constantfactor_quad3d",
        "sa_constantbudget_quadpos",
        "sa_zero_temperature",
    ]


@pytest.mark.parametrize(
    "name",
    reference_reader.load("v143/math/optimization")["block_f_cases"],
)
def test_simulated_annealing_matches_cpp(name: str, cpp: dict[str, Any]) -> None:
    cost, constraint, x0, sa, ec = _case(name)
    problem = Problem(cost, constraint, x0)
    ec_type = sa.minimize(problem, ec)

    # Discrete: the RNG stream and the control flow either agree or they don't.
    assert int(ec_type) == cpp[f"{name}_ec"], (
        f"{name}: C++ says {cpp[f'{name}_ec_name']}"
    )
    assert problem.function_evaluation == cpp[f"{name}_nfev"]
    assert problem.gradient_evaluation == cpp[f"{name}_ngev"] == 0

    for got, expected in zip(problem.current_value, cpp[f"{name}_x"], strict=True):
        tolerance.tight(float(got), float(expected))
    tolerance.tight(problem.function_value, float(cpp[f"{name}_f"]))


def test_simulated_annealing_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """Best-ever point under a truncated iteration cap, k = 5, 10, ... 40.

    The iteration counter advances by 2 per reflection, by 1 more on a
    contraction, and by ``n`` more on a full shrink, so the cap lands
    mid-schedule and pins the annealing bookkeeping, not just the endpoint.
    """
    ks: list[int] = cpp["sa_traj_k"]
    dim: int = cpp["sa_traj_n"]
    for step, k in enumerate(ks):
        problem = Problem(RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0]))
        sa = SimulatedAnnealing.constant_factor(
            0.1, 10.0, 0.99, 10, MersenneTwisterUniformRng(42)
        )
        ec_type = sa.minimize(problem, EndCriteria(int(k), 2, 1e-8, 1e-12, 1e-12))
        for i in range(dim):
            tolerance.tight(
                float(problem.current_value[i]),
                float(cpp["sa_traj_x"][dim * step + i]),
            )
        tolerance.tight(problem.function_value, float(cpp["sa_traj_f"][step]))
        assert int(ec_type) == cpp["sa_traj_ec"][step]
        assert problem.function_evaluation == cpp["sa_traj_nfev"][step]


def test_constant_factor_and_constant_budget_select_different_schemes() -> None:
    """The two classmethods mirror the two C++ constructors, and differ.

    ConstantFactor runs ``m`` moves per temperature step, ConstantBudget one;
    with the same seed and the same problem the two therefore consume the RNG
    differently and land somewhere else. Asserted behaviourally because C++
    exposes no accessor for ``scheme_`` either.
    """
    assert SimulatedAnnealing.Scheme is Scheme
    assert list(Scheme) == [Scheme.ConstantFactor, Scheme.ConstantBudget]
    ec = EndCriteria(60, 10, 1e-8, 1e-12, 1e-12)
    results: list[npt.NDArray[np.float64]] = []
    for sa in (
        SimulatedAnnealing.constant_factor(
            0.1, 1.0, 0.5, 3, MersenneTwisterUniformRng(1)
        ),
        SimulatedAnnealing.constant_budget(
            0.1, 1.0, 10, 2.0, MersenneTwisterUniformRng(1)
        ),
    ):
        problem = Problem(RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0]))
        sa.minimize(problem, ec)
        results.append(problem.current_value.copy())
    assert not np.array_equal(results[0], results[1])


def test_rng_defaults_to_a_clock_seeded_mersenne_twister() -> None:
    """# C++ parity: simulatedannealing.hpp:58 — ``const RNG& rng = RNG()``.

    ``MersenneTwisterUniformRng()`` with the default seed 0 routes through
    ``SeedGenerator``, so the default-constructed optimizer is deliberately
    NOT reproducible — same as C++.
    """
    sa = SimulatedAnnealing.constant_factor(0.1, 1.0, 0.5, 3)
    problem = Problem(RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0]))
    # It runs, and it consumes the objective — i.e. a generator was supplied.
    sa.minimize(problem, EndCriteria(20, 5, 1e-8, 1e-12, 1e-12))
    assert problem.function_evaluation > 0


def test_rng_is_copied_not_shared() -> None:
    """C++ stores ``const RNG rng_`` BY VALUE, so two optimizers are independent."""
    rng = MersenneTwisterUniformRng(99)
    a = SimulatedAnnealing.constant_factor(0.1, 1.0, 0.5, 3, rng)
    b = SimulatedAnnealing.constant_factor(0.1, 1.0, 0.5, 3, rng)
    problem_a = Problem(
        RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0])
    )
    problem_b = Problem(
        RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0])
    )
    ec = EndCriteria(60, 10, 1e-8, 1e-12, 1e-12)
    a.minimize(problem_a, ec)
    b.minimize(problem_b, ec)
    for x, y in zip(problem_a.current_value, problem_b.current_value, strict=True):
        tolerance.exact(float(x), float(y))


def test_simulated_annealing_is_an_optimization_method() -> None:
    sa = SimulatedAnnealing.constant_factor(
        0.1, 1.0, 0.5, 3, MersenneTwisterUniformRng(1)
    )
    assert isinstance(sa, OptimizationMethod)
