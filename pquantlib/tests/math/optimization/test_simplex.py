"""Cross-validate ``Simplex`` against C++ v1.43.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block C.

The converged POINT, the evaluation count and the ``EndCriteria.Type`` are
asserted EXACT. That is not optimism: ``Simplex`` only ever calls
``Problem.value`` and does arithmetic on vertex coordinates, so the whole
reflect / expand / contract / shrink sequence is reproduced operation for
operation — including the initial simplex built through
``Constraint.update``, the ``computeSimplexSize`` stopping rule, and both
exit paths. The reported f(x) is TIGHT rather than EXACT because the probe's
Rosenbrock objective is compiled with FP contraction; see the inline comment
in ``test_simplex_matches_cpp_exactly``.

This replaces an earlier scipy-backed implementation. Measured against this
same reference, that version agreed on the converged ``x`` to ~1e-12 on
well-conditioned problems but differed in function-evaluation count on 6 of
these 8 cases (319 vs 149 on the box-constrained one), followed a completely
different iterate path (max|dx| = 0.70 at a truncated cap of 20 iterations),
and landed 6.6e-5 away in ``x`` once ``root_epsilon`` was loosened to 1e-4 —
i.e. exactly where a real calibration would sit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    Constraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.simplex import Simplex
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import (
    RosenbrockAnalytic,
    RosenbrockResiduals,
    WeightedQuadratic,
)

# name -> (cost function, constraint, x0, lambda, EndCriteria args)
_CASES: dict[
    str,
    tuple[
        type[CostFunction] | Any,
        type[Constraint] | Any,
        list[float],
        float,
        tuple[int, int, float, float, float],
    ],
] = {
    # The exact setup the previous, scipy-backed test used.
    "simplex_rosenres_lambda0p1": (
        RosenbrockResiduals,
        NoConstraint,
        [-1.2, 1.0],
        0.1,
        (10000, 1000, 1e-12, 1e-12, 1e-12),
    ),
    "simplex_rosen2d_lambda0p1": (
        RosenbrockAnalytic,
        NoConstraint,
        [-1.2, 1.0],
        0.1,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_rosen2d_lambda1": (
        RosenbrockAnalytic,
        NoConstraint,
        [-1.2, 1.0],
        1.0,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_quad1d": (
        lambda: WeightedQuadratic(np.array([3.0]), np.array([1.0])),
        NoConstraint,
        [0.0],
        0.5,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_quad3d": (
        lambda: WeightedQuadratic(
            np.array([3.0, -2.0, 0.5]), np.array([1.0, 4.0, 0.25])
        ),
        NoConstraint,
        [0.0, 0.0, 0.0],
        1.0,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_quadpos": (
        lambda: WeightedQuadratic(np.array([3.0, 2.0]), np.array([1.0, 4.0])),
        PositiveConstraint,
        [0.5, 0.5],
        1.0,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_quad_boxed": (
        lambda: WeightedQuadratic(np.array([5.0, 5.0]), np.array([1.0, 1.0])),
        lambda: BoundaryConstraint(0.0, 1.0),
        [0.5, 0.5],
        0.25,
        (1000, 100, 1e-12, 1e-12, 1e-10),
    ),
    "simplex_rosen2d_loose_xtol": (
        RosenbrockAnalytic,
        NoConstraint,
        [-1.2, 1.0],
        0.1,
        (10000, 1000, 1e-4, 1e-12, 1e-12),
    ),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def _arr(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def test_case_list_matches_reference(cpp: dict[str, Any]) -> None:
    assert sorted(_CASES) == sorted(cpp["block_c_cases"])


@pytest.mark.parametrize("name", sorted(_CASES))
def test_simplex_matches_cpp_exactly(name: str, cpp: dict[str, Any]) -> None:
    """Converged point, objective, EndCriteria.Type and evaluation count."""
    make_cost, make_constraint, x0, lambda_, ec_args = _CASES[name]
    problem = Problem(make_cost(), make_constraint(), _arr(x0))
    ec_type = Simplex(lambda_=lambda_).minimize(problem, EndCriteria(*ec_args))

    for got, expected in zip(problem.current_value, cpp[f"{name}_x"], strict=True):
        tolerance.exact(float(got), float(expected))
    # ``x`` is bit-exact but the REPORTED objective is only TIGHT: the probe's
    # Rosenbrock ``value`` contains ``x[i+1] - x[i]*x[i]``, which clang
    # contracts into a single fused multiply-add (one rounding instead of
    # two). That last-bit difference never flips a simplex comparison, so the
    # vertex sequence is identical while the published f(x) can differ by an
    # ulp.
    tolerance.tight(problem.function_value, float(cpp[f"{name}_f"]))
    assert int(ec_type) == cpp[f"{name}_ec"]
    assert problem.function_evaluation == cpp[f"{name}_nfev"]
    # Simplex never touches the gradient.
    assert problem.gradient_evaluation == cpp[f"{name}_ngev"] == 0


def test_simplex_leaves_gradient_norm_at_the_null_sentinel(cpp: dict[str, Any]) -> None:
    """``Simplex`` never calls ``set_gradient_norm_value``.

    So ``Problem.gradient_norm_value`` still holds what ``reset()`` put there,
    which C++ defines as ``Null<Real>()`` == ``numeric_limits<float>::max()``.
    """
    make_cost, make_constraint, x0, lambda_, ec_args = _CASES["simplex_rosen2d_lambda1"]
    problem = Problem(make_cost(), make_constraint(), _arr(x0))
    Simplex(lambda_=lambda_).minimize(problem, EndCriteria(*ec_args))
    tolerance.exact(
        problem.gradient_norm_value,
        float(cpp["simplex_rosen2d_lambda1_grad_norm_value"]),
    )


def test_simplex_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """Iterate path under a truncated cap, k = 3..20, bit-exact.

    This is what distinguishes "converged to the same place" from "took the
    same path". The reflect/expand/contract decision at every step, and the
    ``factor``-halving that keeps trials feasible, are all in here.
    """
    ks: list[int] = cpp["simplex_traj_k"]
    dim: int = cpp["simplex_traj_n"]
    assert len(ks) == 18
    for step, k in enumerate(ks):
        problem = Problem(RosenbrockAnalytic(), NoConstraint(), _arr([-1.2, 1.0]))
        ec_type = Simplex(lambda_=0.1).minimize(
            problem, EndCriteria(int(k), 2, 1e-12, 1e-12, 1e-10)
        )
        for i in range(dim):
            tolerance.exact(
                float(problem.current_value[i]),
                float(cpp["simplex_traj_x"][dim * step + i]),
            )
        # TIGHT for f, EXACT for x — see test_simplex_matches_cpp_exactly.
        tolerance.tight(problem.function_value, float(cpp["simplex_traj_f"][step]))
        assert int(ec_type) == cpp["simplex_traj_ec"][step]
        assert problem.function_evaluation == cpp["simplex_traj_nfev"][step]


def test_simplex_rejects_an_infeasible_starting_point() -> None:
    """# C++ parity: simplex.cpp:92-93 — ``QL_FAIL`` before any evaluation."""
    problem = Problem(
        WeightedQuadratic(np.array([3.0]), np.array([1.0])),
        PositiveConstraint(),
        _arr([-1.0]),
    )
    with pytest.raises(LibraryException, match="not in the feasible region"):
        Simplex(lambda_=1.0).minimize(problem, EndCriteria(100, 10, 1e-8, 1e-8, 1e-8))


def test_simplex_returns_stationary_point_on_the_normal_exit(
    cpp: dict[str, Any],
) -> None:
    """The seeded-counter idiom always yields ``StationaryPoint``.

    ``checkStationaryPoint(0, 0, maxStationaryStateIterations, ecType)``
    pre-increments a counter seeded with the maximum, so the criterion fires
    unconditionally (simplex.cpp:151).
    """
    assert cpp["simplex_rosen2d_lambda1_ec"] == int(Type.StationaryPoint)


def test_simplex_lambda_accessor() -> None:
    assert Simplex(lambda_=0.25).lambda_ == 0.25
    # pquantlib keeps a default; C++ has none (``Simplex(Real lambda)``).
    assert Simplex().lambda_ == 1.0


def test_simplex_is_an_optimization_method() -> None:
    assert isinstance(Simplex(0.1), OptimizationMethod)
