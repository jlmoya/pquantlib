"""C++ v1.43 reference pins for ``LevenbergMarquardt`` — currently DIVERGENT.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block D.

pquantlib's ``LevenbergMarquardt`` delegates to
``scipy.optimize.least_squares(method="lm")``. C++'s is a port of MINPACK's
``lmdif`` (ql/math/optimization/lmdif.cpp, ~1700 lines) driven by QuantLib's
own ``EndCriteria``. SciPy wraps MINPACK too, but through a different driver
with a different iteration budget and a different status mapping, so the two
are not the same optimizer. Measured against this reference:

* ``Problem.function_value`` is reported differently. C++ publishes
  ``costFunction().value(x)``, i.e. ``sqrt(mean(r^2))``
  (levenbergmarquardt.cpp:137); pquantlib publishes ``r . r``. On the 8-point
  exponential fit that is 0.00739 against 0.03040 — a factor of 4.1, and
  unbounded in general (the ratio is ``sqrt(m * f)`` for residual norm f).
* the ``EndCriteria.Type`` mapping is wrong in principle, not just in detail.
  C++ maps MINPACK ``info`` 1-4 to ``StationaryFunctionValue``, 5 to
  ``MaxIterations`` and 6 to ``FunctionEpsilon``, so it can NEVER return
  ``ZeroGradientNorm`` or ``StationaryPoint``. pquantlib returns
  ``ZeroGradientNorm`` on Rosenbrock and ``StationaryPoint`` on the
  constrained fit.
* the iteration budget is off by a factor of (n+1). C++ passes
  ``maxfev = maxIterations * (n + 1)`` to ``lmdif``
  (levenbergmarquardt.cpp:58) while pquantlib passes ``max_nfev =
  max_iterations``. The truncated-iteration trajectory shows the resulting
  one-step lag directly: pquantlib at k = 5 sits where C++ was at k = 3.
* the infeasible-point penalty differs. C++ fills every residual with 1e10
  (levenbergmarquardt.cpp:165); pquantlib uses 1e30, which changes the
  finite-difference Jacobian near the boundary and moves the constrained
  optimum by 1.1e-4.
* ``epsfcn`` is not ``diff_step``. MINPACK's ``fdjac2`` uses
  ``h = sqrt(max(epsfcn, eps)) * |x_j|``; SciPy's ``diff_step`` is a plain
  relative multiplier.
* even the converged point misses LOOSE on a real fitting problem: 1.8e-7 in
  ``x`` on the exponential fit (it happens to agree exactly on Rosenbrock
  only because both land on (1, 1) to the last bit).

These tests are ``xfail(strict=True)``. They document the target behaviour
and will start FAILING — loudly — the moment ``lmdif`` is ported and the
delegation is replaced, which is exactly the signal wanted.
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
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import (
    ExpFitResiduals,
    RosenbrockResiduals,
)

_CASES: dict[
    str,
    tuple[
        type[CostFunction],
        type[Constraint],
        list[float],
        tuple[float, float, float],
        tuple[int, int, float, float, float],
    ],
] = {
    "lm_rosenres": (
        RosenbrockResiduals,
        NoConstraint,
        [-1.2, 1.0],
        (1e-8, 1e-8, 1e-8),
        (2000, 100, 1e-12, 1e-12, 1e-12),
    ),
    "lm_rosenres_default": (
        RosenbrockResiduals,
        NoConstraint,
        [0.5, 0.5],
        (1e-8, 1e-8, 1e-8),
        (2000, 100, 1e-12, 1e-12, 1e-12),
    ),
    "lm_expfit": (
        ExpFitResiduals,
        NoConstraint,
        [1.0, -1.0],
        (1e-8, 1e-8, 1e-8),
        (2000, 100, 1e-12, 1e-12, 1e-12),
    ),
    "lm_expfit_positive": (
        ExpFitResiduals,
        PositiveConstraint,
        [1.0, 0.5],
        (1e-8, 1e-8, 1e-8),
        (2000, 100, 1e-12, 1e-12, 1e-12),
    ),
    "lm_expfit_maxiter": (
        ExpFitResiduals,
        NoConstraint,
        [1.0, -1.0],
        (1e-8, 1e-8, 1e-8),
        (3, 2, 1e-12, 1e-12, 1e-12),
    ),
    "lm_expfit_loose_ftol": (
        ExpFitResiduals,
        NoConstraint,
        [1.0, -1.0],
        (1e-8, 1e-8, 1e-8),
        (2000, 100, 1e-12, 1e-3, 1e-12),
    ),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def _arr(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _run(name: str) -> tuple[Problem, Type]:
    make_cost, make_constraint, x0, lm_args, ec_args = _CASES[name]
    problem = Problem(make_cost(), make_constraint(), _arr(x0))
    lm = LevenbergMarquardt(*lm_args)
    return problem, lm.minimize(problem, EndCriteria(*ec_args))


def test_reference_covers_every_case(cpp: dict[str, Any]) -> None:
    """The C++ reference also carries a ``useCostFunctionsJacobian`` case.

    pquantlib's ``use_cost_functions_jacobian`` flag is accepted and ignored
    (SciPy's ``lm`` driver has no analytic-Jacobian hook), so there is nothing
    to run against it here — it is listed as a gap, not tested.
    """
    assert "lm_expfit_costjac" in cpp["block_d_cases"]
    assert set(_CASES) < set(cpp["block_d_cases"])


@pytest.mark.xfail(
    strict=True,
    reason="pquantlib LevenbergMarquardt is a scipy delegation, not a port of "
    "MINPACK lmdif; see the module docstring for the measured divergences",
)
@pytest.mark.parametrize("name", sorted(_CASES))
def test_levenberg_marquardt_matches_cpp(name: str, cpp: dict[str, Any]) -> None:
    """Target behaviour once ``lmdif`` is ported."""
    problem, ec_type = _run(name)
    for got, expected in zip(problem.current_value, cpp[f"{name}_x"], strict=True):
        tolerance.loose(float(got), float(expected))
    tolerance.loose(problem.function_value, float(cpp[f"{name}_f"]))
    assert int(ec_type) == cpp[f"{name}_ec"]
    assert problem.function_evaluation == cpp[f"{name}_nfev"]


@pytest.mark.xfail(
    strict=True,
    reason="pquantlib publishes r.r; C++ publishes costFunction().value(x) == "
    "sqrt(mean(r^2)) (levenbergmarquardt.cpp:137)",
)
def test_reported_function_value_is_the_cost_functions_own_value(
    cpp: dict[str, Any],
) -> None:
    """Independent of the algorithm: the published f(x) is the wrong quantity.

    This one is a one-line fix and does not need the ``lmdif`` port.
    """
    problem, _ = _run("lm_expfit")
    tolerance.loose(problem.function_value, float(cpp["lm_expfit_f"]))


@pytest.mark.xfail(
    strict=True,
    reason="C++ maps every MINPACK success (info 1-4) to StationaryFunctionValue "
    "(levenbergmarquardt.cpp:117-134); pquantlib forwards scipy's status",
)
def test_successful_stop_is_always_stationary_function_value(
    cpp: dict[str, Any],
) -> None:
    """Also independent of the algorithm: the status translation table is wrong."""
    _, ec_type = _run("lm_rosenres")
    assert ec_type is Type.StationaryFunctionValue
    assert cpp["lm_rosenres_ec"] == int(Type.StationaryFunctionValue)


def test_cpp_never_reports_zero_gradient_norm_or_stationary_point(
    cpp: dict[str, Any],
) -> None:
    """Property of the C++ switch, asserted straight off the reference.

    Not xfail: this checks the REFERENCE, so it stays green either way and
    documents what the ported mapping has to produce.
    """
    forbidden = {int(Type.ZeroGradientNorm), int(Type.StationaryPoint)}
    for name in cpp["block_d_cases"]:
        assert cpp[f"{name}_ec"] not in forbidden, name
