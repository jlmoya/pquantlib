"""Cross-validate LBFGSB — new in C++ QuantLib v1.43 — against the ``v143/lbfgsb`` probe.

L-BFGS-B's inner machinery (the generalized Cauchy point, the subspace
minimization, the compact representation and the Wolfe line search) lives in
an anonymous namespace in C++ and is unreachable from outside, so correctness
is pinned three ways:

- many bound and start-point configurations that exercise distinct code paths;
- a truncated-iteration sweep (``max_iterations`` = 3..12) that exposes the
  iterate sequence one step at a time;
- a full trace of every objective evaluation, in order.

Tolerance: LOOSE (1e-8). Iterative optimization accumulates rounding through
the line search and the compact-representation matrix inverse, so the
converged point matches C++ to roughly 1e-9 rather than to the last bits. The
discriminating assertions are the discrete ones — end-criteria type, evaluation
counters, trace length and the iterate trajectory — and those are exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.optimization.constraint import (
    NoConstraint,
    NonhomogeneousBoundaryConstraint,
)
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.lbfgsb import LBFGSB
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance

# --- reference --------------------------------------------------------------


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/lbfgsb")


def _arr(*values: float) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _full(n: int, value: float) -> npt.NDArray[np.float64]:
    return np.full(n, value, dtype=np.float64)


# --- objectives — mirror the probe's cost functions exactly -----------------


class RosenbrockFunction(CostFunction):
    """Extended Rosenbrock with analytic gradient; minimum 0 at (1, ..., 1)."""

    def value(self, x: npt.NDArray[np.float64]) -> float:
        f = 0.0
        for i in range(x.size - 1):
            f += 100.0 * (float(x[i + 1]) - float(x[i]) ** 2) ** 2 + (1.0 - float(x[i])) ** 2
        return f

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([self.value(x)], dtype=np.float64)

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        grad[:] = 0.0
        for i in range(x.size - 1):
            grad[i] += -400.0 * float(x[i]) * (float(x[i + 1]) - float(x[i]) ** 2) - 2.0 * (
                1.0 - float(x[i])
            )
            grad[i + 1] += 200.0 * (float(x[i + 1]) - float(x[i]) ** 2)

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        self.gradient(grad, x)
        return self.value(x)


class WeightedQuadratic(CostFunction):
    """Separable ``sum_i w_i (x_i - c_i)^2``; unconstrained minimum at ``c``."""

    def __init__(self, center: npt.NDArray[np.float64], weight: npt.NDArray[np.float64]) -> None:
        self._center: npt.NDArray[np.float64] = center
        self._weight: npt.NDArray[np.float64] = weight

    def value(self, x: npt.NDArray[np.float64]) -> float:
        f = 0.0
        for i in range(x.size):
            f += float(self._weight[i]) * (float(x[i]) - float(self._center[i])) ** 2
        return f

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([self.value(x)], dtype=np.float64)

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        for i in range(x.size):
            grad[i] = 2.0 * float(self._weight[i]) * (float(x[i]) - float(self._center[i]))

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        self.gradient(grad, x)
        return self.value(x)


class WeightedQuadraticValueOnly(WeightedQuadratic):
    """The same objective with no analytic gradient.

    Forces ``CostFunction``'s central-difference fallback. Those inner
    ``value`` calls bypass ``Problem``, so they do not move its counters —
    which is exactly what the reference's evaluation counts encode.
    """

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        CostFunction.gradient(self, grad, x)

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        return CostFunction.value_and_gradient(self, grad, x)


# --- shared setup — mirrors the probe --------------------------------------

STD_END_CRITERIA = EndCriteria(1000, 100, 1e-12, 1e-12, 1e-10)
MEMORY = 10
PG_TOL = 1e-10
F_TOL = 1e1 * QL_EPSILON
Q_CENTER = _arr(3.0, -2.0, 0.5)
Q_WEIGHT = _arr(1.0, 4.0, 0.25)
INF_BOUND = QL_MAX_REAL


def _check_result(cpp: dict[str, Any], prefix: str, problem: Problem, ec: Type) -> None:
    """Assert the converged state of one probe case."""
    expected_x = cpp[f"{prefix}_x"]
    actual_x = problem.current_value
    assert actual_x.size == len(expected_x)
    for i, expected in enumerate(expected_x):
        tolerance.loose(float(actual_x[i]), float(expected), reason=f"{prefix} x[{i}]")

    tolerance.loose(problem.function_value, float(cpp[f"{prefix}_f"]), reason=f"{prefix} f")
    # The problem caches pgInf SQUARED, not |g|^2. The probe records both the
    # cached value and the unsquared norm, so the relationship is checkable.
    tolerance.loose(
        problem.gradient_norm_value,
        float(cpp[f"{prefix}_grad_norm_value"]),
        reason=f"{prefix} gradient-norm value",
    )
    tolerance.loose(
        problem.gradient_norm_value,
        float(cpp[f"{prefix}_pg_inf_norm"]) ** 2,
        reason=f"{prefix} cached norm is the SQUARE of the projected-gradient inf norm",
    )

    assert ec.name == cpp[f"{prefix}_ec_name"]
    assert int(ec) == cpp[f"{prefix}_ec"]
    assert problem.function_evaluation == cpp[f"{prefix}_nfev"]
    assert problem.gradient_evaluation == cpp[f"{prefix}_ngev"]


def _run_bounded(
    cost: CostFunction,
    lo: npt.NDArray[np.float64],
    hi: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    *,
    memory: int = MEMORY,
    pg_tol: float = PG_TOL,
    f_tol: float = F_TOL,
    end_criteria: EndCriteria = STD_END_CRITERIA,
) -> tuple[Problem, Type]:
    problem = Problem(cost, NonhomogeneousBoundaryConstraint(lo, hi), x0)
    ec = LBFGSB(memory, pg_tol, f_tol).minimize(problem, end_criteria)
    return problem, ec


def _run_unconstrained(
    cost: CostFunction,
    x0: npt.NDArray[np.float64],
    *,
    memory: int = MEMORY,
    pg_tol: float = PG_TOL,
    f_tol: float = F_TOL,
    end_criteria: EndCriteria = STD_END_CRITERIA,
) -> tuple[Problem, Type]:
    problem = Problem(cost, NoConstraint(), x0)
    ec = LBFGSB(memory, pg_tol, f_tol).minimize(problem, end_criteria)
    return problem, ec


# --- constants --------------------------------------------------------------


def test_defaults_match_cpp(cpp: dict[str, Any]) -> None:
    tolerance.exact(1e-8, float(cpp["default_pg_tol"]))
    tolerance.exact(1e7 * QL_EPSILON, float(cpp["default_f_tol"]))
    assert cpp["default_memory"] == 10
    tolerance.exact(QL_EPSILON, float(cpp["ql_epsilon"]))
    tolerance.exact(QL_MAX_REAL, float(cpp["ql_max_real"]))
    tolerance.exact(0.5 * QL_MAX_REAL, float(cpp["no_bound_threshold"]))


def test_factr_is_the_divide_then_multiply_round_trip(cpp: dict[str, Any]) -> None:
    """``factr = f_tol / QL_EPSILON``, later multiplied by QL_EPSILON again.

    Collapsing the pair to ``f_tol * denom`` is not the same number.
    """
    tolerance.exact((1e7 * QL_EPSILON) / QL_EPSILON, float(cpp["default_factr"]))
    tolerance.exact(F_TOL / QL_EPSILON, float(cpp["tight_factr"]))
    tolerance.exact(F_TOL, float(cpp["tight_f_tol"]))


# --- unconstrained: with no active bounds L-BFGS-B is plain L-BFGS ----------


def test_rosenbrock_2d_unconstrained(cpp: dict[str, Any]) -> None:
    problem, ec = _run_unconstrained(RosenbrockFunction(), _full(2, -1.0))
    _check_result(cpp, "rosenbrock_2d_unconstrained", problem, ec)


def test_rosenbrock_10d_unconstrained(cpp: dict[str, Any]) -> None:
    problem, ec = _run_unconstrained(RosenbrockFunction(), _full(10, -1.0))
    _check_result(cpp, "rosenbrock_10d_unconstrained", problem, ec)


def test_rosenbrock_20d_small_memory(cpp: dict[str, Any]) -> None:
    """memory (3) < dimension (20): forces eviction and repeated rebuilds."""
    problem, ec = _run_unconstrained(
        RosenbrockFunction(), _full(20, -1.0), memory=3, pg_tol=1e-8
    )
    _check_result(cpp, "rosenbrock_20d_small_memory", problem, ec)


def test_rosenbrock_2d_memory_one(cpp: dict[str, Any]) -> None:
    """memory == 1: the minimal non-empty compact representation."""
    problem, ec = _run_unconstrained(RosenbrockFunction(), _full(2, -1.0), memory=1)
    _check_result(cpp, "rosenbrock_2d_memory_one", problem, ec)


# --- bound-constrained ------------------------------------------------------


def test_quadratic_interior_optimum(cpp: dict[str, Any]) -> None:
    problem, ec = _run_bounded(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT), _full(3, -10.0), _full(3, 10.0), _full(3, 0.0)
    )
    _check_result(cpp, "quadratic_interior_optimum", problem, ec)


def test_quadratic_active_bounds_with_default_constructor(cpp: dict[str, Any]) -> None:
    """``LBFGSB()`` == ``LBFGSB(10, 1e-8, 1e7 * QL_EPSILON)``.

    The operative stop is the KKT test (ZeroGradientNorm), not the factr
    fallback — a port that mixes the two lands on the same x with the wrong
    end-criteria type.
    """
    problem = Problem(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT),
        NonhomogeneousBoundaryConstraint(_full(3, 0.0), _full(3, 1.0)),
        _full(3, 0.5),
    )
    ec = LBFGSB().minimize(problem, STD_END_CRITERIA)
    _check_result(cpp, "quadratic_active_bounds_default_ctor", problem, ec)


def test_rosenbrock_2d_bounded(cpp: dict[str, Any]) -> None:
    problem, ec = _run_bounded(
        RosenbrockFunction(), _full(2, -2.0), _full(2, 0.5), _full(2, -1.0)
    )
    _check_result(cpp, "rosenbrock_2d_bounded", problem, ec)


def test_quadratic_all_active_corner(cpp: dict[str, Any]) -> None:
    """Every bound active: the free set is EMPTY (subspace step returns early)."""
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(5.0, 5.0, 5.0), _arr(1.0, 1.0, 1.0)),
        _full(3, 0.0),
        _full(3, 1.0),
        _full(3, 0.5),
    )
    _check_result(cpp, "quadratic_all_active_corner", problem, ec)


def test_quadratic_two_active_distinct_breakpoints(cpp: dict[str, Any]) -> None:
    """Disparate weights put the two upper bounds at distinct breakpoints.

    The Cauchy walk must therefore traverse more than one segment.
    """
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(10.0, 10.0), _arr(1.0, 100.0)),
        _full(2, 0.0),
        _full(2, 1.0),
        _arr(0.9, 0.1),
    )
    _check_result(cpp, "quadratic_two_active_distinct_breakpoints", problem, ec)


def test_quadratic_1d_active_bound(cpp: dict[str, Any]) -> None:
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(5.0), _arr(1.0)), _full(1, 0.0), _full(1, 1.0), _full(1, 0.5)
    )
    _check_result(cpp, "quadratic_1d_active_bound", problem, ec)


def test_quadratic_pinned_coordinate(cpp: dict[str, Any]) -> None:
    """A coordinate with ``low == high`` can never move."""
    problem, ec = _run_bounded(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT),
        _arr(-10.0, -10.0, 0.25),
        _arr(10.0, 10.0, 0.25),
        _full(3, 0.0),
    )
    _check_result(cpp, "quadratic_pinned_coordinate", problem, ec)


def test_quadratic_infeasible_start_is_clipped_before_first_evaluation(
    cpp: dict[str, Any],
) -> None:
    problem, ec = _run_bounded(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT), _full(3, 0.0), _full(3, 1.0), _full(3, 5.0)
    )
    _check_result(cpp, "quadratic_infeasible_start", problem, ec)


def test_quadratic_finite_difference_gradient(cpp: dict[str, Any]) -> None:
    """A value-only cost function differences inside itself, off the counters."""
    problem, ec = _run_bounded(
        WeightedQuadraticValueOnly(_arr(3.0, -2.0, 0.7), _arr(1.0, 4.0, 0.25)),
        _full(3, 0.0),
        _full(3, 1.0),
        _full(3, 0.5),
        pg_tol=1e-6,
        f_tol=1e7 * QL_EPSILON,
    )
    _check_result(cpp, "quadratic_finite_difference_gradient", problem, ec)
    # The counters advance once per Problem.value_and_gradient call and are not
    # inflated by the central differences the cost function runs internally.
    assert problem.function_evaluation == problem.gradient_evaluation


def test_quadratic_start_on_upper_bound_gradient_pushes_out(cpp: dict[str, Any]) -> None:
    """Start ON the bound the gradient pushes into: breakpoint ``t_i <= 0``."""
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(5.0, 5.0), _arr(1.0, 1.0)),
        _full(2, 0.0),
        _full(2, 1.0),
        _arr(1.0, 0.5),
    )
    _check_result(cpp, "quadratic_start_on_upper_bound_gradient_pushes_out", problem, ec)


def test_quadratic_start_on_lower_bound_gradient_pushes_out(cpp: dict[str, Any]) -> None:
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(-5.0, -5.0), _arr(1.0, 1.0)),
        _full(2, 0.0),
        _full(2, 1.0),
        _arr(0.0, 0.5),
    )
    _check_result(cpp, "quadratic_start_on_lower_bound_gradient_pushes_out", problem, ec)


def test_quadratic_start_at_optimum_corner(cpp: dict[str, Any]) -> None:
    """pgInf == 0 on the first check: zero iterations, one evaluation."""
    problem, ec = _run_bounded(
        WeightedQuadratic(_arr(5.0, 5.0), _arr(1.0, 1.0)),
        _full(2, 0.0),
        _full(2, 1.0),
        _full(2, 1.0),
    )
    _check_result(cpp, "quadratic_start_at_optimum_corner", problem, ec)
    assert problem.function_evaluation == 1


def test_quadratic_mixed_bounded_unbounded(cpp: dict[str, Any]) -> None:
    """Pins the ``u >= 0.5 * DBL_MAX`` sentinel test, not ``u == DBL_MAX``."""
    problem, ec = _run_bounded(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT),
        _arr(-INF_BOUND, 0.0, -INF_BOUND),
        _arr(INF_BOUND, 1.0, 0.25),
        _arr(0.0, 0.5, 0.0),
    )
    _check_result(cpp, "quadratic_mixed_bounded_unbounded", problem, ec)


# --- explicit bounds override the constraint --------------------------------


def test_explicit_bounds_override_an_unbounded_constraint(cpp: dict[str, Any]) -> None:
    problem = Problem(WeightedQuadratic(Q_CENTER, Q_WEIGHT), NoConstraint(), _full(3, 0.5))
    ec = LBFGSB(
        MEMORY, PG_TOL, F_TOL, lower_bound=_full(3, 0.0), upper_bound=_full(3, 1.0)
    ).minimize(problem, STD_END_CRITERIA)
    _check_result(cpp, "explicit_bounds_ctor_overrides_constraint", problem, ec)


def test_explicit_bounds_widen_a_narrower_constraint(cpp: dict[str, Any]) -> None:
    """The decisive case: constructor bounds OVERRIDE, they do not intersect.

    The constraint says [0,1]^3 and the constructor says [-10,10]^3; the run
    recovers the interior optimum (3, -2, 0.5), which an intersecting
    implementation could never reach.
    """
    problem = Problem(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT),
        NonhomogeneousBoundaryConstraint(_full(3, 0.0), _full(3, 1.0)),
        _full(3, 0.5),
    )
    ec = LBFGSB(
        MEMORY, PG_TOL, F_TOL, lower_bound=_full(3, -10.0), upper_bound=_full(3, 10.0)
    ).minimize(problem, STD_END_CRITERIA)
    _check_result(cpp, "explicit_bounds_ctor_widens_constraint", problem, ec)


# --- stopping criteria ------------------------------------------------------


def test_end_criteria_max_iterations_stop(cpp: dict[str, Any]) -> None:
    problem, ec = _run_unconstrained(
        RosenbrockFunction(),
        _full(10, -1.0),
        end_criteria=EndCriteria(3, 2, 1e-12, 1e-12, 1e-10),
    )
    _check_result(cpp, "rosenbrock_10d_max_iterations_stop", problem, ec)
    assert ec is Type.MaxIterations


def test_end_criteria_zero_gradient_norm_stop_is_fed_the_unsquared_norm(
    cpp: dict[str, Any],
) -> None:
    """``EndCriteria.check_zero_gradient_norm`` receives pgInf, not pgInf^2.

    ``gradient_norm_epsilon`` is 1e-2 while the optimizer's own ``pg_tol`` is
    1e-12, so the EndCriteria branch is what fires. Feeding it the squared norm
    would stop at a different — much tighter — point.
    """
    problem, ec = _run_bounded(
        WeightedQuadratic(Q_CENTER, Q_WEIGHT),
        _full(3, -10.0),
        _full(3, 10.0),
        _full(3, 0.0),
        pg_tol=1e-12,
        end_criteria=EndCriteria(1000, 100, 1e-12, 1e-12, 1e-2),
    )
    _check_result(cpp, "quadratic_endcriteria_gradient_norm_stop", problem, ec)


# --- iterate trajectory -----------------------------------------------------


def test_iterate_trajectory(cpp: dict[str, Any]) -> None:
    """Truncating at successive iteration caps exposes the iterate sequence.

    This is the closest observable proxy for the private Cauchy point and
    subspace minimization: a wrong inner loop diverges at the first iterate
    even when it still converges to the right answer eventually.
    """
    ks = cpp["rosenbrock_2d_bounded_iterate_trajectory_k"]
    n = int(cpp["rosenbrock_2d_bounded_iterate_trajectory_n"])
    xs = cpp["rosenbrock_2d_bounded_iterate_trajectory_x"]
    fs = cpp["rosenbrock_2d_bounded_iterate_trajectory_f"]
    ecs = cpp["rosenbrock_2d_bounded_iterate_trajectory_ec"]
    nfevs = cpp["rosenbrock_2d_bounded_iterate_trajectory_nfev"]
    assert len(ks) == 10

    for step, k in enumerate(ks):
        problem, ec = _run_bounded(
            RosenbrockFunction(),
            _full(2, -2.0),
            _full(2, 0.5),
            _full(2, -1.0),
            end_criteria=EndCriteria(int(k), 2, 1e-12, 1e-12, 1e-10),
        )
        for i in range(n):
            tolerance.loose(
                float(problem.current_value[i]),
                float(xs[n * step + i]),
                reason=f"trajectory k={k} x[{i}]",
            )
        tolerance.loose(problem.function_value, float(fs[step]), reason=f"trajectory k={k} f")
        assert int(ec) == ecs[step]
        assert problem.function_evaluation == nfevs[step]


# --- objective-evaluation traces -------------------------------------------


class _Tracing(CostFunction):
    """Records every ``value_and_gradient`` call, in order.

    LBFGSB reaches the objective only through ``Problem.value_and_gradient``,
    which dispatches to this single hook — so the recording is complete.
    """

    def __init__(self, inner: CostFunction) -> None:
        self._inner: CostFunction = inner
        self.points: list[npt.NDArray[np.float64]] = []
        self.function_values: list[float] = []

    def value(self, x: npt.NDArray[np.float64]) -> float:
        return self._inner.value(x)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._inner.values(x)

    def gradient(self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> None:
        self._inner.gradient(grad, x)

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        f = self._inner.value_and_gradient(grad, x)
        self.points.append(x.astype(np.float64, copy=True))
        self.function_values.append(f)
        return f


def _check_trace(cpp: dict[str, Any], prefix: str, traced: _Tracing) -> None:
    n = int(cpp[f"{prefix}_n"])
    count = int(cpp[f"{prefix}_count"])
    points = cpp[f"{prefix}_points"]
    values = cpp[f"{prefix}_values"]

    assert len(traced.points) == count
    for j in range(count):
        for i in range(n):
            tolerance.loose(
                float(traced.points[j][i]),
                float(points[n * j + i]),
                reason=f"{prefix} eval {j} x[{i}]",
            )
        tolerance.loose(
            traced.function_values[j], float(values[j]), reason=f"{prefix} eval {j} f"
        )


def test_evaluation_trace_quadratic_two_active(cpp: dict[str, Any]) -> None:
    traced = _Tracing(WeightedQuadratic(_arr(10.0, 10.0), _arr(1.0, 100.0)))
    problem, ec = _run_bounded(traced, _full(2, 0.0), _full(2, 1.0), _arr(0.9, 0.1))
    _check_result(cpp, "quadratic_two_active_eval_trace", problem, ec)
    _check_trace(cpp, "quadratic_two_active_eval_trace", traced)


def test_evaluation_trace_rosenbrock_bounded(cpp: dict[str, Any]) -> None:
    """Entry 0 is the clipped start point; the rest are line-search trials.

    Pins the search direction AND the Wolfe line search: a search that lands
    on the right answer by a different route shows up here and nowhere else.
    """
    traced = _Tracing(RosenbrockFunction())
    problem, ec = _run_bounded(traced, _full(2, -2.0), _full(2, 0.5), _full(2, -1.0))
    _check_result(cpp, "rosenbrock_2d_bounded_eval_trace", problem, ec)
    _check_trace(cpp, "rosenbrock_2d_bounded_eval_trace", traced)
    # Entry 0 is the START point, already clipped into [-2, 0.5]^2.
    assert traced.points[0].tolist() == [-1.0, -1.0]


# --- argument validation ----------------------------------------------------


def test_zero_memory_is_rejected(cpp: dict[str, Any]) -> None:
    assert cpp["validation_zero_memory_throws"] is True
    with pytest.raises(LibraryException, match="memory must be positive"):
        LBFGSB(0)


def test_mismatched_bound_sizes_are_rejected(cpp: dict[str, Any]) -> None:
    assert cpp["validation_mismatched_bounds_throws"] is True
    with pytest.raises(LibraryException, match="lower and upper bound sizes are inconsistent"):
        LBFGSB(10, lower_bound=_full(2, 0.0), upper_bound=_full(3, 1.0))


def test_bounds_of_the_wrong_dimension_are_rejected_by_minimize(cpp: dict[str, Any]) -> None:
    """C++ defers this one to ``minimize`` — the constructor cannot know ``n``."""
    assert cpp["validation_wrong_dimension_throws"] is True
    problem = Problem(
        WeightedQuadratic(_arr(1.0, 2.0), _arr(1.0, 1.0)), NoConstraint(), _full(2, 0.0)
    )
    optimizer = LBFGSB(lower_bound=_full(3, 0.0), upper_bound=_full(3, 1.0))
    with pytest.raises(LibraryException, match="bounds size does not match"):
        optimizer.minimize(problem, STD_END_CRITERIA)


def test_bounds_must_be_supplied_together() -> None:
    """Python has no two-argument constructor overload; the pair is enforced."""
    with pytest.raises(LibraryException, match="supplied together"):
        LBFGSB(10, lower_bound=_full(2, 0.0))
