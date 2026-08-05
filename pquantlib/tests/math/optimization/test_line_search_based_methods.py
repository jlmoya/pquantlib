"""Cross-validate SteepestDescent / ConjugateGradient / BFGS against C++ v1.43.

Reference: ``migration-harness/references/v143/math/optimization.json``,
block B. Twelve (method x line search) combinations are run over four
problems each, and three truncated-iteration trajectories pin the iterate
path step by step.

What each assertion claims
--------------------------
``EndCriteria.Type`` and the evaluation counters are DISCRETE: they either
match or the inner loop differs. They are asserted exactly wherever the
objective supplies an analytic gradient.

The converged point is asserted at TIGHT for every run that actually
converged. Two families are deliberately treated differently:

1. **Runs that stop on MaxIterations after ~1000 evaluations.** These are
   non-converging descents down the Rosenbrock valley; a last-bit difference
   in the objective is amplified by the iteration. They are asserted at
   ``custom(rel_tol=1e-9)``, derived below.

2. **Runs over ``RosenbrockResiduals``**, whose gradient is the
   ``CostFunction`` CENTRAL DIFFERENCE with ``h = 1e-8``. A one-ulp
   difference in ``value`` (relative 2^-53 ~ 1.1e-16) becomes an ABSOLUTE
   gradient error of ``|f| * 1.1e-16 / (2h) = |f| * 5.5e-9``. Over hundreds
   of iterations of a descent whose direction is that gradient, the iterate
   difference is not bounded by any tolerance that can be derived, so ``x``
   is NOT pinned for those. What IS pinned for them is the terminal
   ``EndCriteria.Type``, plus — in ``test_bfgs_dense_trajectory_matches_cpp``
   — the first several iterations of the path, which ARE bit-exact.

Where does the one-ulp difference come from? The C++ probe is compiled with
floating-point contraction enabled (clang's default), so ``a - b*c`` becomes
a single fused multiply-add with one rounding instead of two. Python and
numpy never contract. This is visible directly in the block-K reference:
the finite-difference jacobian entry for ``x[0]*x[1] - 0.5`` differs by
1.7e-9 relative, exactly the ``|f| * 1.1e-16 / (2h)`` figure above. It is
not something the port can remove, and it is not something to widen a
tolerance to hide — hence the split above.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.armijo import ArmijoLineSearch
from pquantlib.math.optimization.bfgs import BFGS
from pquantlib.math.optimization.conjugate_gradient import ConjugateGradient
from pquantlib.math.optimization.constraint import (
    Constraint,
    NoConstraint,
    PositiveConstraint,
)
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.goldstein import GoldsteinLineSearch
from pquantlib.math.optimization.line_search import LineSearch
from pquantlib.math.optimization.line_search_based_method import LineSearchBasedMethod
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.steepest_descent import SteepestDescent
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import (
    RosenbrockAnalytic,
    RosenbrockResiduals,
    WeightedQuadratic,
)

# Amplification of one ulp through ~1000 chaotic descent iterations on the
# Rosenbrock valley. See the module docstring.
_CHAOS_REL_TOL = 1e-9
_CHAOS_REASON = (
    "run terminates on MaxIterations after ~1000 objective evaluations; the C++ "
    "probe is compiled with FP contraction (a-b*c -> fma, one rounding instead "
    "of two) so the two sides differ by one ulp somewhere in the descent, and a "
    "non-converging Rosenbrock iteration amplifies that. The evaluation counters "
    "and the EndCriteria.Type still match exactly, which is what pins the path."
)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/optimization")


def _make_line_search(name: str) -> LineSearch:
    if name == "armijo":
        return ArmijoLineSearch()
    if name == "armijoc":
        return ArmijoLineSearch(1e-8, 0.2, 0.5)
    if name == "goldstein":
        return GoldsteinLineSearch()
    if name == "goldsteinc":
        return GoldsteinLineSearch(1e-8, 0.1, 0.7, 2.0)
    raise KeyError(name)


def _make_method(name: str, line_search: LineSearch) -> LineSearchBasedMethod:
    if name == "sd":
        return SteepestDescent(line_search)
    if name == "cg":
        return ConjugateGradient(line_search)
    if name == "bfgs":
        return BFGS(line_search)
    raise KeyError(name)


def _make_problem(
    key: str,
) -> tuple[CostFunction, Constraint, npt.NDArray[np.float64]]:
    if key == "rosen2d":
        return RosenbrockAnalytic(), NoConstraint(), np.array([-1.2, 1.0])
    if key == "quad3d":
        return (
            WeightedQuadratic(
                np.array([3.0, -2.0, 0.5]), np.array([1.0, 4.0, 0.25])
            ),
            NoConstraint(),
            np.zeros(3, dtype=np.float64),
        )
    if key == "rosenres":
        return RosenbrockResiduals(), NoConstraint(), np.array([-1.2, 1.0])
    if key == "quadpos":
        return (
            WeightedQuadratic(np.array([3.0, 2.0]), np.array([1.0, 4.0])),
            PositiveConstraint(),
            np.array([0.5, 0.5]),
        )
    raise KeyError(key)


def _std_end_criteria() -> EndCriteria:
    return EndCriteria(1000, 100, 1e-12, 1e-12, 1e-10)


def _case_names(cpp: dict[str, Any]) -> list[str]:
    names: list[str] = cpp["block_b_cases"]
    return names


def _run(name: str) -> tuple[Problem, Type]:
    _, method, search, problem_key = name.split("_")
    cost, constraint, x0 = _make_problem(problem_key)
    problem = Problem(cost, constraint, x0)
    om = _make_method(method, _make_line_search(search))
    return problem, om.minimize(problem, _std_end_criteria())


def test_block_b_case_list_is_complete(cpp: dict[str, Any]) -> None:
    """3 methods x 4 line searches x 4 problems."""
    assert len(_case_names(cpp)) == 48


@pytest.mark.parametrize(
    "name",
    reference_reader.load("v143/math/optimization")["block_b_cases"],
)
def test_end_criteria_type_matches_cpp(name: str, cpp: dict[str, Any]) -> None:
    """The terminal ``EndCriteria.Type`` is discrete and must match exactly."""
    _, ec_type = _run(name)
    assert int(ec_type) == cpp[f"{name}_ec"], (
        f"{name}: got {Type(int(ec_type)).name}, C++ says {cpp[f'{name}_ec_name']}"
    )


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in reference_reader.load("v143/math/optimization")["block_b_cases"]
        if not n.endswith("_rosenres")
    ],
)
def test_evaluation_counters_match_cpp(name: str, cpp: dict[str, Any]) -> None:
    """Same x with different counts means a different inner loop.

    Restricted to the analytic-gradient problems: with the central-difference
    gradient the iterate noise (see the module docstring) eventually changes
    how many backtracking steps the line search needs.
    """
    problem, _ = _run(name)
    assert problem.function_evaluation == cpp[f"{name}_nfev"]
    assert problem.gradient_evaluation == cpp[f"{name}_ngev"]


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in reference_reader.load("v143/math/optimization")["block_b_cases"]
        if not n.endswith("_rosenres")
    ],
)
def test_converged_point_matches_cpp(name: str, cpp: dict[str, Any]) -> None:
    """Converged point, objective and squared gradient norm.

    Runs that stopped on ``MaxIterations`` get the derived chaos tolerance;
    everything else is TIGHT.
    """
    problem, _ = _run(name)
    chaotic = cpp[f"{name}_ec"] == int(Type.MaxIterations)
    expected_x: list[float] = cpp[f"{name}_x"]
    for got, expected in zip(problem.current_value, expected_x, strict=True):
        if chaotic:
            tolerance.custom(
                float(got),
                float(expected),
                abs_tol=1e-14,
                rel_tol=_CHAOS_REL_TOL,
                reason=_CHAOS_REASON,
            )
        else:
            tolerance.tight(float(got), float(expected))
    if chaotic:
        tolerance.custom(
            problem.function_value,
            float(cpp[f"{name}_f"]),
            abs_tol=1e-14,
            rel_tol=_CHAOS_REL_TOL,
            reason=_CHAOS_REASON,
        )
        tolerance.custom(
            problem.gradient_norm_value,
            float(cpp[f"{name}_grad_norm_value"]),
            abs_tol=1e-14,
            rel_tol=_CHAOS_REL_TOL,
            reason=_CHAOS_REASON,
        )
    else:
        tolerance.tight(problem.function_value, float(cpp[f"{name}_f"]))
        tolerance.tight(
            problem.gradient_norm_value, float(cpp[f"{name}_grad_norm_value"])
        )


def _walk_trajectory(
    cpp: dict[str, Any],
    tag: str,
    problem_key: str,
    method: str,
    search: str,
    *,
    exact_upto: int | None = None,
) -> None:
    ks: list[int] = cpp[f"{tag}_k"]
    dim: int = cpp[f"{tag}_n"]
    xs: list[float] = cpp[f"{tag}_x"]
    for step, k in enumerate(ks):
        if exact_upto is not None and k > exact_upto:
            break
        cost, constraint, x0 = _make_problem(problem_key)
        problem = Problem(cost, constraint, x0)
        om = _make_method(method, _make_line_search(search))
        ec_type = om.minimize(
            problem, EndCriteria(int(k), 2, 1e-12, 1e-12, 1e-10)
        )
        for i in range(dim):
            tolerance.exact(
                float(problem.current_value[i]), float(xs[dim * step + i])
            )
        # TIGHT rather than EXACT: ``x`` is bit-identical, but the reported
        # objective can still differ in the last bit because the C++ probe
        # contracts ``x[i+1] - x[i]*x[i]`` into an fma. A one-ulp difference in
        # the objective does not flip any of the line search's threshold
        # comparisons, which is why the iterate itself stays exact.
        tolerance.tight(problem.function_value, float(cpp[f"{tag}_f"][step]))
        assert int(ec_type) == cpp[f"{tag}_ec"][step]
        assert problem.function_evaluation == cpp[f"{tag}_nfev"][step]
        assert problem.gradient_evaluation == cpp[f"{tag}_ngev"][step]


def test_conjugate_gradient_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """Iterate path of CG + Armijo on 2-D Rosenbrock, capped at k = 3..12.

    EXACT tier: with an analytic gradient the two implementations perform the
    same IEEE-754 operations in the same order, so every published iterate is
    bit-identical.
    """
    _walk_trajectory(cpp, "lsbm_cg_traj", "rosen2d", "cg", "armijo")


def test_bfgs_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """Iterate path of BFGS + Armijo on 2-D Rosenbrock, capped at k = 3..12.

    This is the strongest check on the inverse-Hessian update: the rank-two
    correction, the ``fac > sqrt(1e-8*sumdg*sumxi)`` skip test and the
    ``fae`` (not ``1/fae``) scaling all feed the very next iterate.
    """
    _walk_trajectory(cpp, "lsbm_bfgs_traj", "rosen2d", "bfgs", "armijo")


def test_steepest_descent_goldstein_trajectory_matches_cpp(
    cpp: dict[str, Any],
) -> None:
    """Iterate path of SteepestDescent + Goldstein on the 3-D quadratic.

    Pins the Goldstein bracket: the ``close_enough(tr, 0)`` expansion branch,
    the bisection branch, and the ``loopNumber`` counter that shares the outer
    ``EndCriteria`` cap.
    """
    _walk_trajectory(cpp, "lsbm_sd_gold_traj", "quad3d", "sd", "goldstein")


def test_conjugate_gradient_dense_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """CG + custom Armijo on Rosenbrock, k = 3..10, bit-exact.

    The dense sweep runs to k = 60 in the reference; the first ulp of
    disagreement appears at k = 11 and grows to 1.5e-14 by k = 60. Asserting
    bit-exactness up to k = 10 is what separates "the port takes the same
    path" from "the port converges to the same place".
    """
    _walk_trajectory(
        cpp,
        "lsbm_cg_armijoc_rosen2d_dense",
        "rosen2d",
        "cg",
        "armijoc",
        exact_upto=10,
    )


def test_bfgs_dense_trajectory_matches_cpp(cpp: dict[str, Any]) -> None:
    """BFGS + Armijo on the FINITE-DIFFERENCE Rosenbrock, k = 3..9, bit-exact.

    This is the family excluded from ``test_converged_point_matches_cpp``.
    Up to k = 9 the path is bit-identical; at k = 10 the central-difference
    gradient has amplified one ulp of ``value`` into 2.9e-7 of iterate (see
    the module docstring for the ``|f| * 1.1e-16 / (2h)`` derivation), so the
    trajectory is only pinned where it is determined.
    """
    _walk_trajectory(
        cpp,
        "lsbm_bfgs_armijo_rosenres_dense",
        "rosenres",
        "bfgs",
        "armijo",
        exact_upto=9,
    )


# --- structural / API behaviour ---------------------------------------------


def test_default_line_search_is_armijo() -> None:
    """A ``None`` line search selects a default ``ArmijoLineSearch``.

    # C++ parity: linesearchbasedmethod.cpp:29-33.
    """
    assert isinstance(SteepestDescent().line_search, ArmijoLineSearch)
    assert isinstance(ConjugateGradient().line_search, ArmijoLineSearch)
    assert isinstance(BFGS().line_search, ArmijoLineSearch)


def test_armijo_defaults() -> None:
    """# C++ parity: armijo.hpp:51-53 — eps 1e-8, alpha 0.05, beta 0.65."""
    ls = ArmijoLineSearch()
    assert ls.alpha == 0.05
    assert ls.beta == 0.65


def test_goldstein_defaults() -> None:
    """# C++ parity: goldstein.hpp:33-36 — alpha 0.05, beta 0.65, extrapolation 1.5."""
    ls = GoldsteinLineSearch()
    assert ls.alpha == 0.05
    assert ls.beta == 0.65
    assert ls.extrapolation == 1.5


def test_bfgs_inverse_hessian_starts_empty_and_becomes_identity() -> None:
    """# C++ parity: bfgs.cpp:29-36 — built lazily as the identity."""
    om = BFGS()
    assert om.inverse_hessian.shape == (0, 0)
    cost, constraint, x0 = _make_problem("quad3d")
    om.minimize(Problem(cost, constraint, x0), _std_end_criteria())
    assert om.inverse_hessian.shape == (3, 3)


def test_bfgs_inverse_hessian_persists_across_minimize_calls() -> None:
    """C++ only rebuilds when ``rows() == 0``, so a reused BFGS carries state.

    Documented in bfgs.py; asserted here so that "fixing" it would break a
    test rather than silently change behaviour.
    """
    om = BFGS()
    cost, constraint, x0 = _make_problem("quad3d")
    om.minimize(Problem(cost, constraint, x0), _std_end_criteria())
    first = om.inverse_hessian.copy()
    om.minimize(Problem(cost, constraint, x0), _std_end_criteria())
    assert not np.array_equal(first, om.inverse_hessian)


def test_line_search_based_method_is_an_optimization_method() -> None:
    assert isinstance(SteepestDescent(), OptimizationMethod)
    assert isinstance(ConjugateGradient(), OptimizationMethod)
    assert isinstance(BFGS(), OptimizationMethod)
    assert issubclass(SteepestDescent, LineSearchBasedMethod)
