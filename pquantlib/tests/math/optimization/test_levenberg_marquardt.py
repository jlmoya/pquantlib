"""LevenbergMarquardt behavioral + cross-validation tests.

Cross-validates the scipy-backed pquantlib port against the C++
``LevenbergMarquardt`` from v1.42.1 (probe at
``migration-harness/cpp/probes/l4a/foundations_probe.cpp``).

Tolerance choice: LOOSE for converged ``x`` values because scipy's
MINPACK and C++'s embedded MINPACK terminate on slightly different
stopping criteria (gtol vs ftol), so the iterate count differs by
a handful and the final x is within 1e-10 to 1e-12, not bit-exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.constraint import NoConstraint, PositiveConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.testing import reference_reader, tolerance


class RosenbrockResiduals(CostFunction):
    """Rosenbrock as a sum of two squared residuals.

    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2 = r1^2 + r2^2
    with r1 = 1 - x, r2 = 10 * (y - x^2).
    """

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])], dtype=np.float64)


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return reference_reader.load("l4a/foundations")


def test_lm_rosenbrock_converges_to_one_one(cpp_refs: dict[str, Any]) -> None:
    cf = RosenbrockResiduals()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(2000, 100, 1e-12, 1e-12, 1e-12)
    lm = LevenbergMarquardt(epsfcn=1e-8, xtol=1e-8, gtol=1e-8)
    rc = lm.minimize(problem, ec)
    cpp_x = cpp_refs["levenberg_marquardt"]["x"]
    # LOOSE because scipy/C++ MINPACK terminate slightly differently;
    # converged x should still match to within 1e-8.
    tolerance.loose(float(problem.current_value[0]), float(cpp_x[0]))
    tolerance.loose(float(problem.current_value[1]), float(cpp_x[1]))
    # f at the minimum is essentially zero either way.
    assert rc not in (Type.MaxIterations, Type.Unknown)


def test_lm_rosenbrock_x_matches_unit_target() -> None:
    cf = RosenbrockResiduals()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(2000, 100, 1e-12, 1e-12, 1e-12)
    lm = LevenbergMarquardt(epsfcn=1e-8, xtol=1e-8, gtol=1e-8)
    lm.minimize(problem, ec)
    tolerance.loose(float(problem.current_value[0]), 1.0)
    tolerance.loose(float(problem.current_value[1]), 1.0)


def test_lm_function_value_at_minimum_is_zero() -> None:
    cf = RosenbrockResiduals()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(2000, 100, 1e-12, 1e-12, 1e-12)
    lm = LevenbergMarquardt(epsfcn=1e-8, xtol=1e-8, gtol=1e-8)
    lm.minimize(problem, ec)
    # f = r^T r; at the minimum should be essentially zero.
    assert problem.function_value < 1e-20


def test_lm_returns_success_type() -> None:
    cf = RosenbrockResiduals()
    c = NoConstraint()
    x0 = np.array([0.5, 0.5], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(2000, 100, 1e-12, 1e-12, 1e-12)
    lm = LevenbergMarquardt()
    rc = lm.minimize(problem, ec)
    # Any of the success outcomes counts (StationaryFunctionValue,
    # StationaryPoint, ZeroGradientNorm). scipy reports gtol first
    # for this problem.
    assert rc in (
        Type.StationaryFunctionValue,
        Type.StationaryPoint,
        Type.ZeroGradientNorm,
    )


def test_lm_default_ctor_arguments() -> None:
    # C++ parity: levenbergmarquardt.hpp:51-54 — defaults are
    # epsfcn=1e-8, xtol=1e-8, gtol=1e-8.
    lm = LevenbergMarquardt()
    assert lm.epsfcn == 1e-8
    assert lm.xtol == 1e-8
    assert lm.gtol == 1e-8


def test_lm_custom_ctor_arguments() -> None:
    lm = LevenbergMarquardt(epsfcn=1e-10, xtol=1e-12, gtol=1e-9)
    assert lm.epsfcn == 1e-10
    assert lm.xtol == 1e-12
    assert lm.gtol == 1e-9


def test_lm_inherits_optimization_method() -> None:
    lm = LevenbergMarquardt()
    assert isinstance(lm, OptimizationMethod)


class _PenalizedQuadratic(CostFunction):
    """Quadratic centered on x=2 to exercise PositiveConstraint."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([x[0] - 2.0], dtype=np.float64)


def test_lm_with_positive_constraint() -> None:
    cf = _PenalizedQuadratic()
    c = PositiveConstraint()
    x0 = np.array([5.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(2000, 100, 1e-12, 1e-12, 1e-12)
    lm = LevenbergMarquardt()
    lm.minimize(problem, ec)
    # Minimum is at x=2, which is positive, so constraint shouldn't bind.
    tolerance.loose(float(problem.current_value[0]), 2.0)
