"""Simplex (Nelder-Mead) behavioral + cross-validation tests.

Cross-validates the scipy-backed pquantlib port against the C++
``Simplex`` from v1.42.1 (probe at
``migration-harness/cpp/probes/l4a/foundations_probe.cpp``).

Tolerance choice: LOOSE for converged ``x`` values because scipy and
C++ Simplex implement the same Nelder-Mead but differ in their
contraction-step ordering and the precise condition for collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.math.optimization.simplex import Simplex
from pquantlib.testing import reference_reader, tolerance


class RosenbrockScalar(CostFunction):
    """Rosenbrock as a scalar cost (Simplex doesn't use the residual vector)."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])], dtype=np.float64)

    def value(self, x: npt.NDArray[np.float64]) -> float:
        r1 = 1.0 - x[0]
        r2 = 10.0 * (x[1] - x[0] * x[0])
        return r1 * r1 + r2 * r2


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return reference_reader.load("l4a/foundations")


def test_simplex_rosenbrock_converges_to_one_one(cpp_refs: dict[str, Any]) -> None:
    cf = RosenbrockScalar()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(10000, 1000, 1e-12, 1e-12, 1e-12)
    s = Simplex(lambda_=0.1)
    rc = s.minimize(problem, ec)
    cpp_x = cpp_refs["simplex"]["x"]
    # LOOSE: scipy and C++ Simplex both land within ~1e-12 of (1,1)
    # but on slightly different sides.
    tolerance.loose(float(problem.current_value[0]), float(cpp_x[0]))
    tolerance.loose(float(problem.current_value[1]), float(cpp_x[1]))
    # End criteria should be StationaryPoint (matching C++ exactly).
    cpp_rc = cpp_refs["simplex"]["end_criteria"]
    assert int(rc) == int(cpp_rc) == int(Type.StationaryPoint)


def test_simplex_rosenbrock_x_matches_unit_target() -> None:
    cf = RosenbrockScalar()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(10000, 1000, 1e-12, 1e-12, 1e-12)
    s = Simplex(lambda_=0.1)
    s.minimize(problem, ec)
    tolerance.loose(float(problem.current_value[0]), 1.0)
    tolerance.loose(float(problem.current_value[1]), 1.0)


def test_simplex_function_value_at_minimum_is_zero() -> None:
    cf = RosenbrockScalar()
    c = NoConstraint()
    x0 = np.array([-1.2, 1.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(10000, 1000, 1e-12, 1e-12, 1e-12)
    s = Simplex(lambda_=0.1)
    s.minimize(problem, ec)
    assert problem.function_value < 1e-20


def test_simplex_default_ctor_lambda_is_one() -> None:
    s = Simplex()
    assert s.lambda_ == 1.0


def test_simplex_custom_ctor_lambda() -> None:
    s = Simplex(lambda_=0.5)
    assert s.lambda_ == 0.5


def test_simplex_inherits_optimization_method() -> None:
    s = Simplex()
    assert isinstance(s, OptimizationMethod)


class _ScalarQuadratic(CostFunction):
    """Cost = (x-3)^2 with vector residual ``values = [x-3]``."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([x[0] - 3.0], dtype=np.float64)

    def value(self, x: npt.NDArray[np.float64]) -> float:
        return float((x[0] - 3.0) ** 2)


def test_simplex_simple_quadratic_converges() -> None:
    cf = _ScalarQuadratic()
    c = NoConstraint()
    x0 = np.array([0.0], dtype=np.float64)
    problem = Problem(cf, c, x0)
    ec = EndCriteria(5000, 500, 1e-12, 1e-12, 1e-12)
    s = Simplex(lambda_=0.5)
    s.minimize(problem, ec)
    tolerance.loose(float(problem.current_value[0]), 3.0)
