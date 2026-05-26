"""OptimizationMethod abstract-class behavioral tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem


def test_optimization_method_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        OptimizationMethod()  # type: ignore[abstract]


class _Identity(CostFunction):
    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()


class _NoopMinimizer(OptimizationMethod):
    """Smoke method: do nothing, return MaxIterations."""

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        del problem, end_criteria
        return Type.MaxIterations


def test_concrete_subclass_minimize_returns_outcome() -> None:
    method = _NoopMinimizer()
    problem = Problem(
        _Identity(),
        NoConstraint(),
        np.array([0.0], dtype=np.float64),
    )
    ec = EndCriteria(10, 5, 1e-8, 1e-9, 1e-7)
    outcome = method.minimize(problem, ec)
    assert outcome == Type.MaxIterations
