"""Problem bundle behavioral tests."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.problem import Problem


class _Sumsq(CostFunction):
    """Test fixture: residuals = identity; value = RMS of components."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()


def test_default_initial_value_is_empty() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    assert p.current_value.size == 0
    assert p.current_value.dtype == np.float64


def test_explicit_initial_value_round_trips() -> None:
    initial = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    p = Problem(_Sumsq(), NoConstraint(), initial)
    assert np.array_equal(p.current_value, initial)
    # Stored value is a copy — mutating the input does not leak.
    initial[0] = 99.0
    assert p.current_value[0] == 1.0


def test_function_value_and_gradient_norm_defaults_to_nan() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    assert math.isnan(p.function_value)
    assert math.isnan(p.gradient_norm_value)


def test_setters_round_trip() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    p.set_current_value(np.array([7.0, 8.0], dtype=np.float64))
    p.set_function_value(1.5)
    p.set_gradient_norm_value(0.25)
    assert np.array_equal(p.current_value, np.array([7.0, 8.0]))
    assert p.function_value == 1.5
    assert p.gradient_norm_value == 0.25


def test_value_increments_function_evaluation_counter() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    assert p.function_evaluation == 0
    p.value(np.array([3.0, 4.0], dtype=np.float64))
    assert p.function_evaluation == 1
    p.value(np.array([1.0, 1.0], dtype=np.float64))
    assert p.function_evaluation == 2


def test_values_increments_function_evaluation_counter() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    p.values(np.array([1.0, 2.0], dtype=np.float64))
    assert p.function_evaluation == 1


def test_gradient_increments_gradient_evaluation_counter() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    x = np.array([1.0, 1.0], dtype=np.float64)
    grad = np.zeros_like(x)
    p.gradient(grad, x)
    assert p.gradient_evaluation == 1
    # Function evaluations from the central-difference probe go through
    # the CostFunction directly (not the Problem), so the Problem-level
    # counter for that does NOT advance — matching C++.
    assert p.function_evaluation == 0


def test_value_and_gradient_increments_both_counters() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    x = np.array([1.0, 1.0], dtype=np.float64)
    grad = np.zeros_like(x)
    p.value_and_gradient(grad, x)
    assert p.function_evaluation == 1
    assert p.gradient_evaluation == 1


def test_reset_zeros_counters_and_nans_cached_state() -> None:
    p = Problem(_Sumsq(), NoConstraint())
    p.set_function_value(1.0)
    p.set_gradient_norm_value(2.0)
    p.value(np.array([0.0], dtype=np.float64))
    p.gradient(np.zeros(1), np.array([0.0]))
    p.reset()
    assert p.function_evaluation == 0
    assert p.gradient_evaluation == 0
    assert math.isnan(p.function_value)
    assert math.isnan(p.gradient_norm_value)


def test_constraint_and_cost_function_are_identity_preserved() -> None:
    cf = _Sumsq()
    constraint = NoConstraint()
    p = Problem(cf, constraint)
    assert p.cost_function is cf
    assert p.constraint is constraint
