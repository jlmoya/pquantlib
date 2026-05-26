"""CostFunction abstract-class behavioral tests."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.cost_function import CostFunction


class _Quadratic(CostFunction):
    """Test fixture: residuals are coordinates themselves; values f = ||x||_RMS."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()


class _ExplicitValue(CostFunction):
    """Test fixture: override ``value`` directly (skip the RMS aggregation)."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x.copy()

    def value(self, x: npt.NDArray[np.float64]) -> float:
        return float(np.sum(x))


def test_cost_function_is_abstract() -> None:
    with pytest.raises(TypeError, match="Can't instantiate"):
        CostFunction()  # type: ignore[abstract]


def test_default_value_is_rms_of_residuals() -> None:
    cf = _Quadratic()
    x = np.array([3.0, 4.0], dtype=np.float64)
    # values = [3, 4]; mean(sq) = (9 + 16) / 2 = 12.5; sqrt = 3.5355...
    assert math.isclose(cf.value(x), math.sqrt(12.5), rel_tol=1e-15)


def test_override_value_takes_precedence() -> None:
    cf = _ExplicitValue()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert cf.value(x) == 6.0


def test_central_difference_gradient_for_quadratic() -> None:
    cf = _Quadratic()
    # f(x) = sqrt((x1^2 + x2^2 + ... + xn^2) / n)
    # df/dx_i at x = (x_i / n) / f(x); for x = (1, 1, 1) and n = 3:
    # f = 1; df/dx_i = 1/3.
    x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    grad = np.zeros_like(x)
    cf.gradient(grad, x)
    expected = np.full(3, 1.0 / 3.0)
    assert np.allclose(grad, expected, atol=1e-5)


def test_finite_difference_epsilon_default() -> None:
    cf = _Quadratic()
    assert cf.finite_difference_epsilon() == 1e-8
