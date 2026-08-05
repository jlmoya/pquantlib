"""CostFunction abstract-class behavioral tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.math.optimization.cost_function import (
    CostFunction,
    ParametersTransformation,
    SimpleCostFunction,
)
from pquantlib.testing import reference_reader, tolerance
from tests.math.optimization._cpp_cost_functions import RosenbrockResiduals, simple_values


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


def test_value_and_gradient_defaults_to_gradient_then_value() -> None:
    cf = _Quadratic()
    x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    grad = np.zeros_like(x)
    f = cf.value_and_gradient(grad, x)
    assert math.isclose(f, cf.value(x), rel_tol=1e-15)
    assert np.allclose(grad, np.full(3, 1.0 / 3.0), atol=1e-5)


def test_value_and_gradient_is_overridable_as_a_single_dispatch() -> None:
    class _OnePass(CostFunction):
        def __init__(self) -> None:
            self.calls: int = 0

        def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return x.copy()

        def value_and_gradient(
            self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
        ) -> float:
            self.calls += 1
            grad[:] = 2.0 * x
            return float(np.sum(x * x))

    cf = _OnePass()
    x = np.array([1.0, 2.0], dtype=np.float64)
    grad = np.zeros_like(x)
    assert cf.value_and_gradient(grad, x) == 5.0
    assert np.array_equal(grad, np.array([2.0, 4.0]))
    assert cf.calls == 1


# --- v1.43 additions: jacobian / values_and_jacobian / SimpleCostFunction ----


def _cpp_optimization() -> dict[str, Any]:

    return reference_reader.load("v143/math/optimization")


def test_default_jacobian_matches_cpp() -> None:
    """Central-difference Jacobian, ``jac[j][i] = d values_j / d x_i``.

    # C++ parity: costfunction.hpp:72-85.

    LOOSE tier, derived: a central difference with ``h = 1e-8`` divides a
    difference of two values by ``2h``, so a one-ulp difference in ``values``
    (relative 2^-53) shows up as an absolute error of ``|f| * 1.1e-16 / 2e-8
    = |f| * 5.5e-9``. The C++ probe is compiled with FP contraction, which
    changes ``x[0]*x[1] - 0.5`` by exactly one such ulp — visible in the
    ``scf_jacobian`` entry below at 1.7e-9 relative.
    """

    cpp = _cpp_optimization()
    f = RosenbrockResiduals()
    x = np.array(cpp["cf_x"], dtype=np.float64)
    jac = np.zeros((2, 2), dtype=np.float64)
    f.jacobian(jac, x)
    for got, expected in zip(jac.reshape(-1), cpp["cf_jacobian"], strict=True):
        tolerance.loose(float(got), float(expected))


def test_values_and_jacobian_matches_cpp() -> None:
    """# C++ parity: costfunction.hpp:89-93 — ``jacobian(jac, x); return values(x)``."""

    cpp = _cpp_optimization()
    f = RosenbrockResiduals()
    x = np.array(cpp["cf_x"], dtype=np.float64)
    jac = np.zeros((2, 2), dtype=np.float64)
    values = f.values_and_jacobian(jac, x)
    for got, expected in zip(values, cpp["cf_values_and_jacobian_values"], strict=True):
        tolerance.exact(float(got), float(expected))
    for got, expected in zip(jac.reshape(-1), cpp["cf_values_and_jacobian_jac"], strict=True):
        tolerance.loose(float(got), float(expected))


def test_default_value_and_gradient_match_cpp() -> None:
    """The RMS default ``value`` and the central-difference default ``gradient``."""

    cpp = _cpp_optimization()
    f = RosenbrockResiduals()
    x = np.array(cpp["cf_x"], dtype=np.float64)
    tolerance.exact(f.value(x), float(cpp["cf_value"]))
    grad = np.zeros(2, dtype=np.float64)
    f.gradient(grad, x)
    for got, expected in zip(grad, cpp["cf_gradient"], strict=True):
        tolerance.loose(float(got), float(expected))
    tolerance.exact(f.finite_difference_epsilon(), float(cpp["cf_finite_difference_epsilon"]))


def test_simple_cost_function_wraps_a_callable() -> None:
    """# C++ parity: costfunction.hpp:99-107 — the template that only supplies ``values``."""
    cpp = _cpp_optimization()
    scf = SimpleCostFunction(simple_values)
    x = np.array(cpp["scf_x"], dtype=np.float64)
    for got, expected in zip(scf.values(x), cpp["scf_values"], strict=True):
        tolerance.exact(float(got), float(expected))
    tolerance.exact(scf.value(x), float(cpp["scf_value"]))
    grad = np.zeros(2, dtype=np.float64)
    scf.gradient(grad, x)
    for got, expected in zip(grad, cpp["scf_gradient"], strict=True):
        tolerance.loose(float(got), float(expected))
    jac = np.zeros((3, 2), dtype=np.float64)
    scf.jacobian(jac, x)
    for got, expected in zip(jac.reshape(-1), cpp["scf_jacobian"], strict=True):
        tolerance.loose(float(got), float(expected))


def test_parameters_transformation_is_abstract() -> None:
    """# C++ parity: costfunction.hpp:109-114 — two pure-virtual methods."""

    with pytest.raises(TypeError):
        ParametersTransformation()  # type: ignore[abstract]

    class _Exp(ParametersTransformation):
        def direct(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            return np.exp(x)

        def inverse(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            return np.log(x)

    t = _Exp()
    x = np.array([0.5, 1.5], dtype=np.float64)
    assert np.allclose(t.inverse(t.direct(x)), x)
