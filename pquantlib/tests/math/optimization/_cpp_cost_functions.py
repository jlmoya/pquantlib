"""Cost functions mirroring those in the v143_math_optimization C++ probe.

Each class here is a line-for-line Python twin of the corresponding class in
``migration-harness/cpp/probes/v143_math_optimization/probe.cpp``, so the
cross-validation tests rebuild the same objective rather than restating
literals from the reference JSON.

Keep the arithmetic in the same ORDER as the probe: these functions are
evaluated hundreds of times inside chaotic descent iterations, and a
re-associated expression changes the trajectory.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.least_square import LeastSquareProblem

# Exponential-fit data, verbatim from probe.cpp (kExpT / kExpY).
EXP_T: npt.NDArray[np.float64] = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float64
)
EXP_Y: npt.NDArray[np.float64] = np.array(
    [2.55, 2.05, 1.79, 1.44, 1.26, 1.02, 0.88, 0.71], dtype=np.float64
)


class RosenbrockAnalytic(CostFunction):
    """Extended Rosenbrock with an analytic gradient; minimum 0 at (1, ..., 1)."""

    def value(self, x: npt.NDArray[np.float64]) -> float:
        f = 0.0
        for i in range(x.size - 1):
            f += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
        return float(f)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([self.value(x)], dtype=np.float64)

    def gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> None:
        grad[:] = 0.0
        for i in range(x.size - 1):
            grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i])
            grad[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i])

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        self.gradient(grad, x)
        return self.value(x)


class WeightedQuadratic(CostFunction):
    """Separable quadratic ``sum_i w_i (x_i - c_i)^2`` with an analytic gradient."""

    def __init__(
        self, center: npt.NDArray[np.float64], weight: npt.NDArray[np.float64]
    ) -> None:
        self._center: npt.NDArray[np.float64] = center
        self._weight: npt.NDArray[np.float64] = weight

    def value(self, x: npt.NDArray[np.float64]) -> float:
        f = 0.0
        for i in range(x.size):
            f += self._weight[i] * (x[i] - self._center[i]) ** 2
        return float(f)

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([self.value(x)], dtype=np.float64)

    def gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> None:
        for i in range(x.size):
            grad[i] = 2.0 * self._weight[i] * (x[i] - self._center[i])

    def value_and_gradient(
        self, grad: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
    ) -> float:
        self.gradient(grad, x)
        return self.value(x)


class RosenbrockResiduals(CostFunction):
    """Rosenbrock as a residual vector, with NO overrides.

    Exercises the ``CostFunction`` defaults: ``value`` is
    ``sqrt(mean(values^2))`` and ``gradient`` / ``jacobian`` are central
    differences with ``finite_difference_epsilon() == 1e-8``.
    """

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r = np.empty(2, dtype=np.float64)
        r[0] = 1.0 - x[0]
        r[1] = 10.0 * (x[1] - x[0] * x[0])
        return r


class ExpFitResiduals(CostFunction):
    """``residual_i = a*exp(b*t_i) - y_i``; m = 8 residuals, n = 2 parameters."""

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r = np.empty(8, dtype=np.float64)
        for i in range(8):
            r[i] = x[0] * np.exp(x[1] * EXP_T[i]) - EXP_Y[i]
        return r


class ExpFitLSP(LeastSquareProblem):
    """The same exponential fit expressed as a ``LeastSquareProblem``."""

    def size(self) -> int:
        return 8

    def target_and_value(
        self,
        x: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        for i in range(8):
            target[i] = EXP_Y[i]
            fct2fit[i] = x[0] * np.exp(x[1] * EXP_T[i])

    def target_value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
        grad_fct2fit: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        fct2fit: npt.NDArray[np.float64],
    ) -> None:
        for i in range(8):
            e = np.exp(x[1] * EXP_T[i])
            target[i] = EXP_Y[i]
            fct2fit[i] = x[0] * e
            grad_fct2fit[i][0] = e
            grad_fct2fit[i][1] = x[0] * EXP_T[i] * e


def simple_values(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The free function wrapped by ``SimpleCostFunction`` in the probe."""
    r = np.empty(3, dtype=np.float64)
    r[0] = x[0] - 1.0
    r[1] = 2.0 * x[1] + x[0]
    r[2] = x[0] * x[1] - 0.5
    return r
