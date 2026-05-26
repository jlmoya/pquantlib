"""Halley 1-D solver.

# C++ parity: ql/math/solvers1d/halley.hpp (v1.42.1) class ``Halley``.

Cubic-rate Newton-like update using both first and second derivatives.
The integrand must expose ``derivative(x) -> float`` AND
``second_derivative(x) -> float``. Falls back to :class:`NewtonSafe`
if the iterate jumps out of the bracket.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.solvers1d.newton_safe import NewtonSafe, derivative_of
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


def _second_derivative(f: RealFunction, x: float) -> float:
    """Call ``f.second_derivative(x)`` with a clear error if absent.

    # C++ parity: halley.hpp expects ``f.secondDerivative(root_)``.
    """
    sd = getattr(f, "second_derivative", None)
    qassert.require(callable(sd), "Halley requires function's second derivative")
    return float(sd(x))  # type: ignore[misc]


class Halley(Solver1D):
    """Halley's method — cubic-rate convergence using first + second derivatives."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        while True:
            self._evaluation_number += 1
            if self._evaluation_number > self._max_evaluations:
                break
            fx = f(self._root)
            f_prime = derivative_of(f, self._root)
            lf = fx * _second_derivative(f, self._root) / (f_prime * f_prime)
            step = 1.0 / (1.0 - 0.5 * lf) * fx / f_prime
            self._root -= step

            # jumped out of brackets, switch to NewtonSafe
            if (self._x_min - self._root) * (self._root - self._x_max) < 0.0:
                s = NewtonSafe()
                s.set_max_evaluations(self._max_evaluations - self._evaluation_number)
                return s.solve(f, x_accuracy, self._root + step, self._x_min, self._x_max)

            if abs(step) < x_accuracy:
                f(self._root)
                self._evaluation_number += 1
                return self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
