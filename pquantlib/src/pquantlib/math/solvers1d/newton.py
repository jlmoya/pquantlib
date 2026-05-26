"""Newton 1-D solver.

# C++ parity: ql/math/solvers1d/newton.hpp (v1.42.1) class ``Newton``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.

The integrand must expose ``derivative(x) -> float``. If the iterate
jumps out of the bracket, control falls back to :class:`NewtonSafe`.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.solvers1d.newton_safe import NewtonSafe, derivative_of
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


class Newton(Solver1D):
    """Plain Newton — falls back to NewtonSafe if it jumps out of bracket."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        f_root = f(self._root)
        df_root = derivative_of(f, self._root)
        self._evaluation_number += 1

        while self._evaluation_number <= self._max_evaluations:
            dx = f_root / df_root
            self._root -= dx
            # jumped out of brackets, switch to NewtonSafe
            if (self._x_min - self._root) * (self._root - self._x_max) < 0.0:
                s = NewtonSafe()
                s.set_max_evaluations(self._max_evaluations - self._evaluation_number)
                return s.solve(f, x_accuracy, self._root + dx, self._x_min, self._x_max)
            if abs(dx) < x_accuracy:
                f(self._root)
                self._evaluation_number += 1
                return self._root
            f_root = f(self._root)
            df_root = derivative_of(f, self._root)
            self._evaluation_number += 1

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
