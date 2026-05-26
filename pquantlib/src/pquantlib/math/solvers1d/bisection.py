"""Bisection 1-D solver.

# C++ parity: ql/math/solvers1d/bisection.hpp (v1.42.1) class ``Bisection``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


class Bisection(Solver1D):
    """Classic bisection root finder. Linear convergence, always robust."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # Orient the search so that f>0 lies at root_+dx
        if self._fx_min < 0.0:
            dx = self._x_max - self._x_min
            self._root = self._x_min
        else:
            dx = self._x_min - self._x_max
            self._root = self._x_max

        while self._evaluation_number <= self._max_evaluations:
            dx /= 2.0
            x_mid = self._root + dx
            f_mid = f(x_mid)
            self._evaluation_number += 1
            if f_mid <= 0.0:
                self._root = x_mid
            if abs(dx) < x_accuracy or close(f_mid, 0.0):
                f(self._root)
                self._evaluation_number += 1
                return self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
