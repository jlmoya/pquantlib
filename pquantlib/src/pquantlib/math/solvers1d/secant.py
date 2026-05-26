"""Secant 1-D solver.

# C++ parity: ql/math/solvers1d/secant.hpp (v1.42.1) class ``Secant``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.

Note: secant uses a 2-point linear extrapolation rather than a bracketed
search, so the iterate is not guaranteed to remain in [xMin_, xMax_].
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


class Secant(Solver1D):
    """Secant root finder — linear extrapolation through the last two iterates."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # Pick the bound with the smaller function value as the most recent guess
        if abs(self._fx_min) < abs(self._fx_max):
            self._root = self._x_min
            f_root = self._fx_min
            xl = self._x_max
            fl = self._fx_max
        else:
            self._root = self._x_max
            f_root = self._fx_max
            xl = self._x_min
            fl = self._fx_min

        while self._evaluation_number <= self._max_evaluations:
            dx = (xl - self._root) * f_root / (f_root - fl)
            xl = self._root
            fl = f_root
            self._root += dx
            f_root = f(self._root)
            self._evaluation_number += 1
            if abs(dx) < x_accuracy or close(f_root, 0.0):
                return self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
