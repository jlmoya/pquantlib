"""False-position (regula falsi) 1-D solver.

# C++ parity: ql/math/solvers1d/falseposition.hpp (v1.42.1) class
# ``FalsePosition``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


class FalsePosition(Solver1D):
    """False-position root finder — linear interpolation between bracket endpoints."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # Identify the limits so that xl corresponds to the low side
        if self._fx_min < 0.0:
            xl = self._x_min
            fl = self._fx_min
            xh = self._x_max
            fh = self._fx_max
        else:
            xl = self._x_max
            fl = self._fx_max
            xh = self._x_min
            fh = self._fx_min

        while self._evaluation_number <= self._max_evaluations:
            # Increment with respect to latest value
            self._root = xl + (xh - xl) * fl / (fl - fh)
            f_root = f(self._root)
            self._evaluation_number += 1
            if f_root < 0.0:  # Replace appropriate limit
                delta = xl - self._root
                xl = self._root
                fl = f_root
            else:
                delta = xh - self._root
                xh = self._root
                fh = f_root
            # Convergence criterion
            if abs(delta) < x_accuracy or close(f_root, 0.0):
                return self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
