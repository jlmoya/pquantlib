"""Safe (bracketed) Newton 1-D solver with finite-difference derivatives.

# C++ parity: ql/math/solvers1d/finitedifferencenewtonsafe.hpp (v1.42.1)
# class ``FiniteDifferenceNewtonSafe``.

Like NewtonSafe but the derivative is approximated by a one-sided finite
difference between consecutive iterates. No ``f.derivative`` is required.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


class FiniteDifferenceNewtonSafe(Solver1D):
    """Safe Newton with first-order finite-difference derivative."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # Orient the search so that f(xl) < 0
        if self._fx_min < 0.0:
            xl = self._x_min
            xh = self._x_max
        else:
            xh = self._x_min
            xl = self._x_max

        f_root = f(self._root)
        self._evaluation_number += 1
        # first order finite difference derivative
        if self._x_max - self._root < self._root - self._x_min:
            df_root = (self._fx_max - f_root) / (self._x_max - self._root)
        else:
            df_root = (self._fx_min - f_root) / (self._x_min - self._root)

        # xMax_ - xMin_ > 0 verified by base class
        dx = self._x_max - self._x_min
        while self._evaluation_number <= self._max_evaluations:
            f_root_old = f_root
            root_old = self._root
            dx_old = dx
            # Bisect if (out of range || not decreasing fast enough)
            if (((self._root - xh) * df_root - f_root) * ((self._root - xl) * df_root - f_root) > 0.0) or (
                abs(2.0 * f_root) > abs(dx_old * df_root)
            ):
                dx = (xh - xl) / 2.0
                self._root = xl + dx
                # if the root estimate just computed is close to the
                # previous one, we should calculate dfroot at root and
                # xh rather than root and rootold (xl instead of xh would
                # be just as good)
                if close(self._root, root_old, 2500):
                    root_old = xh
                    f_root_old = f(xh)
            else:  # Newton
                dx = f_root / df_root
                self._root -= dx

            # Convergence criterion
            if abs(dx) < x_accuracy:
                return self._root

            f_root = f(self._root)
            self._evaluation_number += 1
            df_root = (f_root_old - f_root) / (root_old - self._root)

            if f_root < 0.0:
                xl = self._root
            else:
                xh = self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
