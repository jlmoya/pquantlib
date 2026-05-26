"""Brent 1-D solver.

# C++ parity: ql/math/solvers1d/brent.hpp (v1.42.1) class ``Brent``.

Inverse-quadratic interpolation with bisection fallback. Algorithm inspired
by Press, Teukolsky, Vetterling and Flannery, *Numerical Recipes in C*,
2nd ed., Cambridge University Press.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


def _sign(a: float, b: float) -> float:
    """Magnitude of ``a`` with the sign of ``b``."""
    return abs(a) if b >= 0.0 else -abs(a)


class Brent(Solver1D):
    """Brent's method — robust hybrid of inverse-quadratic and bisection."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:  # noqa: PLR0915  (faithful C++ port — Brent's algorithm is intrinsically one long function)
        # We want to start with root_ (which equals the guess) on
        # one side of the bracket and both xMin_ and xMax_ on the other.
        f_root = f(self._root)
        self._evaluation_number += 1
        if f_root * self._fx_min < 0.0:
            self._x_max = self._x_min
            self._fx_max = self._fx_min
        else:
            self._x_min = self._x_max
            self._fx_min = self._fx_max
        d = self._root - self._x_max
        e = d

        while self._evaluation_number <= self._max_evaluations:
            if (f_root > 0.0 and self._fx_max > 0.0) or (f_root < 0.0 and self._fx_max < 0.0):
                # Rename xMin_, root_, xMax_ and adjust bounds
                self._x_max = self._x_min
                self._fx_max = self._fx_min
                e = d = self._root - self._x_min
            if abs(self._fx_max) < abs(f_root):
                self._x_min = self._root
                self._root = self._x_max
                self._x_max = self._x_min
                self._fx_min = f_root
                f_root = self._fx_max
                self._fx_max = self._fx_min
            # Convergence check
            x_acc1 = 2.0 * QL_EPSILON * abs(self._root) + 0.5 * x_accuracy
            x_mid = (self._x_max - self._root) / 2.0
            if abs(x_mid) <= x_acc1 or close(f_root, 0.0):
                f(self._root)
                self._evaluation_number += 1
                return self._root
            if abs(e) >= x_acc1 and abs(self._fx_min) > abs(f_root):
                # Attempt inverse quadratic interpolation
                s = f_root / self._fx_min
                if close(self._x_min, self._x_max):
                    p = 2.0 * x_mid * s
                    q = 1.0 - s
                else:
                    q = self._fx_min / self._fx_max
                    r = f_root / self._fx_max
                    p = s * (2.0 * x_mid * q * (q - r) - (self._root - self._x_min) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0.0:
                    q = -q  # Check whether in bounds
                p = abs(p)
                min1 = 3.0 * x_mid * q - abs(x_acc1 * q)
                min2 = abs(e * q)
                if 2.0 * p < (min1 if min1 < min2 else min2):
                    e = d  # Accept interpolation
                    d = p / q
                else:
                    d = x_mid  # Interpolation failed, use bisection
                    e = d
            else:
                # Bounds decreasing too slowly, use bisection
                d = x_mid
                e = d
            self._x_min = self._root
            self._fx_min = f_root
            if abs(d) > x_acc1:
                self._root += d
            else:
                self._root += _sign(x_acc1, x_mid)
            f_root = f(self._root)
            self._evaluation_number += 1

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
