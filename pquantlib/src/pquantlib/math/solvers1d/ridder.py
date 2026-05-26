"""Ridder 1-D solver.

# C++ parity: ql/math/solvers1d/ridder.hpp (v1.42.1) class ``Ridder``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.

Note the C++ comment: tests on Black-Scholes implied volatility show that
Ridder's solver actually provides ~100x tighter accuracy than the user-
requested tolerance — the impl divides the requested accuracy by 100
internally. This is faithfully preserved here.
"""

from __future__ import annotations

import math
import sys

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D

# C++ parity: QL_MIN_REAL = -std::numeric_limits<double>::max() (qldefines.hpp).
# Used as a sentinel root for the first iteration's distance check.
_QL_MIN_REAL: float = -sys.float_info.max


def _sign(a: float, b: float) -> float:
    """Magnitude of ``a`` with the sign of ``b``."""
    return abs(a) if b >= 0.0 else -abs(a)


class Ridder(Solver1D):
    """Ridder's method — exponential interpolation between bracket midpoint and endpoints."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # See class docstring: tighten by 100x to match C++ semantics.
        x_acc = x_accuracy / 100.0

        # Any highly unlikely value, to simplify logic below (matches C++ QL_MIN_REAL).
        self._root = _QL_MIN_REAL

        while self._evaluation_number <= self._max_evaluations:
            x_mid = 0.5 * (self._x_min + self._x_max)
            # First of two function evaluations per iteration
            fx_mid = f(x_mid)
            self._evaluation_number += 1
            s = math.sqrt(fx_mid * fx_mid - self._fx_min * self._fx_max)
            if close(s, 0.0):
                f(self._root)
                self._evaluation_number += 1
                return self._root
            # Updating formula
            sign_term = 1.0 if self._fx_min >= self._fx_max else -1.0
            next_root = x_mid + (x_mid - self._x_min) * (sign_term * fx_mid / s)
            if abs(next_root - self._root) <= x_acc:
                f(self._root)
                self._evaluation_number += 1
                return self._root

            self._root = next_root
            # Second of two function evaluations per iteration
            f_root = f(self._root)
            self._evaluation_number += 1
            if close(f_root, 0.0):
                return self._root

            # Bookkeeping to keep the root bracketed on next iteration
            if _sign(fx_mid, f_root) != fx_mid:
                self._x_min = x_mid
                self._fx_min = fx_mid
                self._x_max = self._root
                self._fx_max = f_root
            elif _sign(self._fx_min, f_root) != self._fx_min:
                self._x_max = self._root
                self._fx_max = f_root
            elif _sign(self._fx_max, f_root) != self._fx_max:
                self._x_min = self._root
                self._fx_min = f_root
            else:
                qassert.fail("never get here.")

            if abs(self._x_max - self._x_min) <= x_acc:
                f(self._root)
                self._evaluation_number += 1
                return self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
