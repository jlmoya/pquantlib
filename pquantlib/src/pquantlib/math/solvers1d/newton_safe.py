"""Safe (bracketed) Newton 1-D solver.

# C++ parity: ql/math/solvers1d/newtonsafe.hpp (v1.42.1) class ``NewtonSafe``.

Algorithm inspired by Press, Teukolsky, Vetterling and Flannery,
*Numerical Recipes in C*, 2nd ed., Cambridge University Press.

The integrand must expose ``derivative(x) -> float``. Python translation
calls ``getattr(f, "derivative")`` and raises ``LibraryException`` if
missing (cf. C++ ``Null<Real>`` sentinel check).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.solvers1d.solver_1d import RealFunction, Solver1D


def derivative_of(f: RealFunction, x: float) -> float:
    """Call ``f.derivative(x)`` with a clear error if absent.

    # C++ parity: newtonsafe.hpp asserts ``dfroot != Null<Real>()`` and
    # raises "NewtonSafe requires function's derivative".
    """
    deriv = getattr(f, "derivative", None)
    qassert.require(callable(deriv), "NewtonSafe requires function's derivative")
    return float(deriv(x))  # type: ignore[misc]


class NewtonSafe(Solver1D):
    """Safe Newton — Newton step when productive, bisection when not."""

    def _solve_impl(self, f: RealFunction, x_accuracy: float) -> float:
        # Orient the search so that f(xl) < 0
        if self._fx_min < 0.0:
            xl = self._x_min
            xh = self._x_max
        else:
            xh = self._x_min
            xl = self._x_max

        # the "stepsize before last"; xMax_-xMin_ > 0 is verified by caller.
        dx_old = self._x_max - self._x_min
        # and the last step
        dx = dx_old

        f_root = f(self._root)
        df_root = derivative_of(f, self._root)
        self._evaluation_number += 1

        while self._evaluation_number <= self._max_evaluations:
            # Bisect if (out of range || not decreasing fast enough)
            if (((self._root - xh) * df_root - f_root) * ((self._root - xl) * df_root - f_root) > 0.0) or (
                abs(2.0 * f_root) > abs(dx_old * df_root)
            ):
                dx_old = dx
                dx = (xh - xl) / 2.0
                self._root = xl + dx
            else:
                dx_old = dx
                dx = f_root / df_root
                self._root -= dx
            # Convergence criterion
            if abs(dx) < x_accuracy:
                f(self._root)
                self._evaluation_number += 1
                return self._root
            f_root = f(self._root)
            df_root = derivative_of(f, self._root)
            self._evaluation_number += 1
            if f_root < 0.0:
                xl = self._root
            else:
                xh = self._root

        qassert.fail(f"maximum number of function evaluations ({self._max_evaluations}) exceeded")
